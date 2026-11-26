"""Build fit-scoped A64, B7, R1 and Q63 arrays from explicit input files."""

import argparse, json, hashlib, warnings
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from method import fit_a64, text_matrices, q63
from r1 import relation_features
from q63_kernel import summarize
from threadpoolctl import threadpool_limits


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return dict(z)


def build(pool, fit, ev, y, vectors, projector=None, device="cpu"):
    if len(fit) != len(y) or len(set(pool["groups"][fit]) & set(pool["groups"][ev])):
        raise ValueError("Fit/evaluation identity violation")
    if projector is not None and not np.array_equal(
        projector["fit_rows"], pool["rows"][fit]
    ):
        raise ValueError("Projector belongs to another fit scope")
    for side in ["source", "target"]:
        if len(vectors[side + "_embeddings"]) != len(pool[side + "_keys"]):
            raise ValueError("Name-vector endpoint count mismatch")
        if side + "_keys" in vectors and not np.array_equal(
            vectors[side + "_keys"], pool[side + "_keys"]
        ):
            raise ValueError("Name-vector URI/snapshot identity mismatch")
    fields = pool["text_bank"][pool["text_indices"]]
    if projector is None:
        A = fit_a64(fields, fit, y, device)
    else:
        xx = text_matrices(fields, fit)
        raw = xx @ projector["L"] - projector["mean"] @ projector["L"]
        A = StandardScaler().fit(raw[fit]).transform(raw).astype(np.float32)
    threadpool_limits(4)
    rawR, R, degree, _, blocks = relation_features(pool, fit, ev, y)
    pos = np.r_[fit, ev]
    n = len(fit)
    B = (
        StandardScaler()
        .fit(pool["raw7"][fit])
        .transform(pool["raw7"][pos])
        .astype(np.float32)
    )
    emb = [vectors["source_embeddings"], vectors["target_embeddings"]]
    extra = []
    own = [pool["u"], pool["v"]]
    edges = [
        {int(u): np.flatnonzero(own[s][fit] == u) for u in np.unique(own[s][fit])}
        for s in [0, 1]
    ]
    # Preserve the original grouped GEMM shape/order, not just algebraic equivalence:
    # scalar dot products can change FP32 cosines before standardization.
    qs = []
    ss = []
    es = []
    hs = []
    N = len(pos)
    qb = np.r_[blocks, np.full(len(ev), -1)]
    for side in [0, 1]:
        Q = np.zeros((N, 3, 4))
        S = Q.copy()
        E = np.zeros((N, 3))
        H = E.copy()
        e = emb[1 - side]
        nb = own[1 - side][fit]
        qnode = own[1 - side][pos]
        for node in np.unique(own[side][pos]):
            ee = edges[side].get(int(node), np.array([], int))
            if not len(ee):
                continue
            qq = np.flatnonzero(own[side][pos] == node)
            for lo in range(0, len(qq), 128):
                ix = qq[lo : lo + 128]
                allow = (qb[ix, None] < 0) | (qb[ix, None] != blocks[ee][None, :])
                sim = np.clip(e[qnode[ix]] @ e[nb[ee]].T, -1, 1)
                _, q, s, ess, h = summarize(sim, allow, y[ee])
                Q[ix] = q
                S[ix] = s
                E[ix] = ess
                H[ix] = h
        qs.append(Q)
        ss.append(S)
        es.append(E)
        hs.append(H)
    midpoint = 0.5 * (qs[0] + qs[1])
    js = 0.5 * sum(
        (q * (np.log(q.clip(1e-30)) - np.log(midpoint.clip(1e-30)))).sum(2) for q in qs
    )
    js[(degree == 0).any(1)] = 0
    raw = np.c_[
        qs[0].reshape(N, 12),
        qs[1].reshape(N, 12),
        ss[0].reshape(N, 12),
        ss[1].reshape(N, 12),
        es[0],
        es[1],
        hs[0],
        hs[1],
        js,
    ]
    mu = raw[:n].mean(0)
    sd = raw[:n].std(0)
    sd[sd < 1e-12] = 1
    Q = ((raw - mu) / sd).astype(np.float32)
    full = np.c_[A[pos], B, R, Q].astype(np.float32)
    # Exact declared fixed-C1 LR-AB recipe; this is scoped fitting, not a full-pool prior.
    for maxiter in [3000, 10000]:
        model = LogisticRegression(
            C=1.0,
            penalty="l2",
            fit_intercept=True,
            class_weight=None,
            solver="lbfgs",
            tol=1e-6,
            max_iter=maxiter,
            n_jobs=1,
            random_state=2026111601,
        )
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            model.fit(full[:n, :71].astype(np.float64), y)
        if not any(issubclass(w.category, ConvergenceWarning) for w in ws):
            break
    else:
        raise ValueError(
            "Common LR fallback did not converge within declared numerical budget"
        )
    return dict(
        x_fit=full[:n],
        x_eval=full[n:],
        y_fit=y,
        fit_rows=pool["rows"][fit],
        eval_rows=pool["rows"][ev],
        degree_eval=degree[n:],
        raw_Q63=raw,
        raw_R1=rawR,
        Q_mean=mu,
        Q_scale=sd,
        fallback_probability=model.predict_proba(full[n:, :71].astype(np.float64)),
        fallback_coef=model.coef_,
        fallback_intercept=model.intercept_,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, type=Path)
    p.add_argument("--fit-labels", required=True, type=Path)
    p.add_argument("--name-vectors", required=True, type=Path)
    p.add_argument("--projector", type=Path)
    p.add_argument("--validation-labels", type=Path)
    p.add_argument("--outer", required=True, type=int)
    p.add_argument("--inner", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    pool = load(a.pool)
    fl = load(a.fit_labels)
    ev = (
        np.flatnonzero(pool["outer_fold"] == a.outer)
        if a.inner is None
        else np.flatnonzero(pool["inner_fold"][a.outer] == a.inner)
    )
    result = build(
        pool,
        fl["positions"],
        ev,
        fl["y"],
        load(a.name_vectors),
        load(a.projector) if a.projector else None,
        a.device,
    )
    if a.validation_labels:
        if a.inner is None:
            raise ValueError("Validation labels may only bind an inner scope")
        val = load(a.validation_labels)
        lut = {int(r): i for i, r in enumerate(val["rows"])}
        result["y_eval"] = val["y"][[lut[int(r)] for r in result["eval_rows"]]]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("xb") as f:
        np.savez_compressed(f, **result)
    print(
        json.dumps(
            dict(
                scope=[a.outer, a.inner],
                fit=len(fl["y"]),
                eval=len(ev),
                from_retained_projector=bool(a.projector),
            )
        )
    )


if __name__ == "__main__":
    main()
