"""Restore declared row identities from independently parsed RDF and retained split metadata."""

from pathlib import Path
import argparse, csv, json, hashlib
import numpy as np
from audit import assertions, concepts, clean, LABELS, sha
from name_features import parent_context
from method import name_and_depth


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--task", required=True)
    p.add_argument("--name-vectors", type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    contract = Path(__file__).resolve().parents[1] / "manifests/data_hashes.csv"
    with contract.open(encoding="utf-8-sig", newline="") as f:
        hashes = list(csv.DictReader(f))
    for filename in ["source.rdf", "target.rdf", "reference.rdf"]:
        records = [r for r in hashes if r["task"] == a.task and r["role"] == filename]
        if len(records) != 1 or sha(a.input / filename) != records[0]["sha256"]:
            raise ValueError("Declared formal RDF snapshot hash mismatch: " + filename)
    rr = assertions(a.input / "reference.rdf", a.task)
    s = concepts(a.input / "source.rdf")
    t = concepts(a.input / "target.rdf")
    good, issues, summary = clean(rr, s, t)
    allowed = {(r["source"], r["target"], r["relation"]) for r in good}
    with a.manifest.open(encoding="utf-8-sig", newline="") as f:
        manifest = list(csv.DictReader(f))
    assert all(
        (
            rr[int(r["cell_index"])]["source_uri"],
            rr[int(r["cell_index"])]["target_uri"],
            rr[int(r["cell_index"])]["relation"],
        )
        in allowed
        for r in manifest
    )
    selected = sorted(
        [r for r in manifest if int(r["outer"]) >= 0], key=lambda r: int(r["row_id"])
    )
    raw = [rr[int(r["cell_index"])] for r in selected]
    rows = np.array([int(r["row_id"]) for r in selected])
    y = np.array([LABELS.index(r["relation"]) for r in raw])
    source_hash = sha(a.input / "source.rdf")
    target_hash = sha(a.input / "target.rdf")
    source_keys, u = np.unique(
        [source_hash + "|" + r["source_uri"] for r in raw], return_inverse=True
    )
    target_keys, v = np.unique(
        [target_hash + "|" + r["target_uri"] for r in raw], return_inverse=True
    )
    fields = np.array(
        [
            [
                s[r["source_uri"]]["name"],
                parent_context(s, r["source_uri"]),
                t[r["target_uri"]]["name"],
                parent_context(t, r["target_uri"]),
            ]
            for r in raw
        ]
    )
    bank, idx = np.unique(fields, return_inverse=True)
    pool = dict(
        rows=rows,
        groups=np.array([r["group"] for r in selected]),
        outer_fold=np.array([int(r["outer"]) for r in selected]),
        inner_fold=np.array(
            [[int(r[f"inner{o}"]) for r in selected] for o in range(5)]
        ),
        source_keys=source_keys,
        target_keys=target_keys,
        u=u,
        v=v,
        text_bank=bank,
        text_indices=idx.reshape(-1, 4),
    )
    if a.name_vectors:
        with np.load(a.name_vectors, allow_pickle=False) as z:
            ve = dict(z)
        if len(ve["source_embeddings"]) != len(source_keys) or len(
            ve["target_embeddings"]
        ) != len(target_keys):
            raise ValueError(
                "Name vectors must follow the exact sorted snapshot+URI key order"
            )
        mx = [max(r["depth"] for r in s.values()), max(r["depth"] for r in t.values())]
        pool["raw7"] = np.stack(
            [
                name_and_depth(
                    s[r["source_uri"]]["name"],
                    t[r["target_uri"]]["name"],
                    s[r["source_uri"]]["depth"],
                    t[r["target_uri"]]["depth"],
                    mx,
                    [ve["source_embeddings"][u[i]], ve["target_embeddings"][v[i]]],
                )
                for i, r in enumerate(raw)
            ]
        )
    np.savez_compressed(a.out / f"{a.task}__pool_public.npz", **pool)
    np.savez_compressed(a.out / f"{a.task}__outer_truth.npz", rows=rows, y=y)
    for o in range(5):
        for inner in [None, 0, 1, 2]:
            ft = (
                np.flatnonzero(pool["outer_fold"] != o)
                if inner is None
                else np.flatnonzero(
                    (pool["outer_fold"] != o) & (pool["inner_fold"][o] != inner)
                )
            )
            tag = f"o{o}" + ("" if inner is None else f"i{inner}")
            np.savez_compressed(
                a.out / f"{a.task}__{tag}__fit_labels.npz",
                positions=ft,
                rows=rows[ft],
                y=y[ft],
            )
            if inner is not None:
                ev = np.flatnonzero(pool["inner_fold"][o] == inner)
                np.savez_compressed(
                    a.out / f"{a.task}__{tag}__validation_labels.npz",
                    positions=ev,
                    rows=rows[ev],
                    y=y[ev],
                )
    (a.out / "reconstruction.json").write_text(
        json.dumps(
            dict(
                raw=summary,
                formal=len(rows),
                source_sha256=source_hash,
                target_sha256=target_hash,
                reference_sha256=sha(a.input / "reference.rdf"),
                roles_and_groups="retained metadata, not rediscovered",
                raw7=(
                    "computed with explicitly supplied cached name vectors"
                    if a.name_vectors
                    else "not built; supply --name-vectors before feature building"
                ),
            )
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
