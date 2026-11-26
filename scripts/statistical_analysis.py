import argparse, json
from pathlib import Path
import numpy as np
from paired_stats import influence, variances, interval
from paired_comparison import combine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dependency-data", type=Path, required=True)
    p.add_argument(
        "--terms",
        type=Path,
        required=True,
        help="JSON list: key,sign,prediction file; terms preserve same seed pairing",
    )
    p.add_argument("--family-size", type=int, default=12)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    with np.load(a.dependency_data, allow_pickle=False) as z:
        d = dict(z)
    terms = json.loads(a.terms.read_text(encoding="utf-8"))
    models = {}
    for t in terms:
        path = Path(t["file"])
        path = path if path.is_absolute() else a.terms.parent / path
        with np.load(path, allow_pickle=False) as z:
            q = dict(z)
        rows = q.get("rows", q.get("row_id"))
        lut = {int(r): i for i, r in enumerate(d["rows"])}
        if (
            rows is None
            or len(set(map(int, rows))) != len(rows)
            or set(map(int, rows)) != set(lut)
        ):
            raise ValueError("Prediction/dependency row identity mismatch")
        reference = d["y"][[lut[int(r)] for r in rows]]
        gold = q.get("y", q.get("gold_label", q.get("reference_label", reference)))
        if not np.array_equal(reference, gold):
            raise ValueError("Prediction reference labels differ")
        if (
            t.get("seed") is not None
            and "seed" in q
            and not np.all(q["seed"] == int(t["seed"]))
        ):
            raise ValueError("Seed identity mismatch")
        fold = q.get("outer_fold", q.get("fold"))
        if (
            fold is not None
            and "fold" in d
            and not np.array_equal(fold, d["fold"][[lut[int(r)] for r in rows]])
        ):
            raise ValueError("Outer-fold identity mismatch")
        if "label_order" in q and list(q["label_order"]) != ["=", ">", "<", "~"]:
            raise ValueError("Label order mismatch")
        prob = q[t.get("probability_key", "probabilities")]
        if (
            prob.shape != (len(rows), 4)
            or not np.isfinite(prob).all()
            or prob.min() < 0
            or not np.allclose(prob.sum(1), 1, atol=1e-6)
        ):
            raise ValueError("Invalid probability matrix")
        models[t["key"]] = dict(rows=rows, y=gold, probabilities=prob)
    est, psi = combine(
        d["y"], d["rows"], models, [(t["key"], t["sign"]) for t in terms], influence
    )
    v = variances(psi, d)["two_way"]
    out = dict(
        estimate=est,
        terms=terms,
        family_size=a.family_size,
        **interval(est, v, 0.05 / a.family_size),
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
