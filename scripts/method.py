"""Text, catalogue and relational features, smoke checks and saved-result replay."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from name_features import jaccard, norm_name
from proof import Network
from q63_kernel import summarize
from r1 import relation_features
from paired_stats import cm, macro

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as file:
        return dict(file)


def name_and_depth(source, target, source_depth, target_depth, maxima, vectors):
    left, right = norm_name(source), norm_name(target)
    ds, dt = source_depth / maxima[0], target_depth / maxima[1]
    return np.array(
        [
            ds,
            dt,
            ds - dt,
            abs(ds - dt),
            jaccard(set(left.split()), set(right.split())),
            jaccard(set(zip(left, left[1:])), set(zip(right, right[1:]))),
            np.dot(vectors[0], vectors[1]),
        ],
        dtype=np.float32,
    )


def q63(side_cosines, side_labels):
    summaries = []
    for cos, labels in zip(side_cosines, side_labels, strict=True):
        cos = np.asarray(cos)
        labels = np.asarray(labels, dtype=np.int64)
        allow = np.ones((1, len(cos)), dtype=bool)
        summaries.append(summarize(np.clip(cos, -1, 1)[None, :], allow, labels))
    (_, left, sl, el, hl), (_, right, sr, er, hr) = summaries
    midpoint = (left + right) / 2
    log = lambda arr: np.log(arr.clip(1e-30))
    js = (
        left * (log(left) - log(midpoint)) + right * (log(right) - log(midpoint))
    ).sum(2) / 2
    if not all(len(x) for x in side_labels):
        js[:] = 0
    return np.concatenate(
        [v.ravel() for v in [left, right, sl, sr, el, er, hl, hr, js]]
    )


def text_matrices(fields, fit):
    fit_fields = np.unique(fields[fit])
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=12000,
        sublinear_tf=True,
        norm="l2",
        lowercase=True,
        dtype=np.float64,
    )
    tfidf.fit(fit_fields[fit_fields != ""])
    return sparse.hstack(
        [tfidf.transform(fields[:, i]) for i in range(4)], format="csr"
    )


def fit_a64(fields, fit, fit_labels, device):
    import torch
    from projection import solve
    from threadpoolctl import threadpool_limits

    threadpool_limits(4)
    torch.set_num_threads(4)
    all_text = text_matrices(fields, fit)
    train = all_text[fit]
    svd = TruncatedSVD(
        n_components=64, n_iter=7, n_oversamples=10, random_state=20261300
    )
    svd.fit(train)
    mean = np.asarray(train.mean(0)).ravel()
    centered_norm = float(train.multiply(train).sum() - len(fit) * (mean @ mean))

    def csr(matrix):
        matrix = matrix.tocsr()
        matrix.sort_indices()
        return torch.sparse_csr_tensor(
            matrix.indptr,
            matrix.indices,
            matrix.data,
            size=matrix.shape,
            device=device,
            dtype=torch.float64,
        )

    def tensor(value):
        return torch.as_tensor(value, device=device)

    projection, _, _, _ = solve(
        csr(train),
        csr(train.T),
        tensor(mean),
        float(centered_norm),
        tensor(fit_labels),
        tensor(svd.components_.T),
    )
    basis = projection.detach().cpu().numpy()
    raw = all_text @ basis - mean @ basis
    return StandardScaler().fit(raw[fit]).transform(raw).astype(np.float32)


def check_scores(directory):
    with (ROOT / "manifests/main_prediction_hashes.csv").open(
        encoding="utf-8-sig"
    ) as file:
        identities = list(csv.DictReader(file))
    with (ROOT / "results/main_prediction_rescore.csv").open(
        encoding="utf-8-sig"
    ) as file:
        expected = list(csv.DictReader(file))
    for identity in identities:
        path = directory / identity["prediction_file"]
        if file_hash(path) != identity["sha256"]:
            raise ValueError(f"Prediction identity changed: {path.name}")
        saved = load_npz(path)
        y = saved.get("y", saved.get("gold_label"))
        for row in expected:
            if (row["task"], row["seed"]) != (identity["task"], identity["seed"]):
                continue
            arm = row["arm"]
            key = arm if row["contract"] == "operational" else arm + "_raw"
            if key not in saved:
                key = "raw_" + arm
            values = saved[key]
            prediction = values.argmax(1) if values.ndim == 2 else values
            np.testing.assert_allclose(
                macro(cm(y, prediction)), float(row["macro_f1"]), atol=1e-13, rtol=0
            )
            np.testing.assert_allclose(
                np.mean(y == prediction), float(row["accuracy"]), atol=1e-13, rtol=0
            )
    print(
        f"Verified {len(expected)} main metrics from {len(identities)} prediction files."
    )


def smoke():
    # Fit rows share one endpoint with the evaluation row. They share a block with
    # each other, so no fit row may see that block.
    pool = dict(
        u=np.array([0, 0, 0]),
        v=np.array([0, 1, 2]),
        groups=np.array(["g1", "g2", "g3"]),
        source_keys=["s"],
        target_keys=["a", "b", "c"],
    )
    raw, _, degree, _, _ = relation_features(
        pool, np.array([0, 1]), np.array([2]), np.array([1, 3])
    )
    np.testing.assert_array_equal(degree, [(0, 0), (0, 0), (2, 0)])
    np.testing.assert_array_equal(raw[2, 4:8], [0, 0.5, 0, 0.5])
    feat = q63([[0.9, 0.8], []], [[0, 3], []])
    assert len(feat) == 63 and feat[4] > feat[7] and not feat[-3:].any()
    assert np.isclose(feat[4:8].sum(), 1)
    b7 = name_and_depth("CABLE 1", "cable 2", 1, 2, [2, 2], [[1, 0], [1, 0]])
    np.testing.assert_allclose(b7[[0, 1, 2, 3, 4, 6]], [0.5, 1, -0.5, 0.5, 1 / 3, 1])
    proof = Network(["a", "b", "c"], [], [("a", "b", ">", 0), ("b", "c", "=", 1)])
    assert proof.query("a", "c")["potential_mask"] & 2
    print("Four synthetic checks passed: B7, R1 scope, Q63 empty side, proof fragment.")


def train_head(input_path, output, arm, lr, epochs, seed, device):
    import torch
    from net import Net

    scope = load_npz(input_path)
    xfit, xeval = [
        np.asarray(scope[k], dtype=np.float32).copy() for k in ["x_fit", "x_eval"]
    ]
    if xfit.shape[1] != 146 or xeval.shape[1] != 146:
        raise ValueError("Expected standardized 146-coordinate input.")
    cutoff = {"AB": 71, "ABR": 83, "Full": 146}[arm]
    xfit[:, cutoff:] = 0
    xeval[:, cutoff:] = 0
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    net = Net(arm).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.001)
    xx = torch.from_numpy(xfit).to(device)
    yy = torch.as_tensor(scope["y_fit"], dtype=torch.long, device=device)
    for ep in range(1, epochs + 1):
        order = np.random.default_rng(np.random.SeedSequence([seed, ep])).permutation(
            len(yy)
        )
        for start in range(0, len(yy), 4096):
            ii = torch.as_tensor(order[start : start + 4096], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                net(xx[ii], None, None, None), yy[ii]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
    net.eval()
    with torch.inference_mode():
        raw = np.concatenate(
            [
                torch.softmax(
                    net(
                        torch.from_numpy(xeval[i : i + 4096]).to(device),
                        None,
                        None,
                        None,
                    ),
                    1,
                )
                .cpu()
                .numpy()
                for i in range(0, len(xeval), 4096)
            ]
        )
    operational = raw.astype(np.float64)
    zero = scope["degree_eval"].sum(axis=1) == 0
    operational[zero] = scope["fallback_probability"][zero]
    with output.open("xb") as file:
        np.savez_compressed(
            file,
            raw_probabilities=raw,
            probabilities=operational,
            rows=scope["eval_rows"],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "smoke", help="Independent synthetic checks with no model training"
    )
    score = commands.add_parser(
        "check-scores", help="Verify saved predictions, without model execution"
    )
    score.add_argument("predictions", type=Path)
    train = commands.add_parser(
        "train-head", help="Run a selected single-scope recipe on prepared features"
    )
    train.add_argument("inputs", type=Path)
    train.add_argument("output", type=Path)
    train.add_argument("--arm", choices=["AB", "ABR", "Full"], required=True)
    train.add_argument("--lr", type=float, choices=[0.0003, 0.001], required=True)
    train.add_argument("--epochs", type=int, choices=range(5, 81, 5), required=True)
    train.add_argument("--seed", type=int, default=2026111601)
    train.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/method.json").read_text(encoding="utf-8"))
    assert config["label_order"] == ["=", ">", "<", "~"]
    if args.command == "smoke":
        smoke()
    elif args.command == "check-scores":
        check_scores(args.predictions)
    else:
        train_head(
            args.inputs,
            args.output,
            args.arm,
            args.lr,
            args.epochs,
            args.seed,
            args.device,
        )


if __name__ == "__main__":
    main()
