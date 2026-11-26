"""Prepare ordered endpoint names and assemble their encoded vector cache."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit import concepts


def load_pool(path):
    with np.load(path, allow_pickle=False) as data:
        return {side: data[f"{side}_keys"].copy() for side in ("source", "target")}


def prepare(pool_path, source_rdf, target_rdf, out):
    keys = load_pool(pool_path)
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for side, rdf in (("source", source_rdf), ("target", target_rdf)):
        with rdf.open("rb") as stream:
            snapshot = hashlib.file_digest(stream, "sha256").hexdigest()
        records = concepts(rdf)
        names = []
        for key in keys[side]:
            digest, uri = str(key).split("|", 1)
            if digest != snapshot or uri not in records:
                raise ValueError(f"{side} endpoint does not match the RDF snapshot")
            names.append(records[uri]["name"])
        (out / f"{side}_names.json").write_text(
            json.dumps(names, ensure_ascii=False), encoding="utf-8"
        )
        index[side] = keys[side].tolist()
    (out / "endpoint_keys.json").write_text(json.dumps(index), encoding="utf-8")


def pack(pool_path, names_dir, source_vectors, target_vectors, out):
    keys = load_pool(pool_path)
    recorded = json.loads(
        (names_dir / "endpoint_keys.json").read_text(encoding="utf-8")
    )
    result = {}
    for side, path in (("source", source_vectors), ("target", target_vectors)):
        if recorded[side] != keys[side].tolist():
            raise ValueError(f"{side} pool identity changed after name preparation")
        names = json.loads(
            (names_dir / f"{side}_names.json").read_text(encoding="utf-8")
        )
        with np.load(path, allow_pickle=False) as encoded:
            if encoded["names"].tolist() != names:
                raise ValueError(f"{side} encoded names are out of order")
            vectors = np.asarray(encoded["embeddings"], dtype=np.float32)
        if vectors.shape != (len(keys[side]), 384) or not np.isfinite(vectors).all():
            raise ValueError(f"Invalid {side} embedding array")
        result[f"{side}_keys"] = keys[side]
        result[f"{side}_embeddings"] = vectors
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        np.savez_compressed(stream, **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    names = commands.add_parser("prepare", help="Write endpoint names in pool order")
    names.add_argument("--pool", type=Path, required=True)
    names.add_argument("--source-rdf", type=Path, required=True)
    names.add_argument("--target-rdf", type=Path, required=True)
    names.add_argument("--out", type=Path, required=True)
    vectors = commands.add_parser("pack", help="Assemble encoded endpoint arrays")
    vectors.add_argument("--pool", type=Path, required=True)
    vectors.add_argument("--names-dir", type=Path, required=True)
    vectors.add_argument("--source-vectors", type=Path, required=True)
    vectors.add_argument("--target-vectors", type=Path, required=True)
    vectors.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.pool, args.source_rdf, args.target_rdf, args.out)
    else:
        pack(
            args.pool,
            args.names_dir,
            args.source_vectors,
            args.target_vectors,
            args.out,
        )


if __name__ == "__main__":
    main()
