"""Frozen local ONNX encoder, explicit model/tokenizer files and ordered name list."""

import argparse, json, hashlib
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--names", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--tokenizer", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    import onnxruntime as ort
    from tokenizers import Tokenizer

    names = json.loads(a.names.read_text(encoding="utf-8"))
    tok = Tokenizer.from_file(str(a.tokenizer))
    tok.enable_truncation(max_length=256)
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 4
    s = ort.InferenceSession(str(a.model), opt, providers=["CPUExecutionProvider"])
    out = []
    for i in range(0, len(names), 32):
        batch = tok.encode_batch(names[i : i + 32])
        mask = np.array([q.attention_mask for q in batch], dtype=np.int64)
        feed = dict(
            input_ids=np.array([q.ids for q in batch], dtype=np.int64),
            attention_mask=mask,
            token_type_ids=np.array([q.type_ids for q in batch], dtype=np.int64),
        )
        v = s.run(None, {q.name: feed[q.name] for q in s.get_inputs()})[0]
        if v.ndim == 3:
            v = (v * mask[:, :, None]).sum(1) / np.maximum(
                mask.sum(1, keepdims=True), 1
            )
        out.append(
            (v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)).astype(
                np.float32
            )
        )
    np.savez_compressed(a.out, names=np.array(names), embeddings=np.concatenate(out))
    print("Encoded names in input order; output vectors are L2-normalized")


if __name__ == "__main__":
    main()
