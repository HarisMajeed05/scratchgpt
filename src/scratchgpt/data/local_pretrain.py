"""Convert quality-filtered local {text, split} JSONL into bounded uint16 pretraining shards."""

import argparse
from pathlib import Path

import numpy as np

from ..safeio import atomic_write
from ..tokenizer import tokenizer
from ..tokens import EOT
from .quality import read_rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="quality-data output directory or JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-tokens", type=int, default=1000000)
    a = ap.parse_args(argv)
    if a.shard_tokens < 100:
        ap.error("shard-tokens must be >= 100")
    out, source = Path(a.out), Path(a.input)
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("*.bin")):
        ap.error("local import needs fresh output directory")
    files = [source / "train.jsonl", source / "val.jsonl"] if source.is_dir() else [source]
    buffers, numbers = {"train": [], "val": []}, {"train": 0, "val": 0}

    def flush(kind):
        values = np.asarray(buffers[kind], dtype=np.uint16)
        atomic_write(out / f"{kind}_{numbers[kind]:05d}.bin", lambda f: f.write(values.tobytes()))
        buffers[kind].clear()
        numbers[kind] += 1

    for file in files:
        for row in read_rows(file):
            kind = row.get("split", file.stem if file.stem in buffers else "train")
            if kind not in buffers:
                ap.error("split must be train or val")
            ids = tokenizer.encode(row["text"]) + [EOT]
            for start in range(0, len(ids), a.shard_tokens):
                buffers[kind].extend(ids[start : start + a.shard_tokens])
                if len(buffers[kind]) >= a.shard_tokens:
                    flush(kind)
    for kind in buffers:
        if buffers[kind]:
            flush(kind)
    print("wrote shards", numbers)


if __name__ == "__main__":
    main()
