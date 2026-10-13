"""Held-out answer NLL/perplexity and optional math/code accuracy, with retrieval disabled."""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from ..data.quality import read_rows
from ..inference.engine import Engine
from ..safeio import atomic_json, load_torch
from .posttrain import logprob
from .verifiers import verify


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--generate", action="store_true")
    a = ap.parse_args(argv)
    ck = load_torch(a.checkpoint, a.device)
    if ck is None:
        ap.error("checkpoint not found")
    eng = Engine(a.device, checkpoint=ck)
    rows = list(read_rows(a.data))
    if not rows:
        ap.error("empty evaluation data")
    nll, tokens, correct, checked, details = 0.0, 0, 0, 0, []
    t0 = time.perf_counter()
    with torch.no_grad():
        for row in rows:
            context = row.get("context") or [{"role": "user", "content": row["prompt"]}]
            if "response" in row:
                lp, count = logprob(eng.model, context, row["response"], a.device)
                nll -= lp.item()
                tokens += count.item()
            if a.generate:
                response = "".join(
                    v
                    for k, v in eng.generate(context, max_new=a.max_new, temperature=0, use_memory=False)
                    if k == "text"
                )
                rec = {"prompt": row["prompt"], "response": response}
                if row.get("kind") in ("math", "code"):
                    rec["reward"] = verify(row, response)
                    correct += rec["reward"]
                    checked += 1
                details.append(rec)
    report = {
        "checkpoint": a.checkpoint,
        "context": eng.model.c.block_size,
        "device": a.device,
        "examples": len(rows),
        "answer_tokens": tokens,
        "answer_nll": nll / tokens if tokens else None,
        "answer_perplexity": math.exp(min(nll / tokens, 80)) if tokens else None,
        "verified_accuracy": correct / checked if checked else None,
        "verified_examples": checked,
        "seconds": time.perf_counter() - t0,
        "generations": details,
    }
    atomic_json(Path(a.out), report)
    print(json.dumps({k: v for k, v in report.items() if k != "generations"}, indent=2))


if __name__ == "__main__":
    main()
