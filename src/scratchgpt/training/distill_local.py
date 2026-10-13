"""One local pipeline: teacher generates answers, quality filtering splits them, student predicts them with CE.

Teacher and student use different tokenizers, so distillation is via decoded answer text, not incompatible logit KL.
"""

import argparse
from pathlib import Path

from ..data import distill, quality
from . import posttrain


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--teacher", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--teacher-device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--student-device", default="cpu")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--teacher-int8", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    raw, clean, trained = work / "teacher.jsonl", work / "quality", work / "student"
    args = [
        "--prompts",
        a.prompts,
        "--out",
        str(raw),
        "--teacher",
        a.teacher,
        "--revision",
        a.revision,
        "--device",
        a.teacher_device,
        "--max-new",
        str(a.max_new),
    ]
    if a.offline:
        args.append("--offline")
    if a.teacher_int8:
        args.append("--int8")
    if a.verify:
        args.append("--verify")
    distill.main(args)
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    quality.main(["--input", str(raw), "--out", str(clean)])
    args = [
        "--mode",
        "distill",
        "--base",
        a.base,
        "--data",
        str(clean / "train.jsonl"),
        "--out",
        str(trained),
        "--steps",
        str(a.steps),
        "--device",
        a.student_device,
    ]
    if (clean / "val.jsonl").stat().st_size:
        args += ["--val", str(clean / "val.jsonl")]
    posttrain.main(args)


if __name__ == "__main__":
    main()
