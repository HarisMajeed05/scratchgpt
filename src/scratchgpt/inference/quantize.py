"""Post-training CPU dynamic INT8 linear weights. Embeddings/norms/KV cache remain floating point."""

import argparse
from pathlib import Path

import torch

from ..model.gpt import GPT
from ..safeio import atomic_write, load_torch


def quantize_model(model):
    model = model.cpu().float().eval()
    # Keep tied output embedding/head in FP32: avoid duplication and large output quality shifts.
    names = {name for name, m in model.named_modules() if isinstance(m, torch.nn.Linear) and name != "head"}
    return torch.ao.quantization.quantize_dynamic(model, qconfig_spec=names, dtype=torch.qint8, inplace=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ck = load_torch(a.checkpoint)
    if ck is None or ck.get("quantization"):
        ap.error("input must be an existing floating-point checkpoint")
    floating = GPT.from_checkpoint(ck)
    parameter_count = sum(p.numel() for p in floating.parameters())
    model = quantize_model(floating)
    state = dict(
        model=model.state_dict(),
        cfg=model.c.to_dict(),
        quantization="dynamic-int8",
        stage="inference",
        parameter_count=parameter_count,
    )
    atomic_write(a.out, lambda f: torch.save(state, f))
    print(f"INT8 checkpoint: {Path(a.out).stat().st_size:,} bytes; CPU only; use --no-learning")


if __name__ == "__main__":
    main()
