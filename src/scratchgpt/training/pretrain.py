"""Pretrain the language model on the prepared shards. Safe to Ctrl+C or lose power: re-run to resume.

The learning rate warms up, then stays constant (so you can train for months), and decays only once you pass
--decay-start <step> to wrap up. Checkpoints are fsync'ed + atomic and the previous one is kept as latest.pt.prev.
"""
import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import torch

from .. import paths
from ..model.gpt import GPT, GPTConfig
from ..safeio import atomic_write, clean_tmp, load_torch


class ShardSampler:
    """Random windows from the .bin shards (re-scanned every step, so new shards are picked up as they appear)."""

    def __init__(self, folder, T, device):
        self.folder, self.T, self.device = Path(folder), T, device
        self.files = {"train": [], "val": []}
        self.sizes = {"train": None, "val": None}
        self.mm = {}

    def refresh(self):
        for kind in self.files:
            fs = [f for f in sorted(glob.glob(str(self.folder / f"{kind}_*.bin"))) if os.path.getsize(f) > 2 * (self.T + 2)]
            self.files[kind] = fs
            sz = np.array([os.path.getsize(f) for f in fs], dtype=np.float64)
            self.sizes[kind] = sz / sz.sum() if len(fs) else None

    def ready(self):
        return bool(self.files["train"]) and bool(self.files["val"])

    def batch(self, kind, bs):
        xs, ys = [], []
        for _ in range(bs):
            f = self.files[kind][np.random.choice(len(self.files[kind]), p=self.sizes[kind])]
            if f not in self.mm:
                self.mm[f] = np.memmap(f, dtype=np.uint16, mode="r")
            d = self.mm[f]
            i = np.random.randint(len(d) - self.T - 1)
            c = d[i:i + self.T + 1].astype(np.int64)
            xs.append(c[:-1])
            ys.append(c[1:])
        x = torch.from_numpy(np.stack(xs)).to(self.device, non_blocking=True)
        y = torch.from_numpy(np.stack(ys)).to(self.device, non_blocking=True)
        return x, y


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(paths.PRETRAIN_DATA))
    ap.add_argument("--out", default=str(paths.LM_CKPT))
    ap.add_argument("--micro-bs", type=int, default=8, help="raise if VRAM allows")
    ap.add_argument("--accum", type=int, default=64, help="8*1024*64 = ~524k tokens per step")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=700)
    ap.add_argument("--decay-start", type=int, default=10**9, help="step at which the final LR decay begins")
    ap.add_argument("--decay-steps", type=int, default=20000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=100, help="steps between checkpoints (~20 min by default)")
    ap.add_argument("--milestone-every", type=int, default=5000)
    ap.add_argument("--compile", action="store_true", help="needs triton-windows on Windows, or use WSL2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seq-len", type=int, default=1024, help="actual training length (curriculum: 1k -> 4k -> 8k -> 32k)")
    ap.add_argument("--context", type=int, default=None, help="configured maximum (new model: 32768; resume: keep checkpoint limit)")
    ap.add_argument("--base", help="continue pretraining a floating-point checkpoint")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--kv-heads", type=int, default=3)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0, help="0 runs until Ctrl+C; otherwise stop at total step")
    a = ap.parse_args(argv)
    if a.seq_len < 1 or (a.context is not None and a.context < a.seq_len) or min(a.micro_bs, a.accum, a.save_every, a.eval_every, a.milestone_every) < 1:
        ap.error("positive batch/intervals and seq-len <= context required")

    dev, T = a.device, a.seq_len
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    sampler = ShardSampler(a.data, T, dev)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clean_tmp(out)
    path, step = out / "latest.pt", 0
    ck = load_torch(path, dev)        # falls back to latest.pt.prev if latest.pt is unreadable
    if ck is None and a.base:
        ck = load_torch(a.base, dev)
        if ck is None:
            ap.error("base checkpoint not found")
    if ck and ck.get("quantization"):
        ap.error("cannot train an INT8 checkpoint")
    if ck:
        model = GPT.from_checkpoint(ck, dev)
        if model.c.architecture == "modern":
            model.c.block_size = a.context or model.c.block_size
        if T > model.c.block_size:
            ap.error("sequence length exceeds checkpoint context")
        opt = model.configure_optimizer(a.lr)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        step = ck.get("step", 0)
        print(f"continued at step {step}")
    else:
        model = GPT(GPTConfig(block_size=a.context or 32768, n_layer=a.layers, n_head=a.heads,
                              n_kv_head=a.kv_heads, n_embd=a.width)).to(dev)
        opt = model.configure_optimizer(a.lr)
    model.c.gradient_checkpointing = a.gradient_checkpointing
    cm = torch.compile(model) if a.compile else model

    def get_lr(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        if s < a.decay_start:
            return a.lr
        f = min(1.0, (s - a.decay_start) / a.decay_steps)
        return a.lr + (a.min_lr - a.lr) * f

    def save():
        atomic_write(path, lambda f: torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                                                 "cfg": model.c.to_dict()}, f))

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot = 0.0
        for _ in range(20):
            x, y = sampler.batch("val", a.micro_bs)
            with torch.autocast(torch.device(dev).type, dtype=torch.bfloat16, enabled=torch.device(dev).type == "cuda"):
                _, loss = cm(x, y)
            tot += loss.item()
        model.train()
        return tot / 20

    try:
        while not a.max_steps or step < a.max_steps:
            sampler.refresh()
            if not sampler.ready():
                print("waiting for data shards (need at least one train and one val shard) ...")
                time.sleep(30)
                continue
            for g in opt.param_groups:
                g["lr"] = get_lr(step)
            t0, tot = time.time(), 0.0
            for _ in range(a.accum):
                x, y = sampler.batch("train", a.micro_bs)
                with torch.autocast(torch.device(dev).type, dtype=torch.bfloat16, enabled=torch.device(dev).type == "cuda"):
                    _, loss = cm(x, y)
                (loss / a.accum).backward()
                tot += loss.item() / a.accum
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            dt = time.time() - t0
            print(f"step {step} loss {tot:.4f} lr {get_lr(step):.2e} norm {norm:.2f} "
                  f"{a.micro_bs * T * a.accum / dt:,.0f} tok/s", flush=True)
            if step % a.eval_every == 0:
                print(f"== val loss {evaluate():.4f}", flush=True)
            if step % a.save_every == 0:
                save()
            if step % a.milestone_every == 0:
                atomic_write(out / f"step_{step}.pt",
                             lambda f: torch.save({"model": model.state_dict(), "cfg": model.c.to_dict()}, f),
                             keep_prev=False)
    except KeyboardInterrupt:
        pass
    save()
    print("saved, bye")


if __name__ == "__main__":
    main()
