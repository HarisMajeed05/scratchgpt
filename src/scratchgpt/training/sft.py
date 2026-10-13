"""Instruction-tune the pretrained model into a chat model (loss only on assistant replies).

Starts from checkpoints/lm/latest.pt and writes checkpoints/sft/latest.pt. Resumable and crash-safe; the data order is a
function of the step number, so a resumed run continues exactly where it stopped.
"""
import argparse
import glob
import math
import time
from pathlib import Path

import numpy as np
import torch

from .. import paths
from ..data.sft_format import Shard, collate, to_xy
from ..model.gpt import GPT
from ..safeio import atomic_write, clean_tmp, load_torch


def _load_rows(folder, kind):
    shards = [Shard(p) for p in sorted(glob.glob(str(Path(folder) / f"{kind}_*.npz")))]
    return shards, [(si, ri) for si, s in enumerate(shards) for ri in range(len(s))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(paths.SFT_DATA))
    ap.add_argument("--base", default=str(paths.LM_CKPT / "latest.pt"))
    ap.add_argument("--out", default=str(paths.SFT_CKPT))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8, help="conversations per step = micro-bs * accum")
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    if min(a.micro_bs, a.accum, a.save_every, a.eval_every) < 1 or a.epochs <= 0:
        ap.error("batch sizes, intervals and epochs must be positive")

    dev = a.device
    shards, rows = _load_rows(a.data, "train")
    vshards, vrows = _load_rows(a.data, "val")
    if not rows:
        raise SystemExit("no SFT data found. Run: scratchgpt prepare-sft")
    N, cps = len(rows), a.micro_bs * a.accum
    steps_per_epoch = max(math.ceil(N / cps), 1)
    total = max(1, math.ceil(steps_per_epoch * a.epochs))
    print(f"{N:,} train conversations, {len(vrows):,} val | {steps_per_epoch} steps/epoch, {total} steps total")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clean_tmp(out)
    path, step = out / "latest.pt", 0
    ck = load_torch(path, dev)
    base_step = None
    if ck:
        model = GPT.from_checkpoint(ck, dev)
        opt = model.configure_optimizer(a.lr, weight_decay=0.0)
        opt.load_state_dict(ck["opt"])
        step, base_step = ck["step"], ck.get("base_step")
        print(f"resumed at step {step}")
    else:
        base = load_torch(a.base, dev)
        if base is None:
            raise SystemExit(f"base checkpoint {a.base} not found. Pretrain first: scratchgpt pretrain")
        model = GPT.from_checkpoint(base, dev)
        opt = model.configure_optimizer(a.lr, weight_decay=0.0)
        base_step = base.get("step")
        print(f"starting from the pretrained model at step {base_step}")

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = (s - a.warmup) / max(total - a.warmup, 1)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    def make_batch(shard_list, row_list, idxs):
        pairs = [to_xy(*shard_list[row_list[i][0]].row(row_list[i][1])) for i in idxs]
        X, Y = collate(pairs)
        return torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev)

    def save():
        atomic_write(path, lambda f: torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                                                 "cfg": model.c.to_dict(), "base_step": base_step, "stage": "sft"}, f))

    @torch.no_grad()
    def evaluate():
        if not vrows:
            return float("nan")
        model.eval()
        rng, tot, n = np.random.default_rng(0), 0.0, 0
        for _ in range(10):
            idx = rng.choice(len(vrows), size=min(a.micro_bs, len(vrows)), replace=False)
            x, y = make_batch(vshards, vrows, idx)
            with torch.autocast(torch.device(dev).type, dtype=torch.bfloat16, enabled=torch.device(dev).type == "cuda"):
                _, loss = model(x, y)
            tot += loss.item()
            n += 1
        model.train()
        return tot / n

    model.train()
    try:
        while step < total:
            epoch, k = divmod(step, steps_per_epoch)
            order = np.random.default_rng(a.seed + epoch).permutation(N)
            chunk = order[k * cps:min((k + 1) * cps, N)]
            micros = [chunk[m:m+a.micro_bs] for m in range(0, len(chunk), a.micro_bs)]
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            t0, tot = time.time(), 0.0
            for micro in micros:
                x, y = make_batch(shards, rows, micro)
                with torch.autocast(torch.device(dev).type, dtype=torch.bfloat16, enabled=torch.device(dev).type == "cuda"):
                    _, loss = model(x, y)
                (loss / len(micros)).backward()
                tot += loss.item() / len(micros)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            print(f"step {step}/{total} loss {tot:.4f} lr {lr_at(step):.2e} norm {norm:.2f} {time.time() - t0:.1f}s", flush=True)
            if step % a.eval_every == 0:
                print(f"== val loss {evaluate():.4f}", flush=True)
            if step % a.save_every == 0:
                save()
    except KeyboardInterrupt:
        pass
    save()
    print("saved, bye")


if __name__ == "__main__":
    main()
