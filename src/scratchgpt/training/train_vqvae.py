"""Image tokenizer (VQ-VAE): cache images, then train.

  scratchgpt vq-cache --n 200000      stream + resize images into artifacts/data/vq_images (~10 GB; resumable)
  scratchgpt vq-train --steps 60000   resumable; saves checkpoints/vq/vq.pt and recon_<step>.png (top: original, bottom: output)

Look at the recon PNGs before using the model. Do NOT train it further once image tokens have been written to the data.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .. import paths
from ..safeio import atomic_json, atomic_write, clean_tmp, load_json, load_torch, resume_stream, stream_state
from ..tokens import IMG_SIZE
from ..vision.imageio import open_pairs, prep_image, to_input, to_uint8
from ..vision.vqvae import VQVAE, K


def cache(n):
    """Resumable: progress is checkpointed every ~10k images; re-run the same command after a crash."""
    from concurrent.futures import ThreadPoolExecutor

    folder = paths.VQ_IMAGE_CACHE
    folder.mkdir(parents=True, exist_ok=True)
    clean_tmp(folder)
    npy, sp, src = folder / "images.npy", folder / "cache_state.json", "pixparse/cc12m-wds"
    st = load_json(sp) or {"done": 0, "consumed": 0, "ds": None}
    keys = {}

    def make():
        ds, keys["ik"], keys["ck"] = open_pairs(src)
        return ds

    if st["done"] and npy.exists():
        mm = np.load(npy, mmap_mode="r+")
        if len(mm) != n:
            raise SystemExit(f"{npy} was created with --n {len(mm)}; use that value or delete {folder}")
        print(f"resuming at {st['done']:,} images")
    else:
        st = {"done": 0, "consumed": 0, "ds": None}
        mm = np.lib.format.open_memmap(npy, mode="w+", dtype=np.uint8, shape=(n, IMG_SIZE, IMG_SIZE, 3))
    ds, exact = resume_stream(make, st["ds"], st["consumed"])
    ik = keys["ik"]
    it = iter(ds)
    done, consumed = st["done"], st["consumed"]

    def safe(b):
        try:
            return prep_image(b)
        except Exception:
            return None

    def checkpoint():
        mm.flush()
        atomic_json(sp, {"done": done, "consumed": consumed, "ds": stream_state(ds) if exact else None})

    chunks = 0
    try:
        with ThreadPoolExecutor(8) as pool:
            while done < n:
                chunk = []
                for _ in range(512):
                    try:
                        chunk.append(next(it)[ik]["bytes"])
                    except StopIteration:
                        break
                if not chunk:
                    break
                consumed += len(chunk)
                arrs = [a for a in pool.map(safe, chunk) if a is not None]
                k = min(len(arrs), n - done)
                if k:
                    mm[done:done + k] = np.stack(arrs[:k])
                    done += k
                chunks += 1
                if chunks % 20 == 0:
                    checkpoint()
                    print(f"cached {done:,}/{n:,}", flush=True)
    except KeyboardInterrupt:
        print("\nCtrl+C: saving progress ...")
    checkpoint()
    print(f"cached {done:,} images" + ("" if done == n else f" of {n:,}; re-run to continue, or train with what you have"))


def train(steps, bs, lr=2e-4):
    dev = "cuda"
    paths.VQ_CKPT.mkdir(parents=True, exist_ok=True)
    data = np.load(paths.VQ_IMAGE_CACHE / "images.npy", mmap_mode="r")
    n = len(data)
    m = VQVAE().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    path, step = paths.VQ_CKPT / "vq.pt", 0
    clean_tmp(paths.VQ_CKPT)
    ck = load_torch(path, dev)
    if ck:
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"resumed at step {step}")

    def save():  # fsync + atomic rename, previous checkpoint kept as vq.pt.prev
        atomic_write(path, lambda f: torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": step}, f))

    try:
        while step < steps:
            ix = np.sort(np.random.randint(0, n, bs))
            x = to_input(torch.from_numpy(data[ix]).to(dev))
            if np.random.rand() < 0.5:
                x = x.flip(3)
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (step + 1) / 500)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out, qloss, idx, z = m(x)
            rec = F.l1_loss(out.float(), x) + F.mse_loss(out.float(), x)
            opt.zero_grad(set_to_none=True)
            (rec + qloss).backward()
            opt.step()
            step += 1
            if step % 1000 == 0 and step < 0.8 * steps:
                print(f"restarted {m.restart_dead(z)} dead codes", flush=True)
            if step % 100 == 0:
                print(f"step {step} recon {rec.item():.4f} q {qloss.item():.4f} "
                      f"codes used in batch {idx.unique().numel()}/{K}", flush=True)
            if step % 1000 == 0:
                save()
            if step % 2000 == 0:
                with torch.no_grad():
                    o = m(x[:8])[0]
                top = np.concatenate(list(to_uint8(x[:8])), 1)
                bot = np.concatenate(list(to_uint8(o)), 1)
                Image.fromarray(np.concatenate([top, bot], 0)).save(paths.VQ_CKPT / f"recon_{step}.png")
    except KeyboardInterrupt:
        pass
    save()
    print("saved")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--n", type=int, default=200_000)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=60_000)
    t.add_argument("--bs", type=int, default=48)
    a = ap.parse_args(argv)
    cache(a.n) if a.cmd == "cache" else train(a.steps, a.bs)


if __name__ == "__main__":
    main()
