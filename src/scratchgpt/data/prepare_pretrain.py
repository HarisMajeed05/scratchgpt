"""Pretraining data: stream a weighted mix of datasets, tokenize, write crash-safe uint16 shards.

Image-caption pairs become one sequence each, in a random order so the model learns both directions:
  caption <|img|> 256 image codes <|/img|> <|endoftext|>     (text -> image)
  <|img|> 256 image codes <|/img|> caption <|endoftext|>     (image -> text)

CRASH SAFETY
  - Every shard and the state file are written to a .tmp file, fsync'ed, then renamed (atomic): a power cut can only
    lose the work since the last shard (~100M tokens), never corrupt anything.
  - state.json keeps a .prev backup and stores each dataset's exact stream position, so resuming is instant.
  - Ctrl+C (or any error) flushes what is in memory into a final shard and saves state before exiting.
  - If state.json is lost entirely it is rebuilt from the shards on disk (some early documents then repeat).
Stages: run with --until 90e9 first (images are skipped while no VQ-VAE exists), train the VQ-VAE, then run again
without --until: the image sources catch up. Never retrain the VQ-VAE after image tokens were written.
"""
import argparse
import glob
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .. import paths
from ..safeio import atomic_json, atomic_write, clean_tmp, load_json, resume_stream, stream_state
from ..sources import PRETRAIN_SOURCES
from ..tokenizer import tokenizer
from ..tokens import BOI, EOI, EOT, IMG_BASE
from ..vision.imageio import open_pairs, prep_image, to_input, vq_checkpoint_exists

TOKV = "bpe32k-v1"
DEFAULT_TOTAL = 110e9                      # ~220 GB as uint16
SHARD, VAL_SHARD = 100_000_000, 10_000_000  # a shard is written once it holds this many tokens
BATCH_DOCS = 256
MAX_CHARS = 200_000


def _load_dataset(**kw):
    from datasets import load_dataset
    return load_dataset(**kw)


class Preparer:
    def __init__(self, out, until, device="cuda"):
        self.out, self.until, self.device = Path(out), until, device
        self.out.mkdir(parents=True, exist_ok=True)
        clean_tmp(self.out)
        self.state_path = self.out / "state.json"
        st = load_json(self.state_path)
        if st is None:
            st = self._recover_from_shards()
        if st and st.get("tok") != TOKV:
            raise SystemExit("data was made with a different tokenizer. Delete the data folder and re-run.")
        st["tok"] = TOKV
        for key, default in (("train", 0), ("val", 0), ("docs", {}), ("toks", {}), ("ds", {})):
            st.setdefault(key, default)
        self.st = st
        self.src = PRETRAIN_SOURCES
        self.docs = {k: st["docs"].get(k, 0) for k in self.src}      # live counters
        self.toks = {k: st["toks"].get(k, 0) for k in self.src}
        # "committed" = exactly what is already inside shards on disk; this is what state.json records
        self.cdocs, self.ctoks = dict(self.docs), dict(self.toks)
        self.cds = {k: st["ds"].get(k) for k in self.src}
        self.parts, self.count = [], 0             # tokens of fully committed batches not yet written to a shard
        self.its, self.dss, self.exact, self.opened = {}, {}, {}, set()
        self.pool = ThreadPoolExecutor(8)
        self.rng = random.Random(0)
        self._check_disk()
        self._setup_vision()

    # ---------- startup ----------
    def _recover_from_shards(self):
        """state.json (and its backup) unreadable: keep shards on disk and continue numbering after them.
        Stream positions are unknown, so datasets restart from their beginning and some early documents repeat."""
        tr = sorted(glob.glob(str(self.out / "train_*.bin")))
        va = sorted(glob.glob(str(self.out / "val_*.bin")))
        if not tr and not va:
            return {}
        # A complete shard holds at least SHARD tokens; a smaller LAST one may have been cut off mid-write by the crash
        # (or is a Ctrl+C flush). Either way it is only set aside (renamed .bad), never deleted.
        for files, size in ((tr, 2 * SHARD), (va, 2 * VAL_SHARD)):
            if files and os.path.getsize(files[-1]) < size:
                bad = files.pop()
                os.replace(bad, bad + ".bad")
                print(f"set aside incomplete shard {bad}", flush=True)
        total = sum(os.path.getsize(f) for f in tr + va) // 2
        text = {k: v["share"] for k, v in PRETRAIN_SOURCES.items() if v["kind"] == "text"}
        ts = sum(text.values())
        print(f"WARNING: state lost. Kept {len(tr)} train + {len(va)} val shards ({total / 1e9:.2f}B tokens). "
              f"Datasets restart from their beginning; up to ~{total / 1e9:.1f}B tokens of early documents may repeat.")
        return {"tok": TOKV, "train": len(tr), "val": len(va), "docs": {}, "ds": {},
                "toks": {k: int(total * s / ts) for k, s in text.items()}}

    def _check_disk(self):
        for kind in ("train", "val"):
            missing = [i for i in range(self.st[kind]) if not (self.out / f"{kind}_{i:06d}.bin").exists()]
            if missing:
                print(f"WARNING: {len(missing)} {kind} shard(s) in state are missing on disk (first: {missing[0]})")
            extra = len(glob.glob(str(self.out / f"{kind}_*.bin"))) - (self.st[kind] - len(missing))
            if extra > 0:
                print(f"note: {extra} {kind} shard(s) newer than the saved state (crash right after writing); they will be rewritten")
        done = sum(self.toks.values())
        need_gb = max(self.until - done, 0) * 2 / 1e9
        free_gb = shutil.disk_usage(self.out).free / 1e9
        print(f"target {self.until / 1e9:.0f}B tokens; already {done / 1e9:.2f}B; still ~{need_gb:,.0f} GB to write, "
              f"free disk {free_gb:,.0f} GB")
        if free_gb < need_gb + 20:
            print("WARNING: not enough free space for the full target; stop with Ctrl+C when you hit your limit.")

    def _setup_vision(self):
        self.vq = None
        self.exhausted = set()
        if vq_checkpoint_exists():
            import torch  # noqa: F401

            from ..vision.vqvae import load_vq
            self.vq = load_vq(device=self.device)
            step = self.vq.step
            if self.st.get("vq_step") not in (None, step) and any(
                    self.ctoks[k] for k, v in self.src.items() if v["kind"] == "img"):
                raise SystemExit(f"data already holds image tokens made with the VQ-VAE at step {self.st['vq_step']}, "
                                 f"but the checkpoint is step {step}. Codes from different VQ-VAE versions are "
                                 "incompatible; do not train the VQ-VAE further.")
            self.st["vq_step"] = step              # saved with the next shard
        else:
            self.exhausted = {k for k, v in self.src.items() if v["kind"] == "img"}
            print("No trained VQ-VAE yet: image sources skipped. Re-run (without --until) after `scratchgpt vq-train`.")

    # ---------- streaming ----------
    def _open_src(self, k, use_state):
        s = self.src[k]
        keys = {}

        def make():
            if s["kind"] == "img":
                ds, keys["ik"], keys["ck"] = open_pairs(s["kw"]["path"])
                return ds
            return _load_dataset(**s["kw"], split="train", streaming=True)

        ds, exact = resume_stream(make, self.cds.get(k) if use_state else None, self.docs[k])
        self.dss[k], self.exact[k] = ds, exact
        it = iter(ds)
        if s["kind"] == "img":
            ik, ck = keys["ik"], keys["ck"]

            def nxt():
                e = next(it)
                return e[ik]["bytes"], e[ck]
        else:
            field = s["field"]

            def nxt():
                return next(it)[field][:MAX_CHARS]
        return nxt

    def _pull(self, k, n):
        out, fails = [], 0
        while len(out) < n:
            try:
                if k not in self.its:
                    self.its[k] = self._open_src(k, use_state=k not in self.opened)
                    self.opened.add(k)
                out.append(self.its[k]())
                self.docs[k] += 1
                fails = 0
            except StopIteration:
                break
            except Exception as e:          # network hiccup / rate limit: reopen at the current position and retry
                fails += 1
                if fails > 20:
                    raise
                print(f"[{k}] {type(e).__name__}: {str(e)[:120]} - retry {fails}/20 in 30s", flush=True)
                self.its.pop(k, None)
                time.sleep(30)
        return out

    def _tokenize(self, k, items):
        if self.src[k]["kind"] == "text":
            return [np.array(ids + [EOT], dtype=np.uint16) for ids in tokenizer.encode_batch(items)]
        import torch

        def safe(b):
            try:
                return prep_image(b)
            except Exception:
                return None

        arrs = list(self.pool.map(safe, [b for b, _ in items]))
        ok = [(a, c) for a, (_, c) in zip(arrs, items) if a is not None]
        if not ok:
            return []
        imgs = torch.from_numpy(np.stack([a for a, _ in ok])).to(self.device)
        codes = []
        for i in range(0, len(imgs), 128):
            with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
                codes.append(self.vq.encode(to_input(imgs[i:i + 128])).cpu().numpy())
        codes = np.concatenate(codes).astype(np.int64) + IMG_BASE
        caps = tokenizer.encode_batch([(c or "")[:300] for _, c in ok])
        out = []
        for code, cap in zip(codes, caps):
            img = np.concatenate([[BOI], code, [EOI]])
            cap = np.array(cap, dtype=np.int64)
            seq = [cap, img] if self.rng.random() < 0.5 else [img, cap]
            out.append(np.concatenate(seq + [np.array([EOT])]).astype(np.uint16))
        return out

    # ---------- checkpointing ----------
    def _commit(self, k):
        self.cdocs[k], self.ctoks[k] = self.docs[k], self.toks[k]
        self.cds[k] = stream_state(self.dss[k]) if self.exact.get(k) else None

    def _save_state(self):
        self.st["docs"], self.st["toks"], self.st["ds"] = dict(self.cdocs), dict(self.ctoks), dict(self.cds)
        atomic_json(self.state_path, self.st)

    def _write_shard(self, kind):
        arr = np.concatenate(self.parts)             # ALL committed batches, so shard + state always match exactly
        while True:
            path = self.out / f"{kind}_{self.st[kind]:06d}.bin"
            try:
                atomic_write(path, lambda f: f.write(arr.tobytes()), keep_prev=False)
                break
            except PermissionError:                  # e.g. the trainer has this shard open (Windows)
                print(f"{path} is in use; writing to the next index instead", flush=True)
                self.st[kind] += 1
        self.st[kind] += 1
        self._save_state()
        self.parts, self.count = [], 0
        tot = sum(self.ctoks.values())
        mix = " ".join(f"{k}:{100 * v / max(tot, 1):.0f}%" for k, v in self.ctoks.items() if v)
        print(f"wrote {kind} shard {self.st[kind] - 1} ({len(arr) / 1e6:.0f}M tok) | {tot / 1e9:.2f}B tokens | {mix}", flush=True)

    def _maybe_write(self, force=False):
        is_val = self.st["val"] == 0
        target = VAL_SHARD if is_val else SHARD
        if self.count >= target or (force and self.count >= (1_000_000 if is_val else 1)):
            self._write_shard("val" if is_val else "train")
        elif force and self.count:
            print(f"only {self.count:,} tokens buffered (< 1M, too few for the validation shard); not saved, re-read next run")

    # ---------- main loop ----------
    def step(self) -> bool:
        """Process one batch. Returns False when finished."""
        if sum(self.toks.values()) >= self.until:
            return False
        live = [k for k in self.src if k not in self.exhausted]
        if not live:
            return False
        wsum = sum(self.src[k]["share"] for k in live)
        tl = sum(self.toks[k] for k in live) + 1e6
        k = max(live, key=lambda s: self.src[s]["share"] / wsum * tl - self.toks[s])   # most behind its target share
        items = self._pull(k, BATCH_DOCS)
        ended = len(items) < BATCH_DOCS
        arrs = self._tokenize(k, items) if items else []
        n = sum(len(a) for a in arrs)
        self.parts.extend(arrs)
        self.count += n
        self.toks[k] += n
        if items:
            self._commit(k)
        if ended:
            self.exhausted.add(k)
            print(f"[{k}] exhausted at {self.toks[k] / 1e9:.2f}B tokens; other sources will fill in", flush=True)
        self._maybe_write()
        return True

    def run(self):
        try:
            while self.step():
                pass
        except KeyboardInterrupt:
            print("\nCtrl+C received: saving progress ...", flush=True)
        finally:
            self._maybe_write(force=True)            # also runs on crashes: whatever is committed goes to disk
        print(f"stopped at {sum(self.ctoks.values()) / 1e9:.2f}B tokens. Re-run the same command to resume.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prepare the pretraining mix (resumable, crash-safe).")
    ap.add_argument("--until", type=float, default=DEFAULT_TOTAL, help="stop at this many total tokens, e.g. 90e9")
    ap.add_argument("--out", default=str(paths.PRETRAIN_DATA))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    Preparer(a.out, a.until, a.device).run()


if __name__ == "__main__":
    main()
