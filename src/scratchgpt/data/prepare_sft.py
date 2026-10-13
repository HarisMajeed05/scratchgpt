"""Instruction-tuning data -> .npz shards of (tokens, loss mask).

  chat        HuggingFaceTB/smol-smoltalk (multi-turn conversations)
  image chat  synthetic from image-caption pairs: "Draw X" -> image, and [image] "Describe this" -> caption
              (needs the trained VQ-VAE; skipped otherwise)

Crash-safe and resumable like the pretraining preparation (atomic shards, saved stream positions, Ctrl+C flush).
"""
import argparse
import random
from pathlib import Path

import numpy as np

from .. import paths
from ..chat_template import fit_conversation
from ..safeio import atomic_json, clean_tmp, load_json, resume_stream, stream_state
from ..sources import SFT_CHAT_DATASET, SFT_IMAGE_DATASET
from ..vision.imageio import open_pairs, prep_image, to_input, vq_checkpoint_exists
from .sft_format import write_shard

TOKV = "sft-v1"
BATCH = 256
DRAW_PROMPTS = ["Draw {c}", "Generate an image of {c}", "Can you create a picture of {c}?", "Make an image: {c}",
                "Show me {c}", "Please draw {c}."]
DRAW_REPLIES = ["Here is the image you asked for.", "Here you go.", "Here's what I came up with.", ""]
DESCRIBE_PROMPTS = ["Describe this image.", "What is in this picture?", "Write a caption for this image.",
                    "What do you see here?", "Can you describe what's shown?"]


def _tidy(caption):
    c = (caption or "").strip()
    if not c:
        return ""
    c = c[0].upper() + c[1:]
    return c if c[-1] in ".!?" else c + "."


def _lower_first(c):
    return c[:1].lower() + c[1:] if c else c


class SFTPreparer:
    def __init__(self, out, max_len, shard_rows, image_chats, device):
        self.out, self.max_len, self.shard_rows, self.image_chats, self.device = Path(out), max_len, shard_rows, image_chats, device
        self.out.mkdir(parents=True, exist_ok=True)
        clean_tmp(self.out)
        self.state_path = self.out / "state.json"
        st = load_json(self.state_path) or {}
        st["tok"] = TOKV
        st.setdefault("train", 0)
        st.setdefault("val", 0)
        st.setdefault("jobs", {})
        self.st = st
        self.committed = {k: dict(v) for k, v in st["jobs"].items()}
        self.train_rows, self.val_rows = [], []
        self.rng = random.Random(0)

    def _flush(self):
        if self.train_rows:
            write_shard(self.out / f"train_{self.st['train']:05d}.npz", self.train_rows)
            self.st["train"] += 1
        if self.val_rows:
            write_shard(self.out / f"val_{self.st['val']:05d}.npz", self.val_rows)
            self.st["val"] += 1
        self.st["jobs"] = {k: dict(v) for k, v in self.committed.items()}
        atomic_json(self.state_path, self.st)
        print(f"wrote {len(self.train_rows)} train + {len(self.val_rows)} val conversations | shards: "
              f"{self.st['train']} train, {self.st['val']} val", flush=True)
        self.train_rows, self.val_rows = [], []

    def _add(self, rows, doc_index0):
        for i, row in enumerate(rows):
            (self.val_rows if (doc_index0 + i) % 100 == 0 else self.train_rows).append(row)

    # ---------- jobs ----------
    def _chat_rows(self, batch):
        rows = []
        for ex in batch:
            msgs = [{"role": m["role"], "content": m["content"]} for m in ex["messages"]
                    if m["role"] in ("system", "user", "assistant") and isinstance(m["content"], str)]
            ids, mask = fit_conversation(msgs, self.max_len)
            if len(ids) >= 8 and any(mask):
                rows.append((ids, mask))
        return rows

    def _image_rows(self, batch):
        import torch
        arrs = [(prep_image_safe(e[self._ik]["bytes"]), e[self._ck]) for e in batch]
        ok = [(a, c) for a, c in arrs if a is not None and (c or "").strip()]
        if not ok:
            return []
        imgs = torch.from_numpy(np.stack([a for a, _ in ok])).to(self.device)
        codes = []
        for i in range(0, len(imgs), 128):
            with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
                codes.append(self.vq.encode(to_input(imgs[i:i + 128])).cpu().numpy())
        codes = np.concatenate(codes)
        rows = []
        for code, (_, cap) in zip(codes, ok):
            code = [int(c) for c in code]
            cap_t = _tidy(cap)
            if self.rng.random() < 0.5:     # text -> image
                prompt = self.rng.choice(DRAW_PROMPTS).format(c=_lower_first(cap_t).rstrip(".!?"))
                reply = self.rng.choice(DRAW_REPLIES)
                parts = ([{"type": "text", "text": reply}] if reply else []) + [{"type": "image", "codes": code}]
                msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": parts}]
            else:                            # image -> text
                msgs = [{"role": "user", "content": [{"type": "image", "codes": code},
                                                     {"type": "text", "text": self.rng.choice(DESCRIBE_PROMPTS)}]},
                        {"role": "assistant", "content": cap_t}]
            rows.append(fit_conversation(msgs, self.max_len))
        return [r for r in rows if r[0]]

    def _run_job(self, name, make, build, limit=None):
        js = self.committed.setdefault(name, {"docs": 0, "ds": None, "done": False})
        if js["done"]:
            print(f"[{name}] already done")
            return
        ds, exact = resume_stream(make, js["ds"], js["docs"])
        it = iter(ds)
        docs = js["docs"]
        while True:
            batch = []
            for _ in range(BATCH):
                try:
                    batch.append(next(it))
                except StopIteration:
                    break
            if limit is not None:
                batch = batch[: max(limit - docs, 0)]
            self._add(build(batch), docs)
            docs += len(batch)
            finished = len(batch) < BATCH or (limit is not None and docs >= limit)
            self.committed[name] = {"docs": docs, "ds": stream_state(ds) if exact else None, "done": finished}
            if finished or len(self.train_rows) >= self.shard_rows:
                self._flush()
            if finished:
                print(f"[{name}] done: {docs:,} source examples")
                return

    def run(self):
        from datasets import load_dataset

        from ..vision.vqvae import load_vq
        try:
            self._run_job("chat", lambda: load_dataset(SFT_CHAT_DATASET, split="train", streaming=True), self._chat_rows)
            if self.image_chats and vq_checkpoint_exists():
                self.vq = load_vq(device=self.device)

                def make():
                    ds, self._ik, self._ck = open_pairs(SFT_IMAGE_DATASET)
                    return ds

                self._run_job("imagechat", make, self._image_rows, limit=self.image_chats)
            elif self.image_chats:
                print("No trained VQ-VAE yet: skipping image chats. Re-run after `scratchgpt vq-train`.")
        except KeyboardInterrupt:
            print("\nCtrl+C received: saving progress ...", flush=True)
        finally:
            if self.train_rows or self.val_rows:
                self._flush()
        print("stopped. Re-run the same command to resume.")


def prep_image_safe(b):
    try:
        return prep_image(b)
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prepare instruction-tuning data.")
    ap.add_argument("--out", default=str(paths.SFT_DATA))
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--shard-rows", type=int, default=50_000)
    ap.add_argument("--image-chats", type=int, default=60_000, help="synthetic image chats to add (0 = none)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--local-jsonl", help="quality-filtered train.jsonl + val.jsonl directory, or single JSONL with split field")
    a = ap.parse_args(argv)
    if a.local_jsonl:
        from .quality import messages_of, read_rows
        from .sft_format import write_shard
        source, out = Path(a.local_jsonl), Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        if list(out.glob("*_*.npz")):
            ap.error("local import requires a fresh output directory to avoid duplication")
        buffers, numbers = {"train": [], "val": []}, {"train": 0, "val": 0}
        files = [source/"train.jsonl", source/"val.jsonl"] if source.is_dir() else [source]
        def flush(kind):
            if buffers[kind]:
                write_shard(out/f"{kind}_{numbers[kind]:05d}.npz", buffers[kind])
                numbers[kind] += 1
                buffers[kind] = []
        for file in files:
            for row in read_rows(file):
                kind = row.get("split", file.stem if file.stem in buffers else "train")
                if kind not in buffers:
                    ap.error("split must be train or val")
                ids, mask = fit_conversation(messages_of(row), a.max_len)
                if len(ids) >= 2 and any(mask):
                    buffers[kind].append((ids, mask))
                if len(buffers[kind]) >= a.shard_rows:
                    flush(kind)
        for kind in buffers:
            flush(kind)
        print("imported local SFT shards", numbers)
        return
    SFTPreparer(a.out, a.max_len, a.shard_rows, a.image_chats, a.device).run()


if __name__ == "__main__":
    main()
