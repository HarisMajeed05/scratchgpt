"""Continual learning from your corrections.

  fix(context, correction)  logs the correction, then runs a short fine-tune on: the new correction + older ones +
                            replayed instruction / pretraining data (replay stops it forgetting everything else).
  undo()                    rolls back the last fix (kept in memory for the last 3 fixes of this session).
  rebuild                   `scratchgpt rebuild`: start from the CURRENT base model and re-apply every saved correction.
                            Use it after the base model has trained further.
corrections.jsonl is the source of truth; the weights are a cache of it.
"""
import argparse
import glob
import random
from collections import deque

import numpy as np
import torch

from .. import paths
from ..chat_template import encode_example
from ..data.sft_format import Shard, collate, to_xy
from ..model.gpt import GPT
from ..safeio import atomic_write, load_torch
from .memory import CorrectionStore


class LearningError(RuntimeError):
    pass


class Replayer:
    """Samples old training data to mix into every fine-tune step."""

    def __init__(self, T=256):
        self.T = T
        self.sft = sorted(glob.glob(str(paths.SFT_DATA / "train_*.npz")))
        self.pre = sorted(glob.glob(str(paths.PRETRAIN_DATA / "train_*.bin")))
        self._shard = (None, None)

    def _load(self, path):
        if self._shard[0] != path:
            self._shard = (path, Shard(path))
        return self._shard[1]

    def sample(self, n):
        out = []
        for i in range(n):
            if self.sft and (i % 2 == 0 or not self.pre):
                shard = self._load(random.choice(self.sft))
                out.append(to_xy(*shard.row(random.randrange(len(shard)))))
            elif self.pre:
                d = np.memmap(random.choice(self.pre), dtype=np.uint16, mode="r")
                if len(d) > self.T + 2:
                    j = random.randrange(len(d) - self.T - 1)
                    c = d[j:j + self.T + 1].astype(np.int64)
                    out.append((c[:-1], c[1:]))
        return out


def save_live(model):
    atomic_write(paths.LIVE_MODEL, lambda f: torch.save({"model": model.state_dict(), "cfg": model.c.to_dict(),
                                                         "stage": "live"}, f))


def finetune(model, store, steps=60, lr=2e-5, newest=True, device="cuda", replay=True):
    """Fine-tune `model` in place on the stored corrections (+ replay)."""
    recs = store.all()
    if not recs:
        raise ValueError("no corrections to learn from")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    replayer = Replayer() if replay else None
    model.train()
    try:
        for _ in range(steps):
            pick = [random.choice(recs) for _ in range(4)]
            if newest:
                pick[0] = recs[-1]                  # the newest correction is always in the batch
            pairs = [to_xy(*encode_example(r["context"], r["reply"])) for r in pick]
            if replayer:
                pairs += replayer.sample(4)
            X, Y = collate(pairs)
            x, y = torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
                _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
    finally:
        model.eval()


class ContinualLearner:
    def __init__(self, engine, steps=60, lr=2e-5):
        self.engine, self.steps, self.lr = engine, steps, lr
        self.store = engine.store
        self.snapshots = deque(maxlen=3)

    def fix(self, context, correction):
        eng = self.engine
        with eng.lock:
            snap = {k: v.detach().cpu().clone() for k, v in eng.model.state_dict().items()}
            self.store.add(context, correction)
            try:
                finetune(eng.model, self.store, steps=self.steps, lr=self.lr, device=eng.device)
                save_live(eng.model)
            except Exception as e:                  # e.g. CUDA out of memory while pretraining is running
                eng.model.load_state_dict(snap)
                eng.model.zero_grad(set_to_none=True)
                self.store.drop_last()
                if eng.device_type == "cuda":
                    torch.cuda.empty_cache()
                eng.model.eval()
                raise LearningError(f"learning failed ({type(e).__name__}); rolled back. "
                                    "If training is running, pause it and retry.") from e
            self.snapshots.append(snap)

    def undo(self) -> bool:
        eng = self.engine
        with eng.lock:
            if not self.snapshots:
                return False
            eng.model.load_state_dict(self.snapshots.pop())
            self.store.drop_last()
            save_live(eng.model)
            return True


def main(argv=None):
    """`scratchgpt rebuild`"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    store = CorrectionStore()
    if not len(store):
        raise SystemExit("no corrections saved yet")
    ck = load_torch(paths.SFT_CKPT / "latest.pt", a.device) or load_torch(paths.LM_CKPT / "latest.pt", a.device)
    if ck is None:
        raise SystemExit("no base checkpoint found")
    model = GPT.from_checkpoint(ck, a.device)
    finetune(model, store, steps=a.steps, newest=False, device=a.device)
    save_live(model)
    print(f"rebuilt {paths.LIVE_MODEL} from base step {ck.get('step')} + {len(store)} corrections")


if __name__ == "__main__":
    main()
