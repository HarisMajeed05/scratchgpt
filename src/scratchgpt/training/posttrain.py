"""Local JSONL SFT/distillation or DPO, with a frozen reference and assistant-only likelihoods."""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from ..chat_template import encode_conversation, encode_example
from ..data.quality import messages_of, read_rows
from ..model.gpt import GPT
from ..safeio import atomic_write, load_torch


def sequence_tensors(model, context, reply, device):
    ids, mask = encode_example(context, reply)
    if len(ids) > model.c.block_size:
        raise ValueError("training example exceeds context; shorten data or extend/train context explicitly")
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([ids[1:]], device=device)
    learn = torch.tensor([mask[1:]], device=device, dtype=torch.bool)
    return x, y, learn


def logprob(model, context, reply, device):
    x, y, learn = sequence_tensors(model, context, reply, device)
    logits, _ = model(x)
    lps = F.log_softmax(logits.float(), -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return (lps * learn).sum(), learn.sum()


def dpo_loss(pc, pr, rc, rr, beta):
    return -F.logsigmoid(beta * ((pc - pr) - (rc - rr)))


def save_model(model, path, stage, step, opt=None, extra=None):
    state = dict(model=model.state_dict(), cfg=model.c.to_dict(), stage=stage, step=step)
    if opt is not None:
        state["opt"] = opt.state_dict()
    state.update(extra or {})
    atomic_write(path, lambda f: torch.save(state, f))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["sft", "distill", "dpo"], default="distill")
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", help="separate held-out JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    a = ap.parse_args(argv)
    if min(a.steps, a.accum, a.save_every, a.eval_every) < 1 or a.beta <= 0:
        ap.error("steps, accum, intervals and beta must be positive")
    torch.manual_seed(a.seed)
    ck = load_torch(a.base, a.device)
    if ck is None or ck.get("quantization"):
        ap.error("base must be an existing floating-point checkpoint")
    model = GPT.from_checkpoint(ck, a.device)
    model.c.gradient_checkpointing = a.gradient_checkpointing
    model.train()
    reference = GPT.from_checkpoint(ck, a.device).eval().requires_grad_(False) if a.mode == "dpo" else None
    rows = list(read_rows(a.data))
    val = list(read_rows(a.val)) if a.val else []
    if not rows:
        ap.error("empty training dataset")
    if val:
        from ..data.quality import digest

        def group(row):
            return digest("\n".join(m["content"] for m in messages_of(row) if m["role"] != "assistant"))

        if {group(r) for r in rows} & {group(r) for r in val}:
            ap.error("training and validation prompt groups overlap")
    opt = model.configure_optimizer(a.lr, weight_decay=0)

    def loss_for(row):
        if a.mode == "dpo":
            context = row.get("context") or [{"role": "user", "content": row["prompt"]}]
            if not row["chosen"].strip() or not row["rejected"].strip() or row["chosen"] == row["rejected"]:
                raise ValueError("DPO needs distinct nonempty chosen/rejected answers")
            pc, _ = logprob(model, context, row["chosen"], a.device)
            pr, _ = logprob(model, context, row["rejected"], a.device)
            with torch.no_grad():
                rc, _ = logprob(reference, context, row["chosen"], a.device)
                rr, _ = logprob(reference, context, row["rejected"], a.device)
            return dpo_loss(pc, pr, rc, rr, a.beta)
        msgs = messages_of(row)
        ids, mask = encode_conversation(msgs)
        if len(ids) > model.c.block_size or len(ids) < 2 or not any(mask[1:]):
            raise ValueError("SFT example is empty, has no answer or exceeds context")
        x = torch.tensor([ids[:-1]], device=a.device)
        y = torch.tensor([ids[1:]], device=a.device)
        learn = torch.tensor([mask[1:]], device=a.device, dtype=torch.bool)
        y = y.masked_fill(~learn, -100)
        return model(x, y)[1]

    step = 0
    try:
        for step in range(1, a.steps + 1):
            opt.zero_grad(set_to_none=True)
            rng, total = random.Random(a.seed + step), 0.0
            for _ in range(a.accum):
                loss = loss_for(rows[rng.randrange(len(rows))]) / a.accum
                loss.backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            print(json.dumps({"step": step, "loss": total}), flush=True)
            if val and step % a.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    metric = sum(loss_for(r).item() for r in val[:32]) / min(32, len(val))
                print(json.dumps({"step": step, "val_loss": metric}), flush=True)
                model.train()
            if step % a.save_every == 0:
                save_model(model, Path(a.out) / "latest.pt", a.mode, step)
    finally:
        save_model(model, Path(a.out) / "latest.pt", a.mode, step)


if __name__ == "__main__":
    main()
