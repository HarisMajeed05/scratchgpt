"""Small on-policy REINFORCE trainer: checkable rewards, leave-one-out baseline, reference KL.

This is a research baseline, not PPO or production GRPO. Code tasks require Docker.
"""

import argparse
import json
import random
from pathlib import Path

import torch

from ..chat_template import encode_conversation
from ..data.quality import read_rows
from ..model.gpt import GPT
from ..safeio import load_torch
from ..tokenizer import tokenizer
from ..tokens import END_TURN, TEXT_VOCAB
from .posttrain import save_model
from .verifiers import verify


@torch.no_grad()
def rollout(model, prompt, device, max_new):
    ids, _ = encode_conversation([{"role": "user", "content": prompt}], add_generation_prompt=True)
    if len(ids) + max_new > model.c.block_size:
        raise ValueError("RL prompt + completion exceeds context")
    seq = list(ids)
    logits, cache, pos = model.prefill(torch.tensor([ids], device=device))
    for _ in range(max_new):
        # Same legal-token distribution is used in rollout AND policy likelihoods.
        legal = torch.cat((logits[0, :TEXT_VOCAB], logits[0, END_TURN : END_TURN + 1]))
        sampled = int(torch.multinomial(legal.float().softmax(-1), 1))
        tid = END_TURN if sampled == TEXT_VOCAB else sampled
        seq.append(tid)
        if tid == END_TURN:
            break
        logits, cache = model.decode_step(torch.tensor([[tid]], device=device), cache, pos)
        pos += 1
    return seq, len(ids), tokenizer.decode(seq[len(ids) :])


def policy_stats(model, reference, seq, start, device):
    x = torch.tensor([seq[:-1]], device=device)
    y = torch.tensor(seq[start:], device=device)
    logits = model(x)[0][0, start - 1 :]
    legal = torch.cat((logits[:, :TEXT_VOCAB], logits[:, END_TURN : END_TURN + 1]), -1).float()
    lp = legal.log_softmax(-1)
    target = torch.where(y == END_TURN, TEXT_VOCAB, y)
    selected = lp.gather(-1, target[:, None]).sum()
    with torch.no_grad():
        ref_logits = reference(x)[0][0, start - 1 :]
        ref_legal = torch.cat((ref_logits[:, :TEXT_VOCAB], ref_logits[:, END_TURN : END_TURN + 1]), -1).float()
        ref_lp = ref_legal.log_softmax(-1)
    kl = (lp.exp() * (lp - ref_lp)).sum(-1).mean()
    return selected, kl


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--kl", type=float, default=0.02)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.samples < 2 or a.steps < 1 or a.max_new < 1 or a.kl < 0:
        ap.error("samples >= 2, positive steps/max-new and nonnegative KL required")
    torch.manual_seed(a.seed)
    ck = load_torch(a.base, a.device)
    if ck is None or ck.get("quantization"):
        ap.error("RL requires floating-point base checkpoint")
    model = GPT.from_checkpoint(ck, a.device).train()
    reference = GPT.from_checkpoint(ck, a.device).eval().requires_grad_(False)
    opt = model.configure_optimizer(a.lr, weight_decay=0)
    rows = list(read_rows(a.data))
    if not rows:
        ap.error("empty reward dataset")
    step = 0
    try:
        for step in range(1, a.steps + 1):
            row = rows[random.Random(a.seed + step).randrange(len(rows))]
            model.eval()
            runs = [rollout(model, row["prompt"], a.device, a.max_new) for _ in range(a.samples)]
            rewards = [verify(row, r[2]) for r in runs]
            model.train()
            opt.zero_grad(set_to_none=True)
            total = 0
            for i, (seq, start, _) in enumerate(runs):
                baseline = (sum(rewards) - rewards[i]) / (a.samples - 1)
                lp, kl = policy_stats(model, reference, seq, start, a.device)
                loss = (-(rewards[i] - baseline) * lp + a.kl * kl) / a.samples
                loss.backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            print(json.dumps({"step": step, "reward": sum(rewards) / a.samples, "loss": total}), flush=True)
            if step % 10 == 0:
                save_model(model, Path(a.out) / "latest.pt", "rlvr-reinforce", step)
    finally:
        save_model(model, Path(a.out) / "latest.pt", "rlvr-reinforce", step)


if __name__ == "__main__":
    main()
