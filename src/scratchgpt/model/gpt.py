"""GPT-2-style decoder (124M with the default config): pre-LN, tied embeddings, flash attention, KV cache."""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..tokens import VOCAB_SIZE


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = VOCAB_SIZE        # text 32768 + image 8192 + audio 4096 + specials (see tokens.py)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    def to_dict(self):
        return asdict(self)


class Attention(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.n_head = c.n_head

    def forward(self, x, past=None, use_cache=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = [t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v)]
        if past is not None:  # KV cache: decoding a single new token (T == 1)
            k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=past is None)
        y = self.proj(y.transpose(1, 2).reshape(B, T, C))
        return (y, (k, v)) if use_cache else y


class MLP(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(c.n_embd, 4 * c.n_embd, bias=False)
        self.proj = nn.Linear(4 * c.n_embd, c.n_embd, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(c.n_embd), Attention(c)
        self.ln2, self.mlp = nn.LayerNorm(c.n_embd), MLP(c)

    def forward(self, x, past=None, use_cache=False):
        if use_cache:
            a, present = self.attn(self.ln1(x), past, True)
            x = x + a
            return x + self.mlp(self.ln2(x)), present
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.wpe = nn.Embedding(c.block_size, c.n_embd)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.lnf = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.wte.weight  # weight tying
        self.apply(self._init)
        for n, p in self.named_parameters():  # scaled init for residual projections
            if n.endswith("proj.weight"):
                nn.init.normal_(p, 0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0, 0.02)

    @classmethod
    def from_checkpoint(cls, ck, device="cpu"):
        model = cls(GPTConfig(**ck["cfg"])).to(device)
        model.load_state_dict(ck["model"])
        return model

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:  # -100 = ignored (masks prompts / padding in SFT)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    def configure_optimizer(self, lr, weight_decay=0.1, betas=(0.9, 0.95)):
        decay = [p for p in self.parameters() if p.dim() >= 2]
        nodecay = [p for p in self.parameters() if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": nodecay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=torch.cuda.is_available())

    # ---- incremental decoding (batch size 1) ----
    @torch.no_grad()
    def prefill(self, idx):
        """Run the prompt once. -> (logits for the next token (1, V), kv cache, number of tokens so far)."""
        T = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        pasts = []
        for b in self.blocks:
            x, p = b(x, None, True)
            pasts.append(p)
        return self.head(self.lnf(x[:, -1])), pasts, T

    @torch.no_grad()
    def decode_step(self, token, pasts, pos):
        """Feed one token (shape (1, 1)) that sits at position `pos`. -> (logits (1, V), updated kv cache)."""
        x = self.wte(token) + self.wpe(torch.tensor([pos], device=token.device))
        for i, b in enumerate(self.blocks):
            x, pasts[i] = b(x, pasts[i], True)
        return self.head(self.lnf(x[:, -1])), pasts
