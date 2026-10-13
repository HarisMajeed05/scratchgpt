"""Modern decoder with a legacy GPT-2 checkpoint path, GQA KV cache and configurable RoPE context."""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..tokens import VOCAB_SIZE


@dataclass
class GPTConfig:
    block_size: int = 32768
    vocab_size: int = VOCAB_SIZE  # text 32768 + image 8192 + audio 4096 + specials (see tokens.py)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    architecture: str = "modern"
    n_kv_head: int | None = None
    intermediate_size: int | None = None
    rope_theta: float = 1000000.0
    rope_scaling: float = 1.0
    gradient_checkpointing: bool = False

    def __post_init__(self):
        if self.architecture not in ("modern", "legacy"):
            raise ValueError("architecture must be modern or legacy")
        if min(self.block_size, self.n_embd, self.n_head, self.n_layer, self.vocab_size) < 1:
            raise ValueError("model dimensions must be positive")
        if self.n_embd % self.n_head or (self.n_embd // self.n_head) % 2:
            raise ValueError("embedding width must divide heads and head dimension must be even")
        if self.n_kv_head is None:
            self.n_kv_head = max(1, self.n_head // 4)
            while self.n_head % self.n_kv_head:
                self.n_kv_head -= 1
        if self.n_kv_head < 1 or self.n_head % self.n_kv_head:
            raise ValueError("query heads must be divisible by KV heads")
        if self.rope_theta <= 0 or self.rope_scaling < 1:
            raise ValueError("RoPE theta must be positive and scaling >= 1")

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.modern = c.architecture == "modern"
        self.n_head, self.n_kv_head, self.d = c.n_head, c.n_kv_head, c.n_embd // c.n_head
        if self.modern:
            self.q = nn.Linear(c.n_embd, c.n_embd, bias=False)
            self.k = nn.Linear(c.n_embd, self.n_kv_head * self.d, bias=False)
            self.v = nn.Linear(c.n_embd, self.n_kv_head * self.d, bias=False)
            inv = 1.0 / (c.rope_theta ** (torch.arange(0, self.d, 2).float() / self.d))
            self.register_buffer("inv_freq", inv, persistent=False)
            self.rope_scaling = c.rope_scaling
        else:
            self.n_kv_head = self.n_head
            self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)

    def rotate(self, x, offset):
        positions = torch.arange(offset, offset + x.size(2), device=x.device, dtype=torch.float32)
        angles = positions[:, None] * self.inv_freq[None, :] / self.rope_scaling
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)

    def forward(self, x, past=None, use_cache=False):
        B, T, C = x.shape
        if self.modern:
            q, k, v = self.q(x), self.k(x), self.v(x)
        else:
            q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d).transpose(1, 2)
        k, v = [t.view(B, T, self.n_kv_head, self.d).transpose(1, 2) for t in (k, v)]
        offset = 0 if past is None else past[0].size(2)
        if self.modern:
            q, k = self.rotate(q, offset), self.rotate(k, offset)
        if past is not None:
            k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
        cache = (k, v)
        # Native GQA avoids permanently expanding KV caches. PyTorch selects available kernels.
        kwargs = {"enable_gqa": True} if self.n_kv_head != self.n_head else {}
        if past is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, **kwargs)
        elif T == 1:
            y = F.scaled_dot_product_attention(q, k, v, **kwargs)
        else:
            mask = (
                torch.arange(k.size(2), device=x.device)[None, :]
                <= (offset + torch.arange(T, device=x.device))[:, None]
            )
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, **kwargs)
        y = self.proj(y.transpose(1, 2).reshape(B, T, C))
        return (y, cache) if use_cache else y


class MLP(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.modern = c.architecture == "modern"
        hidden = (c.intermediate_size or math.ceil(8 * c.n_embd / 3 / 64) * 64) if self.modern else 4 * c.n_embd
        self.fc = nn.Linear(c.n_embd, hidden, bias=False)
        if self.modern:
            self.gate = nn.Linear(c.n_embd, hidden, bias=False)
        self.proj = nn.Linear(hidden, c.n_embd, bias=False)

    def forward(self, x):
        y = F.silu(self.gate(x)) * self.fc(x) if self.modern else F.gelu(self.fc(x), approximate="tanh")
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln1, self.attn = (
            (RMSNorm(c.n_embd) if c.architecture == "modern" else nn.LayerNorm(c.n_embd)),
            Attention(c),
        )
        self.ln2, self.mlp = (RMSNorm(c.n_embd) if c.architecture == "modern" else nn.LayerNorm(c.n_embd)), MLP(c)

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
        if c.architecture == "legacy":
            self.wpe = nn.Embedding(c.block_size, c.n_embd)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.lnf = RMSNorm(c.n_embd) if c.architecture == "modern" else nn.LayerNorm(c.n_embd)
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
        cfg = dict(ck["cfg"])
        cfg.setdefault("architecture", "legacy")
        if ck.get("quantization") == "dynamic-int8" and torch.device(device).type != "cpu":
            raise ValueError("dynamic INT8 checkpoints require CPU")
        model = cls(GPTConfig(**cfg)).to(device)
        if ck.get("quantization") == "dynamic-int8":
            if torch.device(device).type != "cpu":
                raise ValueError("dynamic INT8 checkpoints require CPU")
            from ..inference.quantize import quantize_model

            model = quantize_model(model)
        model.load_state_dict(ck["model"])
        return model

    def embed(self, idx, offset=0):
        if idx.ndim != 2 or idx.size(1) < 1 or offset < 0 or offset + idx.size(1) > self.c.block_size:
            raise ValueError("input exceeds configured context window or is empty")
        x = self.wte(idx)
        if self.c.architecture == "legacy":
            x = x + self.wpe(torch.arange(offset, offset + idx.size(1), device=idx.device))
        return x

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.embed(idx)
        for b in self.blocks:
            if self.c.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint

                x = checkpoint(b, x, use_reentrant=False)
            else:
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
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=next(self.parameters()).device.type == "cuda")

    # ---- incremental decoding (batch size 1) ----
    @torch.no_grad()
    def prefill(self, idx):
        """Run the prompt once. -> (logits for the next token (1, V), kv cache, number of tokens so far)."""
        T = idx.shape[1]
        x = self.embed(idx)
        pasts = []
        for b in self.blocks:
            x, p = b(x, None, True)
            pasts.append(p)
        return self.head(self.lnf(x[:, -1])), pasts, T

    @torch.no_grad()
    def decode_step(self, token, pasts, pos):
        """Feed one token (shape (1, 1)) that sits at position `pos`. -> (logits (1, V), updated kv cache)."""
        if len(pasts) != len(self.blocks) or any(p[0].size(2) != pos for p in pasts):
            raise ValueError("KV cache length does not match position")
        x = self.embed(token, pos)
        for i, b in enumerate(self.blocks):
            x, pasts[i] = b(x, pasts[i], True)
        return self.head(self.lnf(x[:, -1])), pasts
