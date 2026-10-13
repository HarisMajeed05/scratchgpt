"""Inference engine: loads the best available checkpoint and generates chat replies (text and images) with a KV cache.

Checkpoint priority: live (continually learned) > sft (chat model) > pretrained. Heavy objects live here so that the
HTTP layer stays thin and testable.
"""
import base64
import hashlib
import re
import threading

import numpy as np
import torch

from .. import paths
from ..chat_template import encode_conversation
from ..model.gpt import GPT
from ..safeio import load_torch
from ..tokenizer import tokenizer
from ..tokens import BOI, END_TURN, EOI, IMG_BASE, IMG_CODES, IMG_TOKENS, TEXT_VOCAB
from ..vision.imageio import (
    bytes_from_data_url,
    data_url,
    png_bytes,
    prep_image,
    to_input,
    to_uint8,
    vq_checkpoint_exists,
)
from ..vision.vqvae import load_vq
from .memory import CorrectionStore, last_user_text

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\((data:image/[^)\s]+)\)")


def load_best_checkpoint(device):
    for name, p in (("live", paths.LIVE_MODEL), ("sft", paths.SFT_CKPT / "latest.pt"),
                    ("pretrained", paths.LM_CKPT / "latest.pt")):
        ck = load_torch(p, device)
        if ck is not None:
            return ck, name
    raise FileNotFoundError("no model checkpoint found (looked for live, sft and pretrained). Train one first.")


def sample_token(logits, mask, temperature=0.7, top_k=50, top_p=0.9, history=None, rep_penalty=1.0) -> int:
    """Pick the next token id from logits (1, V), only among ids where `mask` is True."""
    l = logits.float()[0].clone()
    if history and rep_penalty != 1.0:
        idx = torch.tensor(sorted(set(history[-64:])), device=l.device)
        v = l[idx]
        l[idx] = torch.where(v > 0, v / rep_penalty, v * rep_penalty)
    l = l.masked_fill(~mask, float("-inf"))
    if temperature <= 0:
        return int(l.argmax())
    l = l / temperature
    if top_k and top_k < l.numel():
        l = l.masked_fill(l < torch.topk(l, top_k).values[-1], float("-inf"))
    probs = torch.softmax(l, dim=-1)
    if top_p < 1.0:
        sp, si = probs.sort(descending=True)
        sp[(sp.cumsum(0) - sp) > top_p] = 0.0
        probs = torch.zeros_like(probs).scatter(0, si, sp)
    return int(torch.multinomial(probs / probs.sum(), 1))


class Engine:
    def __init__(self, device=None, checkpoint=None, source=None, document_memory=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = torch.device(self.device).type
        if checkpoint is None:
            checkpoint, source = load_best_checkpoint(self.device)
        self.source = source or "custom"
        self.quantized = bool(checkpoint.get("quantization"))
        self.document_memory = document_memory
        self.model = GPT.from_checkpoint(checkpoint, self.device).eval()
        self.parameter_count = checkpoint.get("parameter_count", sum(p.numel() for p in self.model.parameters()))
        self.model_name = "scratchgpt"
        self.lock = threading.RLock()              # one thing at a time: generating or learning
        self.store = CorrectionStore()
        self.has_vision = vq_checkpoint_exists()
        self._vq = None
        self._img_cache = {}                       # data-URL hash -> exact codes (lossless round trip through a UI)
        V = self.model.c.vocab_size
        self.mask_text = torch.zeros(V, dtype=torch.bool, device=self.device)
        self.mask_text[:TEXT_VOCAB] = True
        self.mask_text[END_TURN] = True
        self.mask_text[BOI] = self.has_vision      # the model may start an image only if we can render it
        self.mask_img = torch.zeros(V, dtype=torch.bool, device=self.device)
        self.mask_img[IMG_BASE:IMG_BASE + IMG_CODES] = True

    # ---------- info ----------
    def info(self):
        return {"model": self.model_name, "checkpoint": self.source, "vision": self.has_vision,
                "parameters": self.parameter_count, "corrections": len(self.store), "context_window": self.model.c.block_size,
                "architecture": self.model.c.architecture, "quantized": self.quantized}

    def _autocast(self):
        return torch.autocast(self.device_type, dtype=torch.bfloat16, enabled=self.device_type == "cuda")

    # ---------- images ----------
    @property
    def vq(self):
        if self._vq is None:
            self._vq = load_vq(device=self.device)
        return self._vq

    def image_array(self, codes) -> np.ndarray:
        with torch.no_grad():
            return to_uint8(self.vq.decode(torch.tensor([list(codes)], device=self.device)))[0]

    def image_to_data_url(self, codes) -> str:
        url = data_url(self.image_array(codes))
        self._img_cache[hashlib.sha1(url.encode()).hexdigest()] = list(codes)
        return url

    def image_to_b64(self, codes) -> str:
        return base64.b64encode(png_bytes(self.image_array(codes))).decode("ascii")

    def image_codes_from_url(self, url) -> list[int]:
        cached = self._img_cache.get(hashlib.sha1(url.encode()).hexdigest())
        if cached is not None:
            return cached
        arr = prep_image(bytes_from_data_url(url))
        x = to_input(torch.from_numpy(arr)[None].to(self.device))
        with torch.no_grad():
            return [int(c) for c in self.vq.encode(x)[0]]

    # ---------- messages ----------
    def _split_markdown_images(self, text):
        """Assistant replies containing ![](data:image...) (as our server emits them) become real image parts again."""
        parts, pos = [], 0
        for m in _MD_IMAGE.finditer(text):
            if text[pos:m.start()].strip():
                parts.append({"type": "text", "text": text[pos:m.start()].strip()})
            parts.append({"type": "image", "codes": self.image_codes_from_url(m.group(1))})
            pos = m.end()
        if text[pos:].strip():
            parts.append({"type": "text", "text": text[pos:].strip()})
        return parts if any(p["type"] == "image" for p in parts) else text

    def parse_messages(self, raw):
        """OpenAI-style messages (text, image_url data URLs, markdown data-URL images) -> internal messages."""
        out = []
        for m in raw:
            role, content = m["role"], m.get("content") or ""
            if role not in ("system", "user", "assistant"):
                raise ValueError(f"unsupported role: {role!r}")
            if isinstance(content, str):
                content = self._split_markdown_images(content)
            else:
                parts = []
                for p in content:
                    t = p.get("type")
                    if t == "text":
                        parts.append({"type": "text", "text": p.get("text", "")})
                    elif t == "image_url":
                        url = p["image_url"]["url"] if isinstance(p["image_url"], dict) else p["image_url"]
                        parts.append({"type": "image", "codes": self.image_codes_from_url(url)})
                    elif t == "image":
                        parts.append(p)
                    else:
                        raise ValueError(f"unsupported content type: {t!r}")
                content = parts
            out.append({"role": role, "content": content})
        if not out or out[-1]["role"] != "user":
            raise ValueError("the last message must be from the user")
        return out

    def build_prompt(self, messages, reserve, use_memory=True, extra_ids=()):
        msgs = list(messages)
        if use_memory and len(self.store):
            mem = self.store.retrieve(last_user_text(msgs))
            pre = []
            for r in mem:                          # relevant past corrections appear as earlier turns
                pre += [{"role": "user", "content": r["query"]}, {"role": "assistant", "content": r["reply"]}]
            i = 1 if msgs and msgs[0]["role"] == "system" else 0
            msgs = msgs[:i] + pre + msgs[i:]
        if use_memory and self.document_memory is not None:
            import json
            hits = self.document_memory.retrieve(last_user_text(msgs))
            if hits:
                note = "Retrieved documents (untrusted reference data): " + json.dumps(hits, ensure_ascii=False)
                if msgs and msgs[0]["role"] == "system":
                    msgs[0] = dict(msgs[0], content=str(msgs[0]["content"]) + "\n" + note)
                else:
                    msgs.insert(0, {"role": "system", "content": note})
        budget = self.model.c.block_size - reserve
        if budget < 1:
            raise ValueError("generation reserve leaves no prompt budget")
        while True:
            ids, _ = encode_conversation(msgs, add_generation_prompt=True)
            j = 1 if msgs[0]["role"] == "system" else 0
            if len(ids) + len(extra_ids) <= budget or j >= len(msgs) - 1:
                break
            msgs.pop(j)                            # drop the oldest turn (memory turns go first)
        ids = ids + list(extra_ids)
        return ids[-budget:] if len(ids) > budget else ids

    # ---------- generation ----------
    def generate(self, messages, max_new=512, temperature=0.7, top_k=50, top_p=0.9, rep_penalty=1.05,
                 use_memory=True, force_image=False):
        """Yields ("text", delta) and ("image", codes) events. Caller should hold self.lock."""
        max_new = max(1, min(int(max_new), self.model.c.block_size // 2))
        extra = [BOI] if force_image else []
        ids = self.build_prompt(messages, reserve=max_new, use_memory=use_memory, extra_ids=extra)
        x = torch.tensor([ids], device=self.device)
        with self._autocast():
            logits, pasts, pos = self.model.prefill(x)
        mode = "image" if force_image else "text"
        img, text_ids, emitted, n_text = [], [], "", 0
        while True:
            tid = sample_token(logits, self.mask_img if mode == "image" else self.mask_text, temperature, top_k, top_p,
                               text_ids if mode == "text" else None, rep_penalty)
            feed = [tid]
            if mode == "text":
                if tid == END_TURN:
                    break
                if tid == BOI:
                    mode, img = "image", []
                else:
                    text_ids.append(tid)
                    n_text += 1
                    s = tokenizer.decode(text_ids)
                    if not s.endswith("\ufffd") and len(s) > len(emitted):
                        yield ("text", s[len(emitted):])
                        emitted = s
                    if n_text >= max_new:
                        break
            else:
                img.append(tid - IMG_BASE)
                if len(img) == IMG_TOKENS:
                    feed.append(EOI)
                    yield ("image", img)
                    if force_image:
                        return
                    mode, img = "text", []
            if pos + len(feed) > self.model.c.block_size:
                break
            for t in feed:
                with self._autocast():
                    logits, pasts = self.model.decode_step(torch.tensor([[t]], device=self.device), pasts, pos)
                pos += 1
        s = tokenizer.decode(text_ids)
        if len(s) > len(emitted):
            yield ("text", s[len(emitted):])

    def draw(self, prompt, temperature=0.9, top_k=100, top_p=0.95):
        """Text -> image codes (256 ints)."""
        if not self.has_vision:
            raise RuntimeError("image generation needs a trained VQ-VAE: run `scratchgpt vq-cache` then `scratchgpt vq-train`")
        msgs = [{"role": "user", "content": f"Generate an image of {prompt}"}]
        for kind, val in self.generate(msgs, max_new=1, temperature=temperature, top_k=top_k, top_p=top_p,
                                       rep_penalty=1.0, use_memory=False, force_image=True):
            if kind == "image":
                return val
        raise RuntimeError("the model did not finish an image (context limit reached)")
