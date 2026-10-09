"""Chat format. One conversation is a flat token stream:

    <|system|> text <|end|> <|user|> text [<|img|> 256 codes <|/img|>] <|end|> <|assistant|> reply <|end|> ...

The loss mask is 1 only on assistant content (and its closing <|end|>), so the model learns to answer and to stop.
A message's `content` is either a string or a list of parts:
    {"type": "text", "text": "..."}  |  {"type": "image", "codes": [256 ints in 0..8191]}
"""
from .tokenizer import tokenizer as _default_tokenizer
from .tokens import ASSISTANT, BOI, END_TURN, EOI, IMG_BASE, IMG_CODES, IMG_TOKENS, SYSTEM, TEXT_VOCAB, USER

ROLE_TOKEN = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}


def image_tokens(codes) -> list[int]:
    codes = [int(c) for c in codes]
    if len(codes) != IMG_TOKENS or not all(0 <= c < IMG_CODES for c in codes):
        raise ValueError(f"an image needs exactly {IMG_TOKENS} codes in [0, {IMG_CODES})")
    return [BOI] + [IMG_BASE + c for c in codes] + [EOI]


def encode_content(content, tok=None) -> list[int]:
    tok = tok or _default_tokenizer
    if isinstance(content, str):
        return tok.encode(content)
    ids: list[int] = []
    for part in content:
        if part["type"] == "text":
            ids += tok.encode(part["text"])
        elif part["type"] == "image":
            ids += image_tokens(part["codes"])
        else:
            raise ValueError(f"unknown content part type: {part['type']!r}")
    return ids


def encode_conversation(messages, add_generation_prompt=False, tok=None):
    """-> (ids, mask). With add_generation_prompt the stream ends with <|assistant|>, ready for the model to reply."""
    ids: list[int] = []
    mask: list[int] = []
    for m in messages:
        body = encode_content(m["content"], tok)
        learn = int(m["role"] == "assistant")
        ids += [ROLE_TOKEN[m["role"]]] + body + [END_TURN]
        mask += [0] + [learn] * (len(body) + 1)
    if add_generation_prompt:
        ids.append(ASSISTANT)
        mask.append(0)
    return ids, mask


def encode_example(context, target, tok=None):
    """Training example for ONE reply: loss only on `target` (used for corrections; earlier assistant turns are not trained)."""
    ids, _ = encode_conversation(context, tok=tok)
    body = encode_content(target, tok)
    mask = [0] * len(ids) + [0] + [1] * (len(body) + 1)
    return ids + [ASSISTANT] + body + [END_TURN], mask


def fit_conversation(messages, max_len, tok=None):
    """Longest prefix of `messages` that fits in max_len tokens and ends on an assistant turn -> (ids, mask)."""
    ids: list[int] = []
    mask: list[int] = []
    good = 0
    for m in messages:
        i, mk = encode_conversation([m], tok=tok)
        if len(ids) + len(i) > max_len:
            break
        ids += i
        mask += mk
        if m["role"] == "assistant":
            good = len(ids)
    return ids[:good], mask[:good]


def decode_reply(ids, tok=None):
    """Model output ids -> (text, [image code lists])."""
    tok = tok or _default_tokenizer
    text_ids, images, cur = [], [], None
    for t in ids:
        if t == BOI:
            cur = []
        elif t == EOI:
            if cur is not None and len(cur) == IMG_TOKENS:
                images.append(cur)
            cur = None
        elif cur is not None:
            if IMG_BASE <= t < IMG_BASE + IMG_CODES:
                cur.append(t - IMG_BASE)
        elif t < TEXT_VOCAB:
            text_ids.append(t)
    return tok.decode(text_ids), images
