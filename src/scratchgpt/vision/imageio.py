"""Image helpers: preprocessing, tensor conversion, PNG / data-URL encoding, and image-caption dataset streaming."""
import base64
import io

import numpy as np
from PIL import Image

from .. import paths
from ..tokens import IMG_SIZE

IMG_KEYS = ("jpg", "jpeg", "png", "webp", "image")
CAP_KEYS = ("txt", "caption", "text")


def vq_checkpoint_exists() -> bool:
    """True once the image tokenizer has been trained (kept here so torch-free code can ask)."""
    p = paths.VQ_CKPT / "vq.pt"
    return p.exists() or p.with_name("vq.pt.prev").exists()


def prep_image(b: bytes) -> np.ndarray:
    """Image bytes -> uint8 (IMG_SIZE, IMG_SIZE, 3): resize the short side, center crop."""
    im = Image.open(io.BytesIO(b))
    im.draft("RGB", (IMG_SIZE, IMG_SIZE))        # fast JPEG downscale-on-decode
    im = im.convert("RGB")
    w, h = im.size
    s = IMG_SIZE / min(w, h)
    im = im.resize((max(IMG_SIZE, round(w * s)), max(IMG_SIZE, round(h * s))), Image.BICUBIC)
    w, h = im.size
    left, top = (w - IMG_SIZE) // 2, (h - IMG_SIZE) // 2
    return np.asarray(im.crop((left, top, left + IMG_SIZE, top + IMG_SIZE)))


def to_input(u8):
    """(B,H,W,3) uint8 tensor -> (B,3,H,W) float in [-1,1]."""
    return u8.permute(0, 3, 1, 2).float().div(127.5).sub(1)


def to_uint8(x):
    """(B,3,H,W) float in [-1,1] -> (B,H,W,3) uint8 numpy."""
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def data_url(arr: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(arr)).decode("ascii")


def bytes_from_data_url(url: str) -> bytes:
    """Only data: URLs are accepted (the server never fetches remote URLs)."""
    if not url.startswith("data:image/") or "," not in url:
        raise ValueError("images must be sent as data:image/...;base64,... URLs")
    return base64.b64decode(url.split(",", 1)[1], validate=True)


def open_pairs(path):
    """Stream an image-caption webdataset -> (dataset, image_key, caption_key). Images come as raw bytes."""
    from datasets import Image as HFImage
    from datasets import load_dataset

    ds = load_dataset(path, split="train", streaming=True)
    feats = ds.features or {}
    ik = next((k for k in IMG_KEYS if k in feats), None)
    ck = next((k for k in CAP_KEYS if k in feats), None)
    if ik is None or ck is None:                 # features unknown: peek at one example
        ex = next(iter(ds))
        ik = ik or next(k for k in IMG_KEYS if k in ex)
        ck = ck or next(k for k in CAP_KEYS if k in ex)
    return ds.cast_column(ik, HFImage(decode=False)), ik, ck
