"""Needs torch."""
import pytest

torch = pytest.importorskip("torch")

from scratchgpt.tokens import IMG_CODES, IMG_TOKENS  # noqa: E402
from scratchgpt.vision.imageio import data_url, prep_image, to_input, to_uint8  # noqa: E402
from scratchgpt.vision.vqvae import VQVAE  # noqa: E402


def test_vqvae_encode_decode_shapes_and_range():
    vq = VQVAE().eval()
    x = torch.rand(1, 3, 128, 128) * 2 - 1
    codes = vq.encode(x)
    assert codes.shape == (1, IMG_TOKENS) and int(codes.min()) >= 0 and int(codes.max()) < IMG_CODES
    img = vq.decode(codes)
    assert img.shape == (1, 3, 128, 128) and float(img.abs().max()) <= 1.0


def test_image_helpers_round_trip():
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), (255, 0, 0)).save(buf, format="PNG")
    arr = prep_image(buf.getvalue())
    assert arr.shape == (128, 128, 3) and arr[64, 64, 0] > 200
    assert data_url(arr).startswith("data:image/png;base64,")
    back = to_uint8(to_input(torch.from_numpy(arr)[None]))
    assert abs(int(back[0, 64, 64, 0]) - int(arr[64, 64, 0])) <= 1
