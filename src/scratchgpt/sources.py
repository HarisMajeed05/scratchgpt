"""Data sources and mix ratios. Edit the `share` values to change the pretraining mix (they must sum to 1.0)."""

_SC = "bigcode/starcoderdata"          # gated: accept the terms on its Hub page, then `hf auth login`


def _text(path, share, field="text", **kw):
    return dict(kind="text", kw=dict(path=path, **kw), field=field, share=share)


def _image(path, share):
    return dict(kind="img", kw=dict(path=path), share=share)


PRETRAIN_SOURCES = {
    "web":   _text("HuggingFaceFW/fineweb-edu", 0.50, name="sample-100BT"),
    "py":    _text(_SC, 0.15, "content", data_dir="python"),
    "cpp":   _text(_SC, 0.08, "content", data_dir="cpp"),
    "js":    _text(_SC, 0.02, "content", data_dir="javascript"),
    "ts":    _text(_SC, 0.02, "content", data_dir="typescript"),
    "html":  _text(_SC, 0.02, "content", data_dir="html"),
    "css":   _text(_SC, 0.01, "content", data_dir="css"),
    "math":  _text("HuggingFaceTB/finemath", 0.05, name="finemath-4plus"),
    "wiki":  _text("wikimedia/wikipedia", 0.03, name="20231101.en"),
    "cc12m": _image("pixparse/cc12m-wds", 0.09),      # image-caption pairs
    "cc3m":  _image("pixparse/cc3m-wds", 0.03),
}

SFT_CHAT_DATASET = "HuggingFaceTB/smol-smoltalk"      # ~460k chats, Apache-2.0, built for models under 1B parameters
SFT_IMAGE_DATASET = "pixparse/cc12m-wds"              # turned into "draw X" / "describe this image" chats
