"""Data sources and mix ratios. Edit the `share` values to change the pretraining mix (they must sum to 1.0)."""

_SC = "bigcode/starcoderdata"  # gated: accept the terms on its Hub page, then `hf auth login`


def _text(path, share, field="text", **kw):
    return dict(kind="text", kw=dict(path=path, **kw), field=field, share=share)


def _image(path, share):
    return dict(kind="img", kw=dict(path=path), share=share)


PRETRAIN_SOURCES = {
    # general web text. "web" (a 100B-token sample) was started first; "web_full" streams the complete FineWeb-Edu and
    # takes over once the sample is used up. (The sample is a random subset of the full set, so overlap is only a few %.)
    "web": _text("HuggingFaceFW/fineweb-edu", 0.28, name="sample-100BT"),
    "web_full": _text("HuggingFaceFW/fineweb-edu", 0.29, name="default"),
    # code
    "py": _text(_SC, 0.14, "content", data_dir="python"),
    "cpp": _text(_SC, 0.07, "content", data_dir="cpp"),
    "js": _text(_SC, 0.03, "content", data_dir="javascript"),
    "ts": _text(_SC, 0.02, "content", data_dir="typescript"),
    "html": _text(_SC, 0.02, "content", data_dir="html"),
    "css": _text(_SC, 0.01, "content", data_dir="css"),
    # math and encyclopedic text (small sources: they run out and the others fill in)
    "math": _text("HuggingFaceTB/finemath", 0.05, name="finemath-4plus"),
    "wiki": _text("wikimedia/wikipedia", 0.01, name="20231101.en"),
    # image-caption pairs (bandwidth-limited: the real share will be lower than this target)
    "cc12m": _image("pixparse/cc12m-wds", 0.06),
    "cc3m": _image("pixparse/cc3m-wds", 0.02),
}

SFT_CHAT_DATASET = "HuggingFaceTB/smol-smoltalk"  # ~460k chats, Apache-2.0, built for models under 1B parameters
SFT_IMAGE_DATASET = "pixparse/cc12m-wds"  # turned into "draw X" / "describe this image" chats
