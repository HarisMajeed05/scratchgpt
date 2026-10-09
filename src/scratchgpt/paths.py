"""Single source of truth for where everything lives (all of it is git-ignored).

Everything goes under ./artifacts by default; set SCRATCHGPT_HOME to put it on another drive, e.g.
    $env:SCRATCHGPT_HOME = "E:\\scratchgpt"
"""
import os
from pathlib import Path

HOME = Path(os.environ.get("SCRATCHGPT_HOME", "artifacts"))

TOKENIZER_DIR = HOME / "tokenizer"
TOKENIZER_FILE = TOKENIZER_DIR / "tokenizer.json"

DATA = HOME / "data"
PRETRAIN_DATA = DATA / "pretrain"          # train_*.bin / val_*.bin / state.json
SFT_DATA = DATA / "sft"                    # train_*.npz / val_*.npz / state.json
VQ_IMAGE_CACHE = DATA / "vq_images"        # images.npy + cache_state.json (deletable after VQ training)

CHECKPOINTS = HOME / "checkpoints"
LM_CKPT = CHECKPOINTS / "lm"               # latest.pt (+ .prev), step_*.pt milestones
SFT_CKPT = CHECKPOINTS / "sft"             # latest.pt
VQ_CKPT = CHECKPOINTS / "vq"               # vq.pt + recon_*.png

LIVE = HOME / "live"                       # the continually-learning model
LIVE_MODEL = LIVE / "live.pt"
CORRECTIONS = LIVE / "corrections.jsonl"   # permanent record of every correction you made
OUTPUTS = HOME / "outputs"                 # images saved by the terminal chat
