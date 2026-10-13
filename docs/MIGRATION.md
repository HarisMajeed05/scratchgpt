# Version 0.2 checkpoint migration

New `GPTConfig()` selects the modern RoPE/GQA/SwiGLU/RMSNorm model. Old checkpoints lacking
`architecture` load as `legacy` with unchanged tensors. They remain usable with the original chat/image token layout.
Never label legacy weights as modern: shapes/parameters differ. Train a new modern checkpoint and distill from a
local teacher. Existing tokenized data is compatible when the same tokenizer and VQ checkpoint are retained.

INT8 exports carry `quantization: dynamic-int8` and load only on CPU. The server disables correction training
for these exports. DPO/RL checkpoints can be served with `--checkpoint`; default checkpoint priority is unchanged.

See [UPGRADE_GUIDE.md](UPGRADE_GUIDE.md) for commands and limits.

---

# Migrating from the old flat layout

Before the restructure everything lived in one folder (`data/`, `ckpt_vq/`, `tokenizer.json`, ...). The on-disk formats
did not change, so you can move your files instead of redoing days of work.

1. **Stop everything** (Ctrl+C in each window and wait for "saved"): moving files under a running script breaks it.
2. In the new repo folder, run (PowerShell; `OLD` is your old project folder):

```powershell
$OLD = "D:\Projects\GPT"
mkdir artifacts\data, artifacts\checkpoints, artifacts\tokenizer, artifacts\live | Out-Null
Move-Item "$OLD\data"           artifacts\data\pretrain
Move-Item "$OLD\tokenizer.json" artifacts\tokenizer\tokenizer.json
if (Test-Path "$OLD\data_vq")   { Move-Item "$OLD\data_vq" artifacts\data\vq_images }
if (Test-Path "$OLD\ckpt_vq")   { Move-Item "$OLD\ckpt_vq" artifacts\checkpoints\vq }
if (Test-Path "$OLD\ckpt")      { Move-Item "$OLD\ckpt"    artifacts\checkpoints\lm }
```

3. Resume with the same commands as before, now `scratchgpt prepare-pretrain --until 90e9`, `scratchgpt vq-train ...`.

Notes
- The old `corrections.jsonl` (`{"p", "a"}` rows) is not compatible with the new multi-turn format; start fresh.
- The old `learn.py`, `serve.py`, `train.py`, `prepare_data*.py`, `common.py`, `vqvae.py`, `safeio.py` are replaced by the
  package; delete them from the old folder after moving the data.
- Old pretraining shards and VQ checkpoints are compatible: token IDs and parameter names are unchanged.
