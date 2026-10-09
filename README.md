# scratchgpt

A multimodal GPT (text, code, images) built **completely from scratch** on one consumer GPU: no API, no pretrained weights.
Everything is here: tokenizer, image tokenizer (VQ-VAE), data pipeline, pretraining, chat fine-tuning, continual
learning from your corrections, and an OpenAI-compatible server.

> **Status:** the data pipeline's crash-safety, the chat format, the server and the memory are covered by tests
> (`pytest`). The long training runs have not been completed yet, so there are no published weights or benchmark numbers.
> Expect a small, honest model, not ChatGPT-level quality (see [Limits](#limits)).

## What it does

| Capability | How |
|---|---|
| Chat (multi-turn, system prompt, streaming) | 124M GPT, pretrained on web + code + math, then instruction-tuned on [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) |
| Code | 28% of pretraining tokens (Python, C++, JS/TS, HTML, CSS) |
| Image input and output | one token stream: 128px images become 256 discrete codes (VQ-VAE) |
| Learns from your corrections | `/fix` runs a short fine-tune with replay; every correction is kept in `corrections.jsonl` and can be re-applied to a newer base model |
| Survives power cuts | atomic, fsync'ed writes with backups; data prep and training resume exactly where they stopped |
| Use from any client | `/v1/chat/completions` (OpenAI format, SSE streaming), `/v1/images/generations` |
| Speech-to-text / text-to-speech | planned: token range already reserved, see [roadmap](docs/ROADMAP.md) |

## Quick start (Windows + NVIDIA GPU)

```powershell
conda create -n scratchgpt python=3.11 -y
conda activate scratchgpt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build matching your card (RTX 50xx needs 12.8+)
pip install -e ".[dev]"
hf auth login        # the code dataset is gated: accept its terms on huggingface.co/datasets/bigcode/starcoderdata first
pytest -q            # optional sanity check
```

Run the pipeline (details, timings and disk use in [docs/PIPELINE.md](docs/PIPELINE.md)):

```powershell
scratchgpt train-tokenizer                 # 1. 32k BPE tokenizer
scratchgpt prepare-pretrain --until 90e9   # 2. stream + tokenize text/code (leave running; Ctrl+C is safe)
scratchgpt vq-cache --n 200000             # 3. in a second window: image tokenizer data ...
scratchgpt vq-train --steps 60000          #    ... and training (check artifacts/checkpoints/vq/recon_*.png)
scratchgpt prepare-pretrain                #    then re-run without --until so image-caption pairs are added
scratchgpt pretrain                        # 4. pretrain (resumable; weeks on one GPU)
scratchgpt prepare-sft                     # 5. chat + image-chat data
scratchgpt sft                             # 6. turn the base model into a chat model
scratchgpt serve                           # 7. http://127.0.0.1:8000
```

`scratchgpt --help` lists every command; `scratchgpt <command> --help` shows its options.

## Using it

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="anything")   # key only matters if you set API_KEY
for chunk in client.chat.completions.create(model="scratchgpt", stream=True,
        messages=[{"role": "user", "content": "Write a Python function that reverses a string."}]):
    print(chunk.choices[0].delta.content or "", end="")
```

Images in: send `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`. Images out come back as a
markdown `![](data:image/png;base64,...)` in the reply, which most chat UIs render.

**Teaching it:** `POST /api/fix {"correction": "the right answer"}` after a wrong reply (or the "Fix last answer" button
on the built-in page; `/fix` in `scratchgpt chat`). `POST /api/undo` rolls the last fix back.

**Exposing it** beyond your PC: set `API_KEY` first (anyone who can call `/api/fix` can change the model), then use a
tunnel such as `cloudflared tunnel --url http://localhost:8000`. The server binds to localhost only.

## Repository layout

```
scratchgpt/
├── src/scratchgpt/
│   ├── cli.py                 `scratchgpt <command>` entry point
│   ├── paths.py               where every artifact lives (override with SCRATCHGPT_HOME)
│   ├── tokens.py              token ID layout: text | image | audio | special + chat markers
│   ├── tokenizer.py           text BPE wrapper
│   ├── chat_template.py       conversation <-> tokens, loss masks, image parts
│   ├── sources.py             datasets and mix ratios (edit these)
│   ├── safeio.py              atomic writes, backups, resumable streams
│   ├── model/gpt.py           124M GPT with KV cache
│   ├── vision/                VQ-VAE image tokenizer + image helpers
│   ├── data/                  tokenizer training, pretraining prep, SFT prep + format
│   ├── training/              pretrain, sft, vq-vae training
│   ├── inference/             engine (generation), memory (corrections), continual (learning)
│   ├── server/                FastAPI app, serve command, placeholder web UI
│   └── chat_cli.py            terminal chat
├── tests/                     pytest suite (incl. simulated power-cut resume tests)
├── docs/                      PIPELINE, ARCHITECTURE, MIGRATION, ROADMAP
├── .github/workflows/ci.yml   lint + tests on every push
├── pyproject.toml  requirements.txt  LICENSE  .gitignore
└── artifacts/                 (git-ignored) tokenizer, data shards, checkpoints, live model, corrections
```

## Limits

- 124M parameters on one GPU: expect simple chat, short code snippets, and **blurry 128px images** (the VQ-VAE uses
  L1+MSE only, no GAN or perceptual loss). Fine detail, faces and text in images will not work.
- Context is 1024 tokens; one image uses 258 of them.
- It learns only from corrections you give it; it cannot tell by itself that an answer was wrong.
- Small models can be wrong with confidence. Don't rely on it for facts you can't check.

## Development

```powershell
pip install -e ".[dev]"
ruff check .
pytest -q          # torch-dependent tests run automatically when torch is installed
```

MIT licensed. See [LICENSE](LICENSE).
