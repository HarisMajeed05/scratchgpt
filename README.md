# scratchgpt

A modern multimodal decoder with a from-scratch student and tokenizer, local open-weight teacher distillation,
pretraining, supervised fine-tuning (SFT), preference tuning (DPO), verified-reward reinforcement learning,
FAISS retrieval, opt-in web/Python tools, CPU INT8 export and an OpenAI-compatible chat server.
Image input/output and learning from your corrections are also supported.

**This README is the complete command walkthrough.** For implementation details and limitations, see
[UPGRADE_GUIDE.md](docs/UPGRADE_GUIDE.md). For the checks actually performed, see [VALIDATION.md](docs/VALIDATION.md).

No trained student or teacher weights are bundled. The modern default has **108,989,184 parameters**, RoPE,
12 query heads / 3 KV heads, SwiGLU, RMSNorm, tied embeddings and a configurable **32,768-token context limit**.
Default pretraining uses 1,024-token sequences. Accepting longer inputs does not establish reliable long-context ability;
that requires appropriate training and evaluation. Old GPT-2 checkpoints retain their legacy architecture.

## Choose your starting point

| What you have | Where to start |
|---|---|
| No tokenizer or trained checkpoint | Installation, tokenizer, data preparation, then pretraining |
| Matching tokenizer and pretrained scratchgpt checkpoint | SFT, then distillation |
| Matching tokenizer and instruction-tuned scratchgpt checkpoint | Distillation or evaluation/serving |
| Floating-point model you are satisfied with | Evaluation, INT8 export and serving |

The recommended training order is:

**Tokenizer → pretraining data → pretraining → SFT → distillation → DPO → optional RL → evaluation → quantization → serving.**

Retrieval and tools operate during inference; they are not an additional weight-training stage. Image-tokenizer training
is optional and must be completed before preparing image-token data.

## 1. Installation

Run commands from the extracted project directory containing `pyproject.toml` and this README. Python **3.10+** is
required; Python 3.11 is a suitable environment choice. Examples below use Windows PowerShell.

### Create an environment

Choose **one** of these methods.

Using Conda:

```powershell
cd "C:\path\to\scratchgpt"
conda create -n scratchgpt python=3.11 -y
conda activate scratchgpt
```

Using Python venv:

```powershell
cd "C:\path\to\scratchgpt"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, allow it for the current terminal and retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If activation is inconvenient, use `.\.venv\Scripts\python.exe` in place of `python` and
`.\.venv\Scripts\scratchgpt.exe` in place of `scratchgpt` after installation.

### Install PyTorch first

Install the PyTorch build matching your GPU/driver. The project requires **`torch>=2.5,<2.15`**.
Do not install a CPU build if you intend to train on CUDA. Select a compatible GPU wheel from
[PyTorch's installation instructions](https://pytorch.org/get-started/locally/).

For CPU-only execution:

```powershell
python -m pip install --upgrade pip
python -m pip install "torch>=2.5,<2.15" --index-url https://download.pytorch.org/whl/cpu
```

### Install scratchgpt and its features

```powershell
python -m pip install -e ".[dev,distill,retrieval,tools]"
```

Alternatively, use the requirements file followed by an editable project installation:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

Optional local semantic embeddings:

```powershell
python -m pip install -e ".[semantic]"
```

| Dependency/extra | Purpose |
|---|---|
| Core project | PyTorch, NumPy, tokenizers, datasets, Pillow, Hugging Face Hub, FastAPI and Uvicorn |
| `dev` | pytest, httpx and Ruff |
| `distill` | `transformers>=4.51,<5`, `accelerate>=1.0`, safetensors |
| `retrieval` | `faiss-cpu>=1.8` |
| `tools` | `ddgs>=9` for web search |
| `semantic` | FAISS and `sentence-transformers>=3` |
| Docker, installed separately | Python execution tool and code-reward verification; not required for numeric math RL |

Check your installation:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
scratchgpt --help
python -m pytest -q
```

If the `scratchgpt` executable is not found, use `python -m scratchgpt` instead throughout this README.

The training examples use `--device cuda`. Replace it with `--device cpu` for commands that support that option if
CUDA is unavailable. The existing `vq-train` command is CUDA-only and has no `--device` option. CPU language-model
training is supported, but sustained training and CPU teacher generation can be slow.

### Dataset access and artifact location

Some configured datasets require accepting their access terms, including
[bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata). Authenticate when required:

```powershell
hf auth login
```

Artifacts default to `./artifacts`. To use another drive, set this before running commands:

```powershell
$env:SCRATCHGPT_HOME = "E:\scratchgpt_artifacts"
```

Default paths follow this variable. **Explicit paths such as `--base artifacts/...` in the examples do not change
automatically**: update them too if using a different location. Keep the same working directory and artifact root
between runs, and never retrain/replace the tokenizer after writing tokenized data unless you regenerate that data.

## 2. Train the tokenizer

Run once before preparing training data:

```powershell
scratchgpt train-tokenizer
```

Optional smaller tokenizer-training trial:

```powershell
scratchgpt train-tokenizer --docs 10000 --max-chars 20000
```

Output: `artifacts/tokenizer/tokenizer.json`. The smaller trial changes the tokenizer's training corpus, so select one
version and keep it fixed. All student checkpoints, binary shards and SFT data must use matching token IDs.

## 3. Prepare pretraining data

### Stream the configured text/code mix

For a first pipeline trial:

```powershell
scratchgpt prepare-pretrain --only text --until 1e8 --device cuda
```

`1e8` means approximately 100 million total tokens. This is a pipeline trial, not a claim of enough data to train a
capable general-purpose model. Binary storage is approximately two bytes per token, excluding other artifacts.

Add more text tokens to the same preparation run:

```powershell
scratchgpt prepare-pretrain --only text --add 1e9 --device cuda
```

`--until` is a cumulative total; `--add` is the number of new tokens to write in this invocation. For a longer mix run:

```powershell
scratchgpt prepare-pretrain --until 10e9 --device cuda
```

Available source names and mix ratios are in `src/scratchgpt/sources.py`. For example:

```powershell
scratchgpt prepare-pretrain --only web,py --add 1e8 --device cuda
```

Output: `artifacts/data/pretrain/train_*.bin`, `val_*.bin` and `state.json`.
Preparation resumes after Ctrl+C. Run only one preparer at a time for a given output directory.

### Alternative: quality-filter your own text corpus

Create `corpus.jsonl` with one JSON object per line:

```json
{"text":"A useful explanation or code document...","source":"source identifier","license":"applicable license"}
```

Filter, deduplicate and split it:

```powershell
scratchgpt quality-data --input corpus.jsonl --out artifacts/curated_text --min-chars 20 --max-chars 1000000 --val-fraction 0.05
```

Outputs: `train.jsonl`, `val.jsonl`, `rejected.jsonl` and `audit.json`.
The filter preserves metadata, groups matching prompts into one split, and checks length, encoding, repeated lines and
email addresses. It does not guarantee factual accuracy, remove every form of private information, or perform semantic
near-duplicate removal. Review the audit and use enough examples to obtain validation data.

Convert the cleaned text into binary shards:

```powershell
scratchgpt local-pretrain --input artifacts/curated_text --out artifacts/data/local_pretrain --shard-tokens 1000000
```

Use a fresh output directory. This importer refuses to overwrite existing binary shards. The local quality filter is
opt-in; it is not retroactively applied to the original streaming pipeline's existing shards.

## 4. Pretrain the modern student

For the initial 1,000-step trial:

```powershell
scratchgpt pretrain --data artifacts/data/pretrain --out artifacts/checkpoints/lm --context 32768 --seq-len 1024 --micro-bs 1 --accum 16 --gradient-checkpointing --device cuda --max-steps 1000
```

For sustained training, omit the step limit:

```powershell
scratchgpt pretrain --data artifacts/data/pretrain --out artifacts/checkpoints/lm --context 32768 --seq-len 1024 --micro-bs 1 --accum 16 --gradient-checkpointing --device cuda
```

If using the local corpus, change `--data` to `artifacts/data/local_pretrain`.
Pretraining needs both train and validation shards longer than the requested training sequence. If it reports waiting
for shards, check their existence and sizes.

Output: `artifacts/checkpoints/lm/latest.pt`, its `.prev` backup, and periodic milestone checkpoints.
Ctrl+C saves progress. Re-running resumes the model/optimizer. `--max-steps` is the total step target, not the number
of additional steps after resume.

For a new, smaller student experiment in a separate output directory:

```powershell
scratchgpt pretrain --data artifacts/data/pretrain --out artifacts/checkpoints/small_lm --layers 6 --heads 8 --kv-heads 2 --width 512 --context 32768 --seq-len 1024 --micro-bs 1 --accum 16 --device cuda --max-steps 1000
```

Architecture flags apply when creating a new model. Resumed checkpoints retain their saved layer/head/width structure.
Changing these flags does not convert an existing legacy checkpoint to the modern architecture.

To begin a final learning-rate decay, choose a step appropriate to your experiment:

```powershell
scratchgpt pretrain --device cuda --seq-len 1024 --micro-bs 1 --accum 16 --decay-start 50000 --decay-steps 10000
```

This last command uses the default data/checkpoint directories; specify yours if different. No fixed training budget
or convergence guarantee is established by the example values.

## 5. Instruction-tune with SFT

Prepare text-only instruction conversations:

```powershell
scratchgpt prepare-sft --out artifacts/data/sft --image-chats 0 --max-len 1024 --device cuda
```

Train from your pretrained checkpoint:

```powershell
scratchgpt sft --data artifacts/data/sft --base artifacts/checkpoints/lm/latest.pt --out artifacts/checkpoints/sft --epochs 2 --micro-bs 1 --accum 8 --device cuda
```

Output: `artifacts/checkpoints/sft/latest.pt`. This stage teaches instruction/conversation behavior. Pretraining alone
is not enough to assume a useful chat model. The shard-based SFT trainer supports resume from its output directory.

### SFT from your own conversations

Rows can contain `messages` or a `prompt`/`response` pair:

```json
{"messages":[{"role":"user","content":"What is attention?"},{"role":"assistant","content":"Attention combines information from relevant tokens."}]}
```

Filter/split them, optionally excluding untouched evaluation prompts:

```powershell
scratchgpt quality-data --input conversations.jsonl --out artifacts/chat_clean --heldout examples/heldout.jsonl
```

Import as shards into a fresh directory:

```powershell
scratchgpt prepare-sft --local-jsonl artifacts/chat_clean --out artifacts/data/local_sft --max-len 4096
scratchgpt sft --data artifacts/data/local_sft --base artifacts/checkpoints/lm/latest.pt --out artifacts/checkpoints/local_sft --micro-bs 1 --accum 8 --device cuda
```

Alternatively, train directly from JSONL:

```powershell
scratchgpt posttrain --mode sft --base artifacts/checkpoints/lm/latest.pt --data artifacts/chat_clean/train.jsonl --val artifacts/chat_clean/val.jsonl --out artifacts/checkpoints/jsonl_sft --steps 1000 --accum 4 --device cuda --gradient-checkpointing
```

Direct `posttrain` rejects overlong examples. The shard importer keeps a fitting prefix ending on an assistant turn,
and may drop examples whose first answer does not fit. Loss is applied only to assistant content and its end marker.

## 6. Distill from a local teacher

Default teacher: `Qwen/Qwen3-0.6B`. It is a compact open-weight starting point, not a claim of the best teacher on
all tasks. Select another compatible chat teacher with `--teacher` when hardware and task quality justify it.
First use downloads model files; actual answer generation runs on your machine, without a remote inference API.

Build a substantial prompts file using the format in `examples/teacher_prompts.jsonl`:

```json
{"prompt":"Explain a concept or solve a task you want the student to learn."}
```

**Bundled examples demonstrate formats; they are too small to train a useful model.** Review generated general-purpose
answers and mix them with curated human instruction data. A teacher answer is not automatically correct.

### One-command local distillation

```powershell
scratchgpt distill-local --base artifacts/checkpoints/sft/latest.pt --prompts examples/teacher_prompts.jsonl --work artifacts/distillation --teacher Qwen/Qwen3-0.6B --teacher-device cpu --student-device cuda --steps 1000
```

The command generates teacher answers, quality-filters/splits them, releases teacher memory, then trains the student.
Output: **`artifacts/distillation/student/latest.pt`**.
Use `--teacher-device cuda` when GPU memory permits. `--teacher-int8` is an optional CPU teacher quantization setting.

The student learns by predicting the teacher's next answer token using assistant-only cross-entropy. Teacher answers
are decoded and re-tokenized with the student's tokenizer. Teacher weights are not copied; this is sequence-level
rather than teacher-logit distribution distillation. Float32 0.6B teacher weights alone need roughly 2.4 GB, plus
loading overhead, activations and KV cache.

### Separate teacher generation, filtering and training

```powershell
scratchgpt teacher-data --prompts examples/teacher_prompts.jsonl --out artifacts/teacher_answers.jsonl --teacher Qwen/Qwen3-0.6B --device cpu --max-prompt 2048 --max-new 512
scratchgpt quality-data --input artifacts/teacher_answers.jsonl --out artifacts/teacher_clean --heldout examples/heldout.jsonl
scratchgpt posttrain --mode distill --base artifacts/checkpoints/sft/latest.pt --data artifacts/teacher_clean/train.jsonl --val artifacts/teacher_clean/val.jsonl --out artifacts/checkpoints/distilled --steps 1000 --accum 4 --device cuda --gradient-checkpointing
```

This separate workflow writes `artifacts/checkpoints/distilled/latest.pt`, whereas `distill-local` writes
`artifacts/distillation/student/latest.pt`. Choose one route and use the matching base path in later commands.

For an already downloaded teacher directory with network model loading disabled:

```powershell
scratchgpt teacher-data --prompts examples/teacher_prompts.jsonl --out artifacts/offline_teacher_answers.jsonl --teacher "C:\models\Qwen3-0.6B" --offline --device cpu
```

Use `--revision` to pin a Hub commit for reproducibility. Restarting teacher generation skips already committed prompt
hashes; use a new output file for a different teacher/revision. Overlong prompts and rejected answers are reported.

For answers checked against numeric references or trusted code tests:

```powershell
scratchgpt teacher-data --prompts examples/rewards.jsonl --out artifacts/verified_teacher_answers.jsonl --teacher Qwen/Qwen3-0.6B --device cpu --verify
```

`--verify` requires task metadata on every prompt. The bundled rewards include code, so this example needs Docker.
It cannot verify arbitrary general-purpose prompts.

## 7. Preference tuning with DPO

DPO data contains preferred and inferior answers to the same prompt:

```json
{"prompt":"Question...","chosen":"Preferred answer...","rejected":"Inferior answer..."}
```

Use reviewed or independently verified preferences:

```powershell
scratchgpt posttrain --mode dpo --base artifacts/distillation/student/latest.pt --data examples/preferences.jsonl --out artifacts/checkpoints/dpo --steps 200 --accum 1 --lr 1e-5 --beta 0.1 --device cuda
```

Output: `artifacts/checkpoints/dpo/latest.pt`. DPO uses summed assistant-answer log likelihoods and a frozen reference
loaded from `--base`. Preferred/rejected answers must be distinct and nonempty. Full-model tuning holds both a policy
and reference model, so allow for their combined memory. The example file is only a format demonstration.

## 8. Optional verified-reward RL

Instruction-tune first and begin with tasks the student sometimes solves. RL uses on-policy REINFORCE, a leave-one-out
baseline and categorical reference KL; it is a research baseline, not PPO or production GRPO.

Math-task format:

```json
{"prompt":"What is 12 times 13? End with Final answer: followed by the number.","kind":"math","answer":156}
```

Code-task format:

```json
{"prompt":"Write only Python code defining square(x).","kind":"code","tests":"assert square(3) == 9\nassert square(-4) == 16"}
```

Math verification accepts a final numeric line, `Final answer: number`, or `\boxed{number}`. Fractions, units and symbolic
equivalence are not supported. Code verification uses trusted Python tests and requires Docker Desktop/Engine:

```powershell
docker pull python:3.12-slim
```

Train:

```powershell
scratchgpt rlvr --base artifacts/checkpoints/dpo/latest.pt --data examples/rewards.jsonl --out artifacts/checkpoints/rl --steps 100 --samples 4 --max-new 128 --lr 1e-6 --kl 0.02 --device cuda
```

Output: `artifacts/checkpoints/rl/latest.pt`. Docker execution has no network or host-file mounts, a read-only root,
an unprivileged user and CPU/memory/process/time/output limits. There is no host execution fallback. Missing Docker
infrastructure is an error. Unit tests are correctness checks, not a guarantee against adversarial generated code.

Sparse all-zero rewards offer no positive learning signal. Inspect training reward and evaluate before/after RL.
If skipping this stage, use the DPO checkpoint in the remaining commands.

## 9. Evaluate the trained model

Use held-out prompts that have not entered pretraining, SFT, teacher generation, preference data or memory.
`examples/heldout.jsonl` is a small format demonstration; replace it with a representative untouched dataset.

Answer loss/perplexity when rows contain `response`:

```powershell
scratchgpt evaluate --checkpoint artifacts/checkpoints/rl/latest.pt --data examples/heldout.jsonl --out artifacts/eval_float_nll.json --device cpu
```

Generate answers and check math/code accuracy:

```powershell
scratchgpt evaluate --checkpoint artifacts/checkpoints/rl/latest.pt --data examples/heldout.jsonl --out artifacts/eval_float.json --generate --max-new 256 --device cpu
```

Evaluation disables correction/document retrieval. Code evaluation needs Docker. Reports include answer NLL/perplexity
when available, verified task accuracy when requested, elapsed time and generated answers. These are not automatically
large-context or broad reasoning benchmarks.

## 10. Quantize for low-resource CPU inference

Here, “post quantum training” means **post-training quantization**, not quantum computing.

```powershell
scratchgpt quantize --checkpoint artifacts/checkpoints/rl/latest.pt --out artifacts/checkpoints/int8.pt
```

Dynamic INT8 covers internal linear layers. Embeddings, the tied output head, norms and KV cache remain floating point;
expect hardware-dependent memory/speed changes, not a guaranteed 4x overall size reduction or a guaranteed speedup.
The backend uses PyTorch's deprecated `torch.ao.quantization` API within the constrained supported version range.

Compare against the floating-point model using the same held-out tasks:

```powershell
scratchgpt evaluate --checkpoint artifacts/checkpoints/int8.pt --data examples/heldout.jsonl --out artifacts/eval_int8.json --generate --max-new 256 --device cpu
```

INT8 checkpoints are **CPU-only and inference-only**. Correction fine-tuning is disabled automatically when serving
one. To train further, update the floating-point checkpoint and export again. GPU INT4/GGUF export and quantized KV
caches are not implemented.

## 11. FAISS document and correction memory

Ingest a UTF-8 `.txt`/`.md` file or directory into chunked JSONL:

```powershell
scratchgpt memory-ingest --input examples/notes.md --out artifacts/memory/documents.jsonl --chunk-chars 2000 --overlap 200
```

Document retrieval requires the `retrieval` extra. It defaults to stable lexical feature hashes with FAISS similarity
search; no embedding-model download is needed. Indexes rebuild in RAM when the durable JSONL file changes.

Correction memory uses FAISS when installed, otherwise TF-IDF. To select a backend explicitly:

```powershell
$env:SCRATCHGPT_MEMORY_BACKEND = "faiss"
```

Or:

```powershell
$env:SCRATCHGPT_MEMORY_BACKEND = "tfidf"
```

For semantic document retrieval, install the `semantic` extra and supply an already downloaded local embedding model:

```powershell
scratchgpt serve --checkpoint artifacts/checkpoints/rl/latest.pt --memory artifacts/memory/documents.jsonl --embedding-model "C:\models\my_sentence_transformer" --device cpu
```

For semantic correction retrieval, set the local embedding model path before starting the program:

```powershell
$env:SCRATCHGPT_EMBEDDING_MODEL = "C:\models\my_sentence_transformer"
```

Semantic embedding loading is local-only. Retrieved text is untrusted reference material; training and evaluation are
still needed to make a small model use it correctly.

## 12. Run with web search and Python tools

The `agent` CLI performs a bounded model → JSON tool call → result → model loop. Tools are disabled unless their flag
is passed. `examples/tool_sft.jsonl` illustrates the project's JSON call convention. Train with many varied, verified
examples to develop useful tool behavior; installing a tool does not teach the model when/how to use it.

Retrieval-only request:

```powershell
scratchgpt agent --checkpoint artifacts/checkpoints/int8.pt --prompt "How should stale tracking jobs be handled?" --memory artifacts/memory/documents.jsonl --device cpu
```

Web search:

```powershell
scratchgpt agent --checkpoint artifacts/checkpoints/int8.pt --prompt "Find official PyTorch documentation about grouped-query attention." --web --device cpu
```

Python execution, requiring Docker:

```powershell
scratchgpt agent --checkpoint artifacts/checkpoints/int8.pt --prompt "Calculate 123456 times 789012 using Python." --python --device cpu
```

Both tools plus memory:

```powershell
scratchgpt agent --checkpoint artifacts/checkpoints/int8.pt --prompt "Answer my question using available tools when needed." --web --python --memory artifacts/memory/documents.jsonl --max-rounds 3 --max-new 512 --device cpu
```

Web search returns URLs, titles and snippets rather than full pages. Queries leave your machine. The HTTP chat server
supports document retrieval but does **not** automatically execute this agent's web/Python tool loop.

## 13. Serve the model and chat

### Web UI / HTTP server

For a floating-point model with correction learning enabled:

```powershell
scratchgpt serve --checkpoint artifacts/checkpoints/rl/latest.pt --device cpu --fix-steps 60
```

For CPU INT8 and document memory:

```powershell
scratchgpt serve --checkpoint artifacts/checkpoints/int8.pt --device cpu --no-learning --memory artifacts/memory/documents.jsonl --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Without an explicit checkpoint:

```powershell
scratchgpt serve
```

Automatic checkpoint priority is **live → SFT → pretrained**. Custom distilled/DPO/RL/INT8 output locations require
`serve --checkpoint ...`; they do not become the default automatically. A live checkpoint can override a newer SFT
checkpoint during automatic selection.

### Terminal chat

```powershell
scratchgpt chat --device cpu --max-tokens 400 --temperature 0.7
```

Terminal chat uses the same default checkpoint priority; it currently has no `--checkpoint` option. Use `serve` or
`agent` to select custom checkpoints explicitly. Terminal commands:

| Command | Action |
|---|---|
| `/fix correct answer` | Fine-tune the floating-point model using the correction and replay |
| `/undo` | Restore the previous correction snapshot; up to the last three fixes in this session |
| `/clear` | Clear chat history |
| `/draw a cat` | Generate/save an image if the VQ-VAE and image-trained language model are available |
| `/quit` | Exit |

### HTTP / OpenAI-compatible client

Install the optional client package separately:

```powershell
python -m pip install openai
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="anything")
for chunk in client.chat.completions.create(
    model="scratchgpt",
    stream=True,
    messages=[{"role": "user", "content": "Write a Python function that reverses a string."}],
):
    print(chunk.choices[0].delta.content or "", end="")
```

| Endpoint | Purpose |
|---|---|
| `GET /` | Built-in chat page |
| `GET /health` | Loaded model/status information |
| `GET /v1/models` | Model listing |
| `POST /v1/chat/completions` | Text/image chat, optionally SSE streaming |
| `POST /v1/images/generations` | Image generation with a trained image-capable model |
| `POST /api/fix` | Learn a correction on a floating-point model |
| `POST /api/undo` | Undo a correction in the current session |

Teach the last answer and undo it from PowerShell:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/fix" -Method Post -ContentType "application/json" -Body '{"correction":"The corrected answer."}'
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/undo" -Method Post
```

Use the UI's “Fix last answer” button as an alternative. Corrections are stored at
`artifacts/live/corrections.jsonl`; updated weights go to `artifacts/live/live.pt`. Replay mitigates forgetting but does
not guarantee against it. Fine-tuning requires enough free memory; pause other training if necessary.

To rebuild corrections on the current default SFT/pretrained base:

```powershell
scratchgpt rebuild --steps 500 --device cpu
```

`rebuild` selects default SFT, falling back to default LM; it does not accept an arbitrary custom base checkpoint.

### Authentication and remote access

Set an API key before exposing the server beyond your machine:

```powershell
$env:API_KEY = "replace-with-a-long-random-secret"
scratchgpt serve --checkpoint artifacts/checkpoints/int8.pt --device cpu --no-learning
```

Clients send `Authorization: Bearer <key>` or `x-api-key: <key>`. In the Python client, replace `api_key="anything"` with
the actual secret. Local serving binds to `127.0.0.1` by default. If you already use Cloudflare Tunnel, an example is:

```powershell
cloudflared tunnel --url http://localhost:8000
```

The tunnel executable is a separate optional installation. Correction endpoints modify model weights when enabled.

## 14. Optional image input/output workflow

Text-only users can skip this section. Train and freeze the VQ-VAE **before writing image-token data**:

```powershell
scratchgpt vq-cache --n 200000
scratchgpt vq-train --steps 60000 --bs 48
```

The existing VQ trainer requires CUDA. Inspect `artifacts/checkpoints/vq/recon_*.png` before proceeding.
Output: `artifacts/checkpoints/vq/vq.pt`. Do not keep updating this tokenizer after generating image codes; changing
it makes those codes inconsistent.

Add image-caption tokens to pretraining data and continue pretraining:

```powershell
scratchgpt prepare-pretrain --only images --add 1e7 --device cuda
scratchgpt pretrain --data artifacts/data/pretrain --out artifacts/checkpoints/lm --seq-len 1024 --micro-bs 1 --accum 16 --device cuda
```

Prepare image instruction data in a fresh SFT directory, then tune the language model:

```powershell
scratchgpt prepare-sft --out artifacts/data/multimodal_sft --max-len 1024 --image-chats 60000 --device cuda
scratchgpt sft --data artifacts/data/multimodal_sft --base artifacts/checkpoints/lm/latest.pt --out artifacts/checkpoints/multimodal_sft --micro-bs 1 --accum 8 --device cuda
scratchgpt serve --checkpoint artifacts/checkpoints/multimodal_sft/latest.pt --device cuda
```

A VQ checkpoint alone does not teach the language model to understand/generate images. That requires the image-token
pretraining and image-chat SFT stages. Examples' image counts/token budgets are starting settings, not validated quality
requirements. Later text-only tuning can degrade image ability; retain representative multimodal replay data if needed.

Send an image using a data URL:

```python
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="anything")
encoded = base64.b64encode(Path("example.png").read_bytes()).decode("ascii")
response = client.chat.completions.create(
    model="scratchgpt",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
    ]}],
)
print(response.choices[0].message.content)
```

Generated images appear as Markdown data URLs in chat replies. The image-generation endpoint returns `b64_json`.
The VQ-VAE represents 128px images using 256 image codes; each image consumes 258 tokens including boundary markers.
Expect limited image detail, not photorealistic generation.

## 15. Longer-context training

The modern model's implementation limit defaults to 32,768, but the default training window is 1,024. After initial
training, progressively increase the actual sequence length with suitable long documents and enough memory:

```powershell
scratchgpt pretrain --seq-len 4096 --context 32768 --micro-bs 1 --accum 4 --gradient-checkpointing --device cuda
scratchgpt pretrain --seq-len 8192 --context 32768 --micro-bs 1 --accum 2 --gradient-checkpointing --device cuda
scratchgpt pretrain --seq-len 32768 --context 32768 --micro-bs 1 --accum 1 --gradient-checkpointing --device cuda
```

These commands continue the default LM checkpoint. For larger experimental context in a separate experiment:

```powershell
scratchgpt pretrain --base artifacts/checkpoints/lm/latest.pt --out artifacts/checkpoints/long_context --seq-len 32768 --context 131072 --micro-bs 1 --accum 1 --gradient-checkpointing --device cuda
```

This configures a 131,072-token limit while still training on 32,768 per example. Useful 131k behavior requires
appropriate training/evaluation at that scale; it is not established by changing a number. Existing output checkpoints
have priority over `--base` on resume, so use a fresh output directory when starting this experiment.

After extended pretraining, repeat downstream SFT/distillation as needed, using the longer-trained base and suitable
long-conversation data. Legacy learned-position checkpoints cannot be extended this way beyond their saved limit.

Attention kernels, full-vocabulary logits and activations may make long windows impractical on one consumer GPU.
Gradient checkpointing does not eliminate every allocation. Retrieval can supply relevant chunks while keeping the
active prompt short. There is no validated “largest possible” useful window or automatic long-context benchmark.

## 16. Checkpoints and resume behavior

| Stage | Example output | Restart behavior |
|---|---|---|
| Tokenizer | `artifacts/tokenizer/tokenizer.json` | Run once; do not change after tokenizing data |
| Streamed pretraining preparation | `artifacts/data/pretrain/` | Resume saved source progress |
| VQ cache/training | `artifacts/data/vq_images/`, `artifacts/checkpoints/vq/vq.pt` | Resume; freeze VQ before image-token preparation |
| Pretraining | `artifacts/checkpoints/lm/latest.pt` | Resume model/optimizer from the output directory |
| Shard SFT | `artifacts/checkpoints/sft/latest.pt` | Resume model/optimizer and step |
| Teacher generation | Chosen `--out` JSONL | Skip committed prompt hashes; recover interrupted final append |
| Local import / quality filtering | Chosen output directory | Imports require fresh shard folders; filtering regenerates its output files |
| Direct SFT/distillation/DPO | Chosen `--out/latest.pt` | Periodic weight saves; no automatic optimizer/RNG resume |
| RL | `artifacts/checkpoints/rl/latest.pt` | Periodic weight saves; no automatic optimizer/RNG resume |
| INT8 export | `artifacts/checkpoints/int8.pt` | Export again after floating-point training changes |
| Corrections | `artifacts/live/` | Durable log; last three undo snapshots are session-local |

To continue a direct posttraining run, pass its latest floating-point checkpoint as `--base` and choose a new `--out`.
This starts a new optimization run; DPO also uses that new base as its frozen reference. Do not treat it as exact resume.
Use Ctrl+C to let trainers save; a forced kill loses unsaved steps. Keep experiment output directories separate.

## 17. Every command at a glance

| Command | Purpose |
|---|---|
| `train-tokenizer` | Train the student's 32k byte-level BPE tokenizer |
| `prepare-pretrain` | Stream/tokenize the configured text/code/image data mix |
| `quality-data` | Filter/deduplicate local JSONL and split train/validation |
| `local-pretrain` | Convert curated local text JSONL to binary pretraining shards |
| `vq-cache` | Cache images for image-tokenizer training |
| `vq-train` | Train/freeze the VQ-VAE image tokenizer |
| `pretrain` | Train/continue the language-model student |
| `prepare-sft` | Prepare streamed or local instruction/image-chat shards |
| `sft` | Instruction-tune from SFT shards |
| `teacher-data` | Generate synthetic answers with a locally running teacher |
| `distill-local` | Teacher generation → filtering/splitting → student distillation |
| `posttrain` | Direct JSONL SFT, distillation or DPO |
| `rlvr` | Train with checkable math/code rewards |
| `evaluate` | Held-out answer loss and optional verified generation accuracy |
| `quantize` | Export CPU dynamic INT8 inference weights |
| `memory-ingest` | Chunk local text/Markdown documents for FAISS memory |
| `agent` | Inference with opt-in web/Python tools and document retrieval |
| `serve` | Start the web UI/OpenAI-compatible HTTP server |
| `chat` | Interactive terminal chat using default checkpoint selection |
| `rebuild` | Reapply saved corrections on the current default SFT/LM base |

List commands and inspect every option for a command:

```powershell
scratchgpt --help
scratchgpt pretrain --help
scratchgpt distill-local --help
scratchgpt posttrain --help
scratchgpt serve --help
```

The same `scratchgpt <command> --help` pattern works for all 20 commands.

## 18. Project files and development

| Path | Contents |
|---|---|
| `src/scratchgpt/model/` | Modern/legacy decoder, config and KV decoding |
| `src/scratchgpt/data/` | Tokenizer training, streaming/local preparation, quality filtering and teacher data |
| `src/scratchgpt/training/` | Pretraining, SFT, VQ training, local distillation, DPO/RL and evaluation |
| `src/scratchgpt/inference/` | Engine, correction memory, FAISS retrieval, continual learning, tools and INT8 export |
| `src/scratchgpt/server/` | HTTP API, serving command and built-in UI |
| `src/scratchgpt/cli.py` | All command registrations |
| `src/scratchgpt/paths.py` | Default artifact paths and `SCRATCHGPT_HOME` |
| `src/scratchgpt/tokens.py`, `chat_template.py` | Fixed token ranges, conversation encoding and loss masks |
| `src/scratchgpt/sources.py` | Configured datasets and pretraining mix ratios |
| `examples/` | Teacher prompts, preferences, rewards, held-out tasks, tool conversations and document memory |
| `configs/modern_32k.json` | Example modern model settings; not a CLI `--config` input |
| `tests/` | Original and upgrade regression tests |
| `docs/` | Upgrade guide, pipeline, architecture, migration, roadmap and validation record |
| `.github/workflows/ci.yml` | Lint/test workflow |
| `artifacts/` | Generated data/checkpoints/corrections; not included in the source archive |

Development checks:

```powershell
python -m pip install -e ".[dev,distill,retrieval,tools]"
python -m ruff check .
python -m pytest -q
```

The delivered code previously passed 74 tests; see [VALIDATION.md](docs/VALIDATION.md) for environment and scope.
Tests use small fixtures and do not establish full-model quality. No full student training, real-teacher distillation
benchmark, CUDA performance test or large-context quality benchmark is bundled.

## Troubleshooting and limits

| Symptom | Check/action |
|---|---|
| `scratchgpt` not found | Activate the installed environment or use `python -m scratchgpt` |
| CUDA unavailable | Check the active interpreter, installed PyTorch wheel and GPU driver; use supported CPU commands for a trial |
| Missing tokenizer | Run `train-tokenizer`; verify working directory and `SCRATCHGPT_HOME` |
| Waiting for pretraining shards | Both train/val shards must exist and be longer than `--seq-len` |
| Dataset access denied | Accept the dataset's terms and authenticate with `hf auth login` |
| Out of memory | Reduce micro-batch/sequence length; enable gradient checkpointing; run teacher on CPU; account for DPO/RL reference memory |
| Empty validation split | Use more data/check `audit.json`; prompt-group hash splitting may leave tiny datasets without validation examples |
| Long training example rejected | Shorten it or train/extend the modern model's context before tuning |
| Unexpected model served | Inspect `/health`; automatic loading prioritizes live weights; use `serve --checkpoint` |
| Tool JSON is malformed or unused | Train verified tool-use examples; tool availability alone does not confer tool competence |
| Docker/code task fails | Start Docker and ensure the local `python:3.12-slim` image exists |
| `/api/fix` unavailable on INT8 | Use a floating-point model to learn, then quantize again |
| Image output unavailable | Train VQ and image-token language-model stages; VQ alone is insufficient |

Small models can be confidently wrong. Distillation, DPO and RL can improve or degrade performance; evaluate each
stage on representative untouched data. No superiority over larger models is claimed. The image tokenizer provides
limited 128px reconstructions, speech support remains unimplemented, tool-result prompt injection is not solved by
labeling alone, and large-context ability is unmeasured.

MIT licensed. See [LICENSE](LICENSE). Dataset, teacher and embedding-model licenses are separate and must also be respected.
