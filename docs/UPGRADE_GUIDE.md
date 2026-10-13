# scratchgpt 0.2 upgrade guide

This archive contains runnable implementations, tests and example data. It contains no pretrained student checkpoint, teacher weights or completed training results. The examples illustrate formats; they are far too small to train a useful model. Start with the existing tokenizer/pretraining pipeline, or an existing scratchgpt floating-point checkpoint trained with that same tokenizer.

## What each request implements

| Request | Implementation | Practical boundary |
|---|---|---|
| Better and synthetic data | `quality-data`, `local-pretrain`, local SFT import and `teacher-data` | Exact normalized deduplication, prompt-grouped validation, optional held-out prompt exclusion, encoding/repetition/length checks and conservative email filtering. This does not certify factual correctness or remove all PII; manually review generated general-purpose answers. |
| Modern architecture | `model/gpt.py` | RoPE, GQA, SwiGLU, RMSNorm, tied embeddings, SDPA and gradient checkpointing. New default model: 108,989,184 parameters, 12 layers, width 768, 12 query heads and 3 KV heads. |
| Retrieval/tools | FAISS document/correction retrieval, `agent` and document memory in `serve` | Tools are opt-in. Native tool JSON is a project convention; train it using examples. Small models do not become reliable tool users merely because tools are installed. |
| DPO | `posttrain --mode dpo` | Assistant-only summed sequence log likelihood, distinct chosen/rejected answers and a frozen base reference. Full-model tuning. |
| Verified-reward RL | `rlvr` | On-policy REINFORCE with leave-one-out reward baseline and categorical reference KL. Numeric math or trusted code unit tests; research baseline, not PPO/GRPO. |
| Distillation | `teacher-data`, `posttrain --mode distill`, `distill-local` | Run an open-weight teacher locally, decode its answer, then train the student's next-token predictions against that text with assistant-only cross-entropy. Different tokenizers make direct teacher-logit KL inappropriate. |
| Low-resource inference | `quantize` and explicit INT8 serving | Post-training **quantization**, not quantum computing. CPU INT8 linear weights/activation quantization; embeddings, tied output head, norms and KV cache remain floating point. Hardware-dependent speed, no guaranteed speedup. |
| Large context | Configurable RoPE context, default 32,768 | No fixed 32k ceiling in the modern implementation. Longer settings require memory, longer-sequence training and task-specific evaluation. No long-context quality benchmark is claimed. |

Better data and a good teacher can improve a small model; these code changes do not establish that it beats a larger model. Benchmark the trained model against relevant baselines.

## Installation

From the extracted `scratchgpt` directory, with Python 3.10+:

```powershell
# Install the PyTorch build matching your CPU/GPU first.
# CPU example:
python -m pip install "torch>=2.5,<2.15" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,distill,retrieval,tools]"
python -m pytest -q
```

The optional `semantic` extra adds sentence-transformers. Downloads require internet on your machine; after downloading a teacher into a local directory use `--offline` to prohibit network model loading. No inference API key is needed. Web search sends the query to external search providers.

## 1. Prepare the tokenizer and base model

The original commands remain available:

```powershell
scratchgpt train-tokenizer
scratchgpt prepare-pretrain --until 1e8
scratchgpt pretrain --context 32768 --seq-len 1024 --micro-bs 1 --accum 16 --gradient-checkpointing
```

`1e8` is a pipeline trial, not a recommended sufficient token budget. Original streamed pretraining sources are unchanged; the new quality filter is opt-in for local corpora, not retroactively applied to previously generated binary shards.

For curated local pretraining text, use JSONL rows `{"text":"...","source":"...","license":"..."}`. Keep source/license metadata and obtain permission to use the text. `quality-data` retains the metadata. The importer writes fresh binary shards and refuses to overwrite an existing shard directory.

```powershell
scratchgpt quality-data --input corpus.jsonl --out artifacts/curated
scratchgpt local-pretrain --input artifacts/curated --out artifacts/data/local_pretrain
scratchgpt pretrain --data artifacts/data/local_pretrain --seq-len 1024 --context 32768 --micro-bs 1 --accum 16
```

Pretraining needs train AND validation shards longer than the requested training window. The hash split is 5% validation by default; small inputs can yield no validation rows. Use enough data and check `audit.json` before training. Each document is followed by the end-of-text token; multiple documents share a pretraining window. Boundaries do not isolate attention.

New models default to the modern architecture. Old checkpoints without an `architecture` field load as legacy GPT-2 and retain their original learned-position limit. An old model cannot acquire RoPE/GQA simply by relabeling its config: train a new modern student or distill into it. Keep existing tokenizer/image-token layouts unchanged.

## 2. Distill from a small locally running teacher

Default: `Qwen/Qwen3-0.6B`, a compact Apache-2.0 open-weight teacher with a published 32,768-token context. It is a low-memory starting point, not a claim of being the best model on every task. If your hardware permits, select another compatible chat teacher through `--teacher`; measure its answer quality before using it as supervision. A teacher must be stronger than your student on your target tasks for useful distillation. `--revision` can pin a Hub commit.

One command runs all stages, releases teacher memory, then trains the student:

```powershell
scratchgpt distill-local --base artifacts/checkpoints/lm/latest.pt --prompts examples/teacher_prompts.jsonl --work artifacts/distillation --teacher Qwen/Qwen3-0.6B --teacher-device cpu --student-device cpu --steps 1000
```

For NVIDIA training, use `--student-device cuda` (and optionally `--teacher-device cuda`). `--teacher-int8` reduces CPU teacher linear-layer memory. Float32 teacher weights alone are roughly 2.4 GB for 0.6B parameters; loading overhead, activations and the KV cache need additional RAM. CPU generation can be slow. The combined command defaults to CPU because hardware is unknown.

The standalone commands give more control:

```powershell
scratchgpt teacher-data --prompts examples/teacher_prompts.jsonl --out artifacts/teacher.jsonl --teacher Qwen/Qwen3-0.6B --device cpu
scratchgpt quality-data --input artifacts/teacher.jsonl --out artifacts/teacher_clean --heldout examples/heldout.jsonl
scratchgpt posttrain --mode distill --base artifacts/checkpoints/lm/latest.pt --data artifacts/teacher_clean/train.jsonl --val artifacts/teacher_clean/val.jsonl --out artifacts/checkpoints/distilled --device cuda --steps 1000 --accum 4 --gradient-checkpointing
```

Teacher prompts are JSONL `{"prompt":"..."}`; optional `kind`, `answer`, `tests` metadata is retained. Use `teacher-data --verify` only with checkable math/code prompts. General-purpose prompts cannot be verified by that option. Synthetic replies are not inherently correct. Mix reviewed teacher answers with curated human instruction data, including explicit tool-use demonstrations. No automatic preference labels are inferred from arbitrary teacher responses.

Generation skips already committed prompt hashes on restart, repairs an interrupted final append and rejects teacher/revision changes in the same output file. Use a new output for a changed teacher. Teacher generation defaults to greedy, non-thinking Qwen chat. Overlong prompts and low-quality/failed verified answers are skipped and reported. Each answer is re-tokenized using YOUR student's tokenizer at training time.

The student does **not** copy teacher weights. At each answer-token position, the loss penalizes the student's incorrect prediction of the teacher's next token. This is sequence-level knowledge distillation, not distribution/logit distillation. Teacher generation and student optimization occur in separate phases to reduce peak memory. Posttraining uses floating-point weights.

## 3. SFT with local synthetic or tool data

`posttrain --mode sft` directly consumes JSONL conversations (loss on all assistant turns). For the original shard-based SFT trainer:

```powershell
scratchgpt prepare-sft --local-jsonl artifacts/teacher_clean --out artifacts/data/teacher_sft --max-len 4096
scratchgpt sft --data artifacts/data/teacher_sft --base artifacts/checkpoints/lm/latest.pt --out artifacts/checkpoints/sft --device cuda
```

Local import uses the same chat tokens/loss masks as the existing system and preserves train/val labels. It keeps the longest prefix ending in an assistant turn; overlong first answers can be dropped. Direct `posttrain` rejects overlong examples instead of truncating them. Small partial batches in the original SFT trainer are now handled correctly.

`examples/tool_sft.jsonl` shows the exact JSON tool-call convention and tool-result conversation format. It is only a format sample: collect many diverse verified examples, including failures, deciding when tools are needed, citing retrieved sources and answering without tools.

## 4. Preference tuning and RL

DPO input: `{"prompt":"...","chosen":"preferred answer","rejected":"inferior answer"}`. Optional `context` is a list of chat messages ending on the user's request. Use human-reviewed or verified preferences and separate evaluation prompts.

```powershell
scratchgpt posttrain --mode dpo --base artifacts/checkpoints/distilled/latest.pt --data examples/preferences.jsonl --out artifacts/checkpoints/dpo --device cuda --steps 200 --lr 1e-5 --beta 0.1
scratchgpt rlvr --base artifacts/checkpoints/dpo/latest.pt --data examples/rewards.jsonl --out artifacts/checkpoints/rl --device cuda --steps 100 --samples 4 --max-new 128
```

Math rewards require a numeric reference and a final numeric line, `Final answer: number`, or `\boxed{number}`; no free-form model judge. Fractions, symbolic equivalence and units are currently unsupported. Code rewards require trusted Python `assert` tests. Generated code runs only in Docker with no network, mounts, capabilities or host execution fallback, a read-only root, unprivileged user, CPU/memory/process/time/output limits. Prepare Docker Desktop/Engine and pull the image once:

```powershell
docker pull python:3.12-slim
```

A missing Docker executable/image/daemon is an infrastructure error, not a zero-reward training example. Unit tests are correctness checks for curated training tasks, not an adversarial security oracle. Container security depends on your host Docker setup. Keep generated code separated from confidential files.

The REINFORCE update uses a leave-one-out reward baseline and exact categorical KL on completion-token positions. Rollouts and scored log probabilities share the same allowed-token distribution. Sparse all-zero rewards provide no positive learning signal; instruction-tune first and use solvable tasks. Evaluate before and after RL; no reasoning gain is assumed.

New `posttrain` and `rlvr` commands periodically save weights, but do not restore optimizer/RNG state on restart. To continue, provide the latest checkpoint as `--base` and a fresh `--out`; for DPO this deliberately selects a new frozen reference. Original pretraining/SFT remain resumable as before. Ctrl+C leaves a saved model; a process kill loses steps since the previous save. Separate your experiments' output directories.

## 5. FAISS memory and tools

```powershell
scratchgpt memory-ingest --input examples/notes.md --out artifacts/memory/documents.jsonl
scratchgpt agent --checkpoint artifacts/checkpoints/rl/latest.pt --prompt "How should stale tracking jobs be handled?" --memory artifacts/memory/documents.jsonl
scratchgpt agent --checkpoint artifacts/checkpoints/rl/latest.pt --prompt "Search the official PyTorch GQA documentation." --web
scratchgpt agent --checkpoint artifacts/checkpoints/rl/latest.pt --prompt "Calculate 123456 times 789012 using Python." --python
scratchgpt serve --checkpoint artifacts/checkpoints/rl/latest.pt --memory artifacts/memory/documents.jsonl
```

The `agent` CLI performs a bounded model -> JSON call -> result -> model loop. Tools execute only when their flag is enabled. Search returns titles/URLs/snippets; this is not full-page browsing. The HTTP server supports retrieved document context; its chat endpoint does not automatically run the agent's web/Python loop. Default serving without an explicit checkpoint still chooses live > sft > pretrained; DPO/RL checkpoints require `--checkpoint`.

Correction memory automatically uses FAISS when installed and otherwise retains TF-IDF. Set `SCRATCHGPT_MEMORY_BACKEND=tfidf` or `faiss` to select explicitly. FAISS defaults to stable signed lexical feature hashes (no embedding-model download). For semantic retrieval, install `[semantic]`, download a sentence-transformer separately and supply its local directory with `serve --embedding-model`, or set `SCRATCHGPT_EMBEDDING_MODEL` for correction memory. Semantic model loading is local-only. Document memory requires FAISS. Indexes are rebuilt in RAM from durable JSONL whenever that file changes; there is no separate persistent index to go stale.

Retrieved text and tool results are marked untrusted. This labeling cannot guarantee that a small model ignores prompt injection. Train and evaluate that behavior. The legacy chat format has no dedicated tool role: agent results are user turns labeled as tool data.

## 6. Export low-resource inference weights

```powershell
scratchgpt quantize --checkpoint artifacts/checkpoints/rl/latest.pt --out artifacts/checkpoints/int8.pt
scratchgpt serve --checkpoint artifacts/checkpoints/int8.pt --device cpu --no-learning --memory artifacts/memory/documents.jsonl
```

INT8 checkpoints reload through the same model/engine. Quantized serving automatically disables correction fine-tuning. To learn more, update the floating-point model then export again. CPU dynamic INT8 is implemented with PyTorch's `torch.ao.quantization`; this API is deprecated in recent PyTorch and the project constrains versions below 2.15. Test with your target version/hardware. GPU INT4/GGUF export and quantized KV caches are not implemented here.

Do not expect the entire checkpoint to become 4x smaller: the large tied embedding/head stays FP32. Dynamic activation quantization can yield small differences between full-sequence and cached logits because activation scales depend on the batch. Quantization needs no calibration dataset in this dynamic implementation; compare held-out answer loss and task accuracy after exporting.

## 7. Extend context deliberately

New student default maximum is 32,768, while default pretraining length is 1,024 to control training memory. Continue with increasing actual sequence lengths:

```powershell
scratchgpt pretrain --seq-len 4096 --context 32768 --micro-bs 1 --accum 4 --gradient-checkpointing
scratchgpt pretrain --seq-len 8192 --context 32768 --micro-bs 1 --accum 2 --gradient-checkpointing
# Experimental: only if memory and long training documents permit
scratchgpt pretrain --seq-len 32768 --context 32768 --micro-bs 1 --accum 1 --gradient-checkpointing
# No architectural hard ceiling at 32k; this is a setting, not a validated quality claim
scratchgpt pretrain --base artifacts/checkpoints/lm/latest.pt --out artifacts/checkpoints/long --seq-len 32768 --context 131072 --micro-bs 1 --accum 1 --gradient-checkpointing
```

The final command accepts up to 131,072 input tokens but trains on only 32,768 per example. To develop useful 131k behavior, train/evaluate suitable 131k tasks too. Increasing the limit does not automatically extend learned ability. `rope_scaling` is also available in `GPTConfig` for explicit experiments; it is not enabled automatically when increasing context. No YaRN/LongRoPE training recipe is claimed.

SDPA uses available kernels; on unsupported paths the math kernel can have quadratic attention-memory cost. A full-vocabulary training logit tensor is also large at long sequence lengths. Gradient checkpointing helps activation memory but not every allocation. A single consumer GPU may not fit the largest settings. Retrieval lets the model use relevant pieces of larger documents while keeping active context short.

KV-cache bytes ≈ `2 × layers × KV_heads × head_dim × tokens × batch × bytes_per_element`. For this default, 32k tokens and BF16 cache occupy approximately 288 MiB; FP32 is approximately 576 MiB, excluding weights, activations, temporary attention storage and allocator overhead. GQA cuts stored cache by 4 compared with 12 KV heads. The current cache grows with concatenation and is not paged or preallocated.

## 8. Evaluate and validate

```powershell
scratchgpt evaluate --checkpoint artifacts/checkpoints/rl/latest.pt --data examples/heldout.jsonl --out artifacts/eval_float.json --generate --device cpu
scratchgpt evaluate --checkpoint artifacts/checkpoints/int8.pt --data examples/heldout.jsonl --out artifacts/eval_int8.json --generate --device cpu
```

Evaluation disables correction/document retrieval so training leakage from memory is avoided. For real experiments replace the toy held-out examples with untouched, representative datasets. `--generate` checks math/code rewards when task metadata exists; without it, rows with `response` report assistant-answer NLL/perplexity. There is no automatic large-context benchmark: include position-stratified retrieval questions, long code tasks and short-context regression data yourself.

See `VALIDATION.md` for what was actually run on this archive. Official sources for the implementation choices:

- Qwen3-0.6B model card: https://huggingface.co/Qwen/Qwen3-0.6B
- PyTorch SDPA/GQA: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch dynamic quantization: https://docs.pytorch.org/docs/main/generated/torch.ao.quantization.quantize_dynamic.html
- DPO reference: https://huggingface.co/docs/trl/en/dpo_trainer
