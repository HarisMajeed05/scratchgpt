# Validation of this source archive

Environment: Python 3.12, PyTorch 2.14.1+cpu, Transformers 4.57.6, tokenizers 0.22.2, FAISS CPU 1.15.1, NumPy 2.3.5. No CUDA GPU or Docker executable was available.

- Full test suite: **74 passed**, including the original 54 tests and new upgrade regressions.
- Ruff lint: passed.
- Every CLI command's `--help` was checked.
- Modern and legacy model tests compare full forward with cached single-token and multi-token chunk decoding. Reduced KV-head cache shape and position/context errors are checked.
- Gradient-checkpointed model gradients match the regular model on a small fixture.
- An old checkpoint without an architecture field reloads as legacy.
- A real CPU INT8 export reloads with safe checkpoint loading and performs cached decoding. Small numerical tolerances account for batch-dependent dynamic activation scales.
- Distillation/SFT and DPO each perform a real optimizer step and change student weights. DPO preference gradient signs are tested.
- Local corpus filtering/import and a real CPU pretraining step are tested, followed by shard-based SFT with fewer examples than a full accumulated batch.
- The Transformers local-teacher loader is tested against a locally saved tiny Qwen3 architecture and tokenizer in offline mode. The tiny teacher's forward pass runs; generation is replaced by a known deterministic answer to check templating, re-tokenization, durable output and resume. This is not a quality test of downloaded Qwen3-0.6B weights.
- The RL objective computes reference KL and policy gradients. A complete RL optimizer step uses deterministic mocked rewards to exercise plumbing; the numeric verifier is separately tested on correct/incorrect answers. Held-out NLL evaluation is exercised.
- FAISS document retrieval is tested with real FAISS, including file-change index invalidation. Existing correction-store tests also pass with FAISS installed.
- Quality checks cover duplicate removal, shared-prompt validation grouping, held-out exclusion, email rejection and preservation of case-sensitive code.
- Tool opt-in/argument rejection is tested. Docker verifier success/failure and infrastructure handling use mocks; actual container execution was not run. Live web-search requests were not benchmarked.

No full student pretraining, real teacher distillation run, CUDA throughput benchmark, large-context quality benchmark or deployed Docker security test was performed. No trained model weights are included. Test fixtures are small and establish functionality rather than useful model quality.

Warnings observed: PyTorch dynamic-quantization/TypedStorage deprecations and an existing read-only NumPy image-array warning. They do not fail the tests. The PTQ backend should be migrated before moving beyond the constrained PyTorch range.
