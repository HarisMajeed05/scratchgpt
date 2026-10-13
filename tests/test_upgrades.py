"""Regression coverage for GQA/RoPE, data splits, DPO/RL objectives, PTQ and retrieval."""

import json

import numpy as np
import pytest
import torch

from scratchgpt.model.gpt import GPT, GPTConfig
from scratchgpt.safeio import load_torch
from scratchgpt.tokens import VOCAB_SIZE


def tiny_cfg(**kw):
    return GPTConfig(n_layer=1, n_head=4, n_kv_head=1, n_embd=32, block_size=256, **kw)


@pytest.mark.parametrize("architecture", ["modern", "legacy"])
def test_cache_chunk_and_single_token_match_full(architecture):
    torch.manual_seed(5)
    model = GPT(tiny_cfg(architecture=architecture, vocab_size=256)).eval()
    ids = torch.randint(0, 256, (2, 19))
    with torch.no_grad():
        full = model(ids)[0]
    logits, cache, pos = model.prefill(ids[:, :7])
    assert torch.allclose(logits, full[:, 6], atol=1e-5)
    chunk = model.embed(ids[:, 7:11], 7)
    next_cache = []
    for b, past in zip(model.blocks, cache):
        chunk, updated = b(chunk, past, True)
        next_cache.append(updated)
    assert torch.allclose(model.head(model.lnf(chunk)), full[:, 7:11], atol=1e-5)
    for i in range(11, 19):
        logits, next_cache = model.decode_step(ids[:, i : i + 1], next_cache, i)
        assert torch.allclose(logits, full[:, i], atol=1e-5)
    assert next_cache[0][0].shape[1] == (1 if architecture == "modern" else 4)


def test_context_extension_and_limit():
    model = GPT(tiny_cfg(vocab_size=128)).eval()
    model.c.block_size = 512
    assert model(torch.zeros(1, 300, dtype=torch.long))[0].shape == (1, 300, 128)
    with pytest.raises(ValueError, match="context"):
        model(torch.zeros(1, 513, dtype=torch.long))
    with pytest.raises(ValueError, match="KV cache"):
        _, cache, _ = model.prefill(torch.zeros(1, 3, dtype=torch.long))
        model.decode_step(torch.zeros(1, 1, dtype=torch.long), cache, 2)


def test_old_checkpoint_without_architecture_is_legacy():
    config = tiny_cfg(architecture="legacy", vocab_size=128)
    ck = {
        "cfg": {
            k: v
            for k, v in config.to_dict().items()
            if k in ("block_size", "vocab_size", "n_layer", "n_head", "n_embd")
        },
        "model": GPT(config).state_dict(),
    }
    assert GPT.from_checkpoint(ck).c.architecture == "legacy"


def test_gradient_checkpointing_matches_gradients():
    cfg = tiny_cfg(vocab_size=128)
    base = GPT(cfg)
    other = GPT.from_checkpoint({"cfg": cfg.to_dict(), "model": base.state_dict()})
    other.c.gradient_checkpointing = True
    x = torch.randint(0, 128, (1, 12))
    for model in (base, other):
        model(x, x)[1].backward()
    for a, b in zip(base.parameters(), other.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-6)


def test_quantized_checkpoint_roundtrip_and_cache(tmp_path):
    from scratchgpt.inference.quantize import main

    model = GPT(tiny_cfg(vocab_size=256)).eval()
    base, dest = tmp_path / "base.pt", tmp_path / "int8.pt"
    torch.save({"cfg": model.c.to_dict(), "model": model.state_dict()}, base)
    main(["--checkpoint", str(base), "--out", str(dest)])
    ck = load_torch(dest)
    assert ck is not None and ck["quantization"] == "dynamic-int8"
    quant = GPT.from_checkpoint(ck).eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        expected = quant(x)[0]
        float_logits = model(x)[0]
    assert (expected - float_logits).abs().mean() < 0.05
    _, cache, pos = quant.prefill(x[:, :7])
    logits, _ = quant.decode_step(x[:, 7:], cache, pos)
    # Dynamic activation scales depend on the evaluated token batch.
    assert (expected[:, -1] - logits).abs().max() < 0.02
    with pytest.raises(ValueError, match="CPU"):
        GPT.from_checkpoint(ck, "cuda")


def test_quality_groups_duplicates_pii_and_holdout(tmp_path):
    from scratchgpt.data.quality import digest, main, read_rows

    source = tmp_path / "input.jsonl"
    rows = [
        {
            "prompt": "How does a transformer work?",
            "response": "It processes tokens using attention and feedforward layers.",
        },
        {"prompt": "How does a transformer work?", "response": "Attention combines information from other tokens."},
        {
            "prompt": "How does a transformer work?",
            "response": "It processes tokens using attention and feedforward layers.",
        },
        {"prompt": "Contact person please", "response": "Write to private@example.com for information."},
        {"prompt": "What is gravity?", "response": "Mass and energy curve spacetime."},
    ]
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    heldout = tmp_path / "held.jsonl"
    heldout.write_text(json.dumps({"prompt": "What is gravity?"}) + "\n")
    out = tmp_path / "quality"
    main(["--input", str(source), "--out", str(out), "--heldout", str(heldout)])
    audit = json.loads((out / "audit.json").read_text())
    assert audit["duplicate"] == 1 and audit["email_pii"] == 1 and audit["heldout_overlap"] == 1
    train, val = list(read_rows(out / "train.jsonl")), list(read_rows(out / "val.jsonl"))
    assert len(train) + len(val) == 2 and (len(train) == 0 or len(val) == 0)
    assert all(r["prompt_hash"] == digest("How does a transformer work?") for r in train + val)


def test_text_quality_input(tmp_path):
    from scratchgpt.data.quality import main, read_rows

    p = tmp_path / "corpus.jsonl"
    p.write_text(
        json.dumps({"text": "Useful textbook explanation of current and voltage.", "source": "test", "license": "CC0"})
        + "\n"
    )
    main(["--input", str(p), "--out", str(tmp_path / "out")])
    rows = list(read_rows(tmp_path / "out/train.jsonl")) + list(read_rows(tmp_path / "out/val.jsonl"))
    assert len(rows) == 1 and rows[0]["license"] == "CC0"


def test_dpo_loss_direction():
    from scratchgpt.training.posttrain import dpo_loss

    pc, pr = torch.tensor(-3.0, requires_grad=True), torch.tensor(-4.0, requires_grad=True)
    loss = dpo_loss(pc, pr, torch.tensor(-3.0), torch.tensor(-4.0), 0.1)
    loss.backward()
    assert pc.grad < 0 and pr.grad > 0
    assert torch.allclose(loss, torch.tensor(float(np.log(2)), dtype=loss.dtype), atol=1e-6)


@pytest.mark.parametrize("mode", ["distill", "dpo"])
def test_posttrain_actually_updates_weights(tmp_path, tiny_tok, monkeypatch, mode):
    from scratchgpt.tokenizer import tokenizer
    from scratchgpt.training.posttrain import main

    monkeypatch.setattr(tokenizer, "_tok", tiny_tok._load())
    model = GPT(tiny_cfg(vocab_size=VOCAB_SIZE))
    base = tmp_path / "base.pt"
    torch.save({"cfg": model.c.to_dict(), "model": model.state_dict()}, base)
    row = {"prompt": "what is the capital of France?", "response": "Paris", "chosen": "Paris", "rejected": "London"}
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps(row) + "\n")
    main(
        [
            "--mode",
            mode,
            "--base",
            str(base),
            "--data",
            str(data),
            "--out",
            str(tmp_path / "out"),
            "--steps",
            "1",
            "--accum",
            "1",
            "--device",
            "cpu",
        ]
    )
    trained = load_torch(tmp_path / "out/latest.pt")
    assert trained["stage"] == mode and not torch.equal(trained["model"]["wte.weight"], model.wte.weight)


def test_numeric_verification():
    from scratchgpt.training.verifiers import verify

    task = {"kind": "math", "answer": 42}
    assert verify(task, "Reasoning has the number 42 but the final answer is wrong.\n0") == 0
    assert verify(task, "Reasoning\nFinal answer: 42") == 1
    assert verify(task, r"\boxed{42}") == 1
    assert verify(task, "nan") == 0
    with pytest.raises(ValueError):
        verify({"kind": "math"}, "42")


def test_rl_likelihood_and_gradient(tmp_path, tiny_tok, monkeypatch):
    from scratchgpt.tokenizer import tokenizer
    from scratchgpt.training.rl import policy_stats, rollout

    monkeypatch.setattr(tokenizer, "_tok", tiny_tok._load())
    model = GPT(tiny_cfg(vocab_size=VOCAB_SIZE)).eval()
    ref = GPT.from_checkpoint({"cfg": model.c.to_dict(), "model": model.state_dict()}).eval().requires_grad_(False)
    seq, start, _ = rollout(model, "hello world", "cpu", 4)
    lp, kl = policy_stats(model, ref, seq, start, "cpu")
    assert kl.abs() < 1e-5 and torch.isfinite(lp)
    (-lp).backward()
    assert model.wte.weight.grad.abs().sum() > 0


def test_faiss_document_rebuild_and_similarity(tmp_path):
    pytest.importorskip("faiss")
    from scratchgpt.inference.retrieval import DocumentMemory

    p = tmp_path / "memory.jsonl"
    p.write_text(json.dumps({"text": "France capital Paris"}) + "\n")
    mem = DocumentMemory(p)
    assert mem.retrieve("France capital")[0]["text"] == "France capital Paris"
    p.write_text(json.dumps({"text": "Python code programming"}) + "\n")
    assert mem.retrieve("Python code")[0]["text"] == "Python code programming"
    assert not mem.retrieve("France capital", threshold=0.9)


def test_tools_require_optin_and_never_execute_host_code():
    from scratchgpt.inference.tools import ToolRegistry

    with pytest.raises(ValueError, match="disabled"):
        ToolRegistry().execute({"tool": "python", "arguments": {"code": "print(1)"}})
    with pytest.raises(ValueError):
        ToolRegistry(python=True).execute({"tool": "python", "arguments": {"code": "print(1)", "command": "rm"}})


def test_local_pretrain_and_small_sft_batch(tmp_path, tiny_tok, monkeypatch):
    from scratchgpt.data.local_pretrain import main as import_corpus
    from scratchgpt.data.prepare_sft import main as import_chat
    from scratchgpt.tokenizer import tokenizer
    from scratchgpt.training.pretrain import main as pretrain
    from scratchgpt.training.sft import main as sft

    monkeypatch.setattr(tokenizer, "_tok", tiny_tok._load())
    corpus = tmp_path / "text.jsonl"
    corpus.write_text(
        "".join(
            json.dumps({"text": "hello world voltage and current " * 50, "split": kind}) + "\n"
            for kind in ["train", "val"]
        )
    )
    import_corpus(["--input", str(corpus), "--out", str(tmp_path / "bins"), "--shard-tokens", "2000"])
    pretrain(
        [
            "--data",
            str(tmp_path / "bins"),
            "--out",
            str(tmp_path / "lm"),
            "--device",
            "cpu",
            "--max-steps",
            "1",
            "--seq-len",
            "16",
            "--context",
            "128",
            "--layers",
            "1",
            "--heads",
            "4",
            "--kv-heads",
            "1",
            "--width",
            "32",
            "--micro-bs",
            "1",
            "--accum",
            "1",
        ]
    )
    base = tmp_path / "lm/latest.pt"
    assert load_torch(base)["step"] == 1
    chats = tmp_path / "chats.jsonl"
    chats.write_text(
        "".join(
            json.dumps({"prompt": "hello world", "response": "Hello world!", "split": kind}) + "\n"
            for kind in ["train", "val"]
        )
    )
    import_chat(["--local-jsonl", str(chats), "--out", str(tmp_path / "sft_data"), "--max-len", "128"])
    sft(
        [
            "--data",
            str(tmp_path / "sft_data"),
            "--base",
            str(base),
            "--out",
            str(tmp_path / "sft_model"),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--micro-bs",
            "8",
            "--accum",
            "4",
            "--eval-every",
            "1",
        ]
    )
    assert load_torch(tmp_path / "sft_model/latest.pt")["step"] == 1


def test_rlvr_training_step_and_evaluation(tmp_path, tiny_tok, monkeypatch):
    from scratchgpt import paths
    from scratchgpt.tokenizer import tokenizer
    from scratchgpt.training import rl
    from scratchgpt.training.evaluate import main as evaluate

    monkeypatch.setattr(tokenizer, "_tok", tiny_tok._load())
    monkeypatch.setattr(paths, "CORRECTIONS", tmp_path / "corrections.jsonl")
    model = GPT(tiny_cfg(vocab_size=VOCAB_SIZE))
    base = tmp_path / "base.pt"
    torch.save({"cfg": model.c.to_dict(), "model": model.state_dict()}, base)
    task = tmp_path / "tasks.jsonl"
    task.write_text(json.dumps({"prompt": "What is 1 + 1?", "answer": 2, "kind": "math", "response": "2"}) + "\n")
    rewards = iter([0.0, 1.0])
    monkeypatch.setattr(rl, "verify", lambda row, response: next(rewards))
    rl.main(
        [
            "--base",
            str(base),
            "--data",
            str(task),
            "--out",
            str(tmp_path / "rl"),
            "--steps",
            "1",
            "--samples",
            "2",
            "--max-new",
            "4",
            "--device",
            "cpu",
        ]
    )
    ck = load_torch(tmp_path / "rl/latest.pt")
    assert ck["step"] == 1 and not torch.equal(ck["model"]["wte.weight"], model.wte.weight)
    report = tmp_path / "eval.json"
    evaluate(
        ["--checkpoint", str(tmp_path / "rl/latest.pt"), "--data", str(task), "--out", str(report), "--device", "cpu"]
    )
    assert json.loads(report.read_text())["answer_nll"] > 0


def test_local_teacher_loader_and_resume(tmp_path, tiny_tok, monkeypatch):
    pytest.importorskip("transformers")
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from scratchgpt.data.distill import main
    from scratchgpt.data.quality import read_rows

    local = tmp_path / "teacher"
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tiny_tok._load(), eos_token="<eos>", pad_token="<pad>")
    hf_tok.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    hf_tok.save_pretrained(local)
    config = Qwen3Config(
        vocab_size=len(hf_tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    teacher = Qwen3ForCausalLM(config)
    teacher.save_pretrained(local)
    # Deterministic known answer isolates local loading, prompt templating, retokenization and durable resume.
    calls = []

    def generated(self, input_ids, **kwargs):
        calls.append(1)
        assert torch.isfinite(self(input_ids).logits).all()
        answer = torch.tensor([hf_tok.encode("The capital of France is Paris.", add_special_tokens=False)])
        return torch.cat((input_ids, answer), 1)

    monkeypatch.setattr(Qwen3ForCausalLM, "generate", generated)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"prompt": "what is the capital of France?"}) + "\n")
    output = tmp_path / "teacher_data.jsonl"
    argv = [
        "--prompts",
        str(prompts),
        "--out",
        str(output),
        "--teacher",
        str(local),
        "--offline",
        "--max-prompt",
        "128",
        "--max-new",
        "64",
    ]
    main(argv)
    main(argv)
    rows = list(read_rows(output))
    assert len(calls) == 1 and len(rows) == 1 and rows[0]["response"] == "The capital of France is Paris."


def test_docker_verifier_success_failure_and_infrastructure(monkeypatch):
    from scratchgpt.training import verifiers

    def success(code):
        # Use the harness's success marker without executing any generated code.
        marker = code.split("print(")[-1].split(")")[0].strip("'\"")
        return {"ok": True, "output": marker, "returncode": 0}

    monkeypatch.setattr(verifiers, "run_python", success)
    task = {"kind": "code", "tests": "assert add(2, 3) == 5"}
    assert verifiers.verify(task, "```python\ndef add(a,b): return a+b\n```") == 1
    monkeypatch.setattr(
        verifiers, "run_python", lambda code: {"ok": False, "output": "AssertionError", "returncode": 1}
    )
    assert verifiers.verify(task, "def add(a,b): return 0") == 0
    monkeypatch.setattr(verifiers, "run_python", lambda code: {"ok": False, "output": "daemon down", "returncode": 125})
    with pytest.raises(RuntimeError, match="infrastructure"):
        verifiers.verify(task, "def add(a,b): return a+b")


def test_quality_preserves_case_sensitive_code(tmp_path):
    from scratchgpt.data.quality import main, read_rows

    p = tmp_path / "code.jsonl"
    p.write_text(
        "".join(
            json.dumps({"text": text}) + "\n"
            for text in ["FLAG = True\nprint(FLAG)\n# code example", "flag = True\nprint(flag)\n# code example"]
        )
    )
    main(["--input", str(p), "--out", str(tmp_path / "out")])
    rows = list(read_rows(tmp_path / "out/train.jsonl")) + list(read_rows(tmp_path / "out/val.jsonl"))
    assert len(rows) == 2
