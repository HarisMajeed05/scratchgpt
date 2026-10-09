"""Needs torch (runs in CI and on your machine; skipped where torch is missing)."""
import pytest

torch = pytest.importorskip("torch")

from scratchgpt.inference.engine import Engine, sample_token  # noqa: E402
from scratchgpt.model.gpt import GPT, GPTConfig  # noqa: E402
from scratchgpt.tokens import BOI, END_TURN, IMG_BASE, IMG_CODES, TEXT_VOCAB, VOCAB_SIZE  # noqa: E402


def tiny_cfg():
    return GPTConfig(block_size=64, n_layer=2, n_head=2, n_embd=32)


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    model = GPT(tiny_cfg()).eval()
    idx = torch.randint(0, 1000, (1, 12))
    with torch.no_grad():
        full, _ = model(idx)
    logits, pasts, pos = model.prefill(idx[:, :5])
    assert torch.allclose(logits[0], full[0, 4], atol=1e-4)
    for t in range(5, 12):
        logits, pasts = model.decode_step(idx[:, t:t + 1], pasts, pos)
        pos += 1
        assert torch.allclose(logits[0], full[0, t], atol=1e-4), t


def test_ignored_targets_do_not_contribute_to_the_loss():
    model = GPT(tiny_cfg()).eval()
    x = torch.randint(0, 1000, (2, 10))
    y = x.clone()
    y[:, :5] = -100
    _, loss_a = model(x, y)
    y2 = y.clone()
    _, loss_b = model(x, y2)
    assert torch.isfinite(loss_a) and torch.allclose(loss_a, loss_b)


def test_sample_token_respects_mask_and_greedy():
    logits = torch.zeros(1, 100)
    logits[0, 7] = 5.0
    mask = torch.zeros(100, dtype=torch.bool)
    mask[[3, 9]] = True
    assert sample_token(logits, mask, temperature=0) in (3, 9)
    mask[7] = True
    assert sample_token(logits, mask, temperature=0) == 7
    assert all(sample_token(logits, mask, temperature=1.0, top_k=1) == 7 for _ in range(5))


@pytest.fixture
def engine(tmp_path, tiny_tok, monkeypatch):
    from scratchgpt import paths
    from scratchgpt.inference import engine as eng_mod
    from scratchgpt.tokenizer import tokenizer

    monkeypatch.setattr(tokenizer, "_tok", tiny_tok._load())
    monkeypatch.setattr(paths, "CORRECTIONS", tmp_path / "corrections.jsonl")
    monkeypatch.setattr(paths, "LIVE_MODEL", tmp_path / "live.pt")
    monkeypatch.setattr(eng_mod, "vq_checkpoint_exists", lambda: False)
    torch.manual_seed(0)
    cfg = GPTConfig(block_size=256, n_layer=2, n_head=2, n_embd=32, vocab_size=VOCAB_SIZE)
    ck = {"model": GPT(cfg).state_dict(), "cfg": cfg.to_dict()}
    return Engine(device="cpu", checkpoint=ck, source="test")


def test_engine_masks(engine):
    assert engine.mask_text[:TEXT_VOCAB].all() and engine.mask_text[END_TURN] and not engine.mask_text[BOI]
    assert engine.mask_img[IMG_BASE:IMG_BASE + IMG_CODES].all() and not engine.mask_img[:IMG_BASE].any()


def test_engine_generates_text_deterministically_when_greedy(engine):
    msgs = [{"role": "user", "content": "hello"}]
    run = lambda: [e for e in engine.generate(msgs, max_new=8, temperature=0)]  # noqa: E731
    a, b = run(), run()
    assert a == b and all(kind == "text" for kind, _ in a)


def test_parse_messages_validation(engine):
    assert engine.parse_messages([{"role": "user", "content": "hi"}])[0]["content"] == "hi"
    with pytest.raises(ValueError):
        engine.parse_messages([{"role": "assistant", "content": "hi"}])
    with pytest.raises(ValueError):
        engine.parse_messages([{"role": "tool", "content": "hi"}])


def test_long_history_is_trimmed_to_fit(engine):
    msgs = []
    for i in range(60):
        msgs += [{"role": "user", "content": f"hello world {i}"}, {"role": "assistant", "content": "hello world"}]
    msgs.append({"role": "user", "content": "last question"})
    ids = engine.build_prompt(msgs, reserve=64)
    assert len(ids) <= engine.model.c.block_size - 64


def test_fix_learns_logs_and_can_be_undone(engine):
    from scratchgpt.inference.continual import ContinualLearner

    learner = ContinualLearner(engine, steps=2)
    before = engine.model.wte.weight.detach().clone()
    ctx = engine.parse_messages([{"role": "user", "content": "what is the capital of France"}])
    learner.fix(ctx, "Paris")
    assert len(engine.store) == 1 and not torch.equal(before, engine.model.wte.weight)
    assert learner.undo() is True
    assert len(engine.store) == 0 and torch.equal(before, engine.model.wte.weight)
    assert learner.undo() is False
