"""Crash-safety of the pretraining data preparation, with fake datasets (no network, no torch).

The key property: however the run is interrupted (Ctrl+C, an error, or a power cut), the shards on disk concatenated in
order equal the stream of an uninterrupted run: nothing lost, nothing duplicated.
"""
import numpy as np
import pytest

from scratchgpt.data import prepare_pretrain as pp


class FakeDS:
    """Mimics a streaming HF IterableDataset: iterable, state_dict/load_state_dict, skip."""

    def __init__(self, tag, n):
        self.tag, self.n, self.pos = tag, n, 0

    def __iter__(self):
        while self.pos < self.n:
            i = self.pos
            self.pos += 1
            yield {"text": f"{self.tag}{i:05d}-" + "y" * (i % 13 + 5)}

    def state_dict(self):
        return {"pos": self.pos}

    def load_state_dict(self, sd):
        self.pos = sd["pos"]

    def skip(self, n):
        self.pos += n
        return self


class FakeTok:
    def __init__(self, explode_at=None):
        self.calls, self.explode_at = 0, explode_at

    def encode_batch(self, texts):
        self.calls += 1
        if self.explode_at is not None and self.calls == self.explode_at:
            raise KeyboardInterrupt
        return [[ord(c) % 200 for c in t] for t in texts]


def _sources():
    return {"a": dict(kind="text", kw=dict(path="A"), field="text", share=0.5),
            "b": dict(kind="text", kw=dict(path="B"), field="text", share=0.5)}


@pytest.fixture
def env(monkeypatch):
    def setup(n_docs=100_000, tok=None):
        monkeypatch.setattr(pp, "PRETRAIN_SOURCES", _sources())
        monkeypatch.setattr(pp, "_load_dataset", lambda **kw: FakeDS(kw["path"], n_docs))
        monkeypatch.setattr(pp, "tokenizer", tok or FakeTok())
        monkeypatch.setattr(pp, "BATCH_DOCS", 8)
        monkeypatch.setattr(pp, "SHARD", 2000)
        monkeypatch.setattr(pp, "VAL_SHARD", 500)
        monkeypatch.setattr(pp, "vq_checkpoint_exists", lambda: False)
    return setup


def stream(folder):
    """All shards concatenated in creation order: val first, then train."""
    files = sorted(folder.glob("val_*.bin")) + sorted(folder.glob("train_*.bin"))
    return np.concatenate([np.fromfile(f, dtype=np.uint16) for f in files])


def reference(tmp_path, until):
    out = tmp_path / "reference"
    pp.Preparer(out, until, "cpu").run()
    return stream(out)


def test_uninterrupted_run_writes_shards_and_state(tmp_path, env):
    env()
    ref = reference(tmp_path, 20_000)
    assert len(ref) >= 20_000
    out = tmp_path / "reference"
    assert (out / "val_000000.bin").exists() and (out / "state.json").exists()
    assert not list(out.glob("*.tmp"))


def test_power_cut_resume_loses_and_duplicates_nothing(tmp_path, env):
    env()
    ref = reference(tmp_path, 20_000)
    out = tmp_path / "crashy"
    p = pp.Preparer(out, 20_000, "cpu")
    for _ in range(40):
        p.step()
    del p                                    # power cut: the buffered, unwritten tokens are simply gone
    pp.Preparer(out, 20_000, "cpu").run()
    assert np.array_equal(stream(out), ref)


def test_ctrl_c_flushes_and_resume_continues_exactly(tmp_path, env):
    env()
    ref = reference(tmp_path, 20_000)
    out = tmp_path / "ctrlc"
    env(tok=FakeTok(explode_at=25))          # KeyboardInterrupt in the middle of a batch
    pp.Preparer(out, 20_000, "cpu").run()
    assert stream(out).size > 0
    env()
    pp.Preparer(out, 20_000, "cpu").run()
    assert np.array_equal(stream(out), ref)


def test_corrupt_state_falls_back_to_backup(tmp_path, env):
    env()
    ref = reference(tmp_path, 20_000)
    out = tmp_path / "corrupt"
    pp.Preparer(out, 8_000, "cpu").run()
    (out / "state.json").write_bytes(b"")    # exactly what the original power cut did
    pp.Preparer(out, 20_000, "cpu").run()
    assert np.array_equal(stream(out), ref)


def test_lost_state_is_rebuilt_from_shards(tmp_path, env):
    env()
    out = tmp_path / "lost"
    pp.Preparer(out, 8_000, "cpu").run()
    n_train = len(list(out.glob("train_*.bin")))
    for f in out.glob("state.json*"):
        f.unlink()
    p = pp.Preparer(out, 12_000, "cpu")      # must not crash
    assert p.st["val"] == 1 and p.st["train"] + len(list(out.glob("train_*.bad"))) == n_train
    p.run()
    assert len(list(out.glob("train_*.bin"))) >= p.st["train"] - 1


def test_finite_sources_run_to_exhaustion(tmp_path, env):
    env(n_docs=50)
    out = tmp_path / "finite"
    pp.Preparer(out, 10**9, "cpu").run()
    expected = sum(len(f"{t}{i:05d}-") + (i % 13 + 5) + 1 for t in "AB" for i in range(50))
    assert len(stream(out)) == expected
