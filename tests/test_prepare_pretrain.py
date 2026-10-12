"""Crash-safety and prefetching of the pretraining data preparation, with fake datasets (no network, no torch).

Sources are consumed in a timing-dependent order (background prefetch), so the key property is checked per source:
however the run is interrupted (Ctrl+C, an error, a power cut, a corrupted state file), the documents of each source on
disk are exactly 0, 1, 2, ... in order: nothing lost, nothing duplicated.
"""
import time

import numpy as np
import pytest

from scratchgpt.data import prepare_pretrain as pp
from scratchgpt.tokens import EOT


class FakeDS:
    """Mimics a streaming HF IterableDataset: iterable, state_dict/load_state_dict, skip."""

    def __init__(self, tag, n, delay=0.0):
        self.tag, self.n, self.delay, self.pos = tag, n, delay, 0

    def __iter__(self):
        while self.pos < self.n:
            i = self.pos
            self.pos += 1
            if self.delay:
                time.sleep(self.delay)
            yield {"text": f"{self.tag}{i:05d}-" + "y" * (i % 13 + 5)}

    def state_dict(self):
        return {"pos": self.pos}

    def load_state_dict(self, sd):
        self.pos = sd["pos"]

    def skip(self, n):
        self.pos += n
        return self


class FakeTok:
    """Reversible (ASCII ids), so tests can read the documents back out of the shards."""

    def __init__(self, fail_at=None, exc=KeyboardInterrupt):
        self.calls, self.fail_at, self.exc = 0, fail_at, exc

    def encode_batch(self, texts):
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise self.exc
        return [[ord(c) for c in t] for t in texts]


def _sources():
    return {"a": dict(kind="text", kw=dict(path="A"), field="text", share=0.5),
            "b": dict(kind="text", kw=dict(path="B"), field="text", share=0.5)}


@pytest.fixture
def env(monkeypatch):
    def setup(n_docs=100_000, tok=None, delays=None):
        delays = delays or {}
        monkeypatch.setattr(pp, "PRETRAIN_SOURCES", _sources())
        monkeypatch.setattr(pp, "_load_dataset", lambda **kw: FakeDS(kw["path"], n_docs, delays.get(kw["path"], 0.0)))
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


def documents(folder):
    out, cur = [], []
    for t in stream(folder).tolist():
        if t == EOT:
            out.append("".join(map(chr, cur)))
            cur = []
        else:
            cur.append(t)
    assert not cur, "a document was cut in half"
    return out


def assert_exact(folder):
    """Per source: documents appear as 0, 1, 2, ... (no gaps, no repeats). Returns {tag: count}."""
    docs = documents(folder)
    counts = {}
    for tag in "AB":
        idx = [int(d[1:6]) for d in docs if d[0] == tag]
        assert idx == list(range(len(idx))), f"source {tag}: gaps or repeats"
        counts[tag] = len(idx)
    return counts


def test_uninterrupted_run_writes_shards_and_state(tmp_path, env):
    env()
    out = tmp_path / "ref"
    pp.Preparer(out, 20_000, "cpu").run()
    assert len(stream(out)) >= 20_000
    assert (out / "val_000000.bin").exists() and (out / "state.json").exists()
    assert not list(out.glob("*.tmp"))
    counts = assert_exact(out)
    assert counts["A"] > 0 and counts["B"] > 0


def test_power_cut_resume_loses_and_duplicates_nothing(tmp_path, env):
    env()
    out = tmp_path / "crashy"
    p = pp.Preparer(out, 20_000, "cpu")
    while p.consumed < 40:
        p.step()
    p.stop.set()
    del p                                    # power cut: the buffered, unwritten tokens are simply gone
    pp.Preparer(out, 20_000, "cpu").run()
    assert len(stream(out)) >= 20_000
    assert_exact(out)


def test_ctrl_c_flushes_and_resume_continues_exactly(tmp_path, env):
    env(tok=FakeTok(fail_at=25))             # KeyboardInterrupt inside a prefetch thread
    out = tmp_path / "ctrlc"
    pp.Preparer(out, 20_000, "cpu").run()
    first = len(stream(out))
    assert first > 0
    assert_exact(out)
    env()
    pp.Preparer(out, 20_000, "cpu").run()
    assert len(stream(out)) >= 20_000 > first
    assert_exact(out)


def test_a_crash_in_a_prefetch_thread_surfaces_flushes_and_resumes(tmp_path, env):
    env(tok=FakeTok(fail_at=25, exc=ValueError))
    out = tmp_path / "boom"
    with pytest.raises(RuntimeError, match="data stream failed"):
        pp.Preparer(out, 20_000, "cpu").run()
    assert len(stream(out)) > 0              # what was committed was still written
    assert_exact(out)
    env()
    pp.Preparer(out, 20_000, "cpu").run()
    assert_exact(out)


def test_corrupt_state_falls_back_to_backup(tmp_path, env):
    env()
    out = tmp_path / "corrupt"
    pp.Preparer(out, 8_000, "cpu").run()
    (out / "state.json").write_bytes(b"")    # exactly what the original power cut did
    pp.Preparer(out, 20_000, "cpu").run()
    assert len(stream(out)) >= 20_000
    assert_exact(out)


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
    assert assert_exact(out) == {"A": 50, "B": 50}


def test_slow_source_does_not_block_the_fast_one(tmp_path, env):
    env(delays={"B": 0.05})                  # B needs ~0.4 s per batch; A is instant
    t0 = time.time()
    p = pp.Preparer(tmp_path / "slow", 6_000, "cpu")
    p.run()
    assert time.time() - t0 < 4              # a blocking design would wait ~8 s for B's share
    assert p.ctoks["a"] > 3 * p.ctoks["b"]
    assert_exact(tmp_path / "slow")
