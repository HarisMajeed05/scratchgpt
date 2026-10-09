import os

from scratchgpt.safeio import atomic_json, atomic_write, clean_tmp, load_json, resume_stream, stream_state


def test_atomic_json_keeps_previous_version(tmp_path):
    p = tmp_path / "state.json"
    atomic_json(p, {"v": 1})
    atomic_json(p, {"v": 2})
    assert load_json(p) == {"v": 2}
    assert load_json(str(p) + ".prev") == {"v": 1}


def test_corrupt_main_file_falls_back_to_prev(tmp_path):
    p = tmp_path / "state.json"
    atomic_json(p, {"v": 1})
    atomic_json(p, {"v": 2})
    p.write_bytes(b"")                       # what a power cut during the old non-atomic write left behind
    assert load_json(p) == {"v": 1}


def test_crash_between_the_two_renames(tmp_path):
    p = tmp_path / "state.json"
    atomic_json(p, {"v": 1})
    atomic_json(p, {"v": 2})
    os.remove(p)                             # main file missing, .prev present
    assert load_json(p) == {"v": 1}


def test_failed_write_leaves_no_tmp_and_no_damage(tmp_path):
    p = tmp_path / "x.json"
    atomic_json(p, {"v": 3})

    def boom(f):
        raise RuntimeError("boom")

    try:
        atomic_write(p, boom)
    except RuntimeError:
        pass
    assert not os.path.exists(str(p) + ".tmp")
    assert load_json(p) == {"v": 3}


def test_clean_tmp(tmp_path):
    (tmp_path / "a.bin.tmp").write_bytes(b"x")
    (tmp_path / "keep.bin").write_bytes(b"x")
    clean_tmp(tmp_path)
    assert not (tmp_path / "a.bin.tmp").exists() and (tmp_path / "keep.bin").exists()


class _DS:
    def __init__(self, ok=True):
        self.ok, self.pos, self.skipped = ok, 0, 0

    def state_dict(self):
        return {"pos": self.pos}

    def load_state_dict(self, sd):
        if not self.ok:
            raise ValueError("nope")
        self.pos = sd["pos"]

    def skip(self, n):
        self.skipped = n
        return self


def test_resume_stream_uses_state_then_falls_back_to_skip():
    ds, exact = resume_stream(_DS, {"pos": 7}, 99)
    assert exact and ds.pos == 7 and ds.skipped == 0
    ds, exact = resume_stream(lambda: _DS(ok=False), {"pos": 7}, 99)
    assert not exact and ds.skipped == 99
    ds, exact = resume_stream(_DS, None, 0)
    assert exact
    assert stream_state(_DS()) == {"pos": 0} and stream_state(object()) is None
