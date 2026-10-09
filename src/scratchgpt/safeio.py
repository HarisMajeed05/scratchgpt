"""Crash-safe I/O helpers. A power cut leaves either the old file or the new one, never half of a file.

Every write goes to <file>.tmp, is fsync'ed to disk, and only then renamed over the real file (rename is atomic).
The previous good version is kept as <file>.prev, and loaders fall back to it if the main file is unreadable.
"""
import glob
import json
import os


def atomic_write(path, writer, keep_prev=True):
    """writer(file_obj) writes the content."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        if keep_prev and os.path.exists(path):
            os.replace(path, path + ".prev")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def atomic_json(path, obj, keep_prev=True):
    atomic_write(path, lambda f: f.write(json.dumps(obj).encode("utf-8")), keep_prev)


def load_json(path):
    path = os.fspath(path)
    for p in (path, path + ".prev"):
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"{p} unreadable ({type(e).__name__}); trying backup", flush=True)
    return None


def load_torch(path, map_location="cpu"):
    import torch

    path = os.fspath(path)
    for p in (path, path + ".prev"):
        if os.path.exists(p):
            try:
                return torch.load(p, map_location=map_location)
            except Exception as e:
                print(f"{p} unreadable ({type(e).__name__}); trying backup", flush=True)
    return None


def clean_tmp(folder):
    """Remove half-written .tmp files left behind by a crash."""
    for p in glob.glob(os.path.join(os.fspath(folder), "*.tmp")):
        try:
            os.remove(p)
        except OSError:
            pass


def stream_state(ds):
    """JSON-safe resume point of a streaming HF dataset, or None if unsupported."""
    try:
        sd = ds.state_dict()
        json.dumps(sd)
        return sd
    except Exception:
        return None


def resume_stream(make, state, skip):
    """make() -> fresh streaming dataset. Restore its position from a saved state_dict if possible, otherwise
    skip `skip` examples (slow). Returns (dataset, exact); exact=True means stream_state() is valid for new checkpoints."""
    ds = make()
    if state:
        try:
            ds.load_state_dict(state)
            return ds, True
        except Exception as e:
            print(f"could not restore stream position ({type(e).__name__}); skipping {skip:,} examples instead", flush=True)
            ds = make()
    if skip:
        return ds.skip(skip), False
    return ds, True
