"""On-disk format of instruction-tuning data: .npz shards holding concatenated conversations.

    tokens  uint16  all conversations back to back
    mask    uint8   1 where the model should learn (assistant content), same length as tokens
    lengths int32   tokens per conversation
"""
import numpy as np

from ..safeio import atomic_write


def write_shard(path, rows):
    """rows: list of (ids, mask)."""
    lengths = np.array([len(ids) for ids, _ in rows], dtype=np.int32)
    tokens = np.concatenate([np.asarray(ids, dtype=np.uint16) for ids, _ in rows])
    mask = np.concatenate([np.asarray(m, dtype=np.uint8) for _, m in rows])
    atomic_write(path, lambda f: np.savez(f, tokens=tokens, mask=mask, lengths=lengths), keep_prev=False)


class Shard:
    def __init__(self, path):
        z = np.load(path)
        self.tokens, self.mask, self.lengths = z["tokens"], z["mask"], z["lengths"]
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])

    def __len__(self):
        return len(self.lengths)

    def row(self, i):
        a, b = self.offsets[i], self.offsets[i + 1]
        return self.tokens[a:b], self.mask[a:b]


def to_xy(ids, mask, max_len=1025):
    """One conversation -> (inputs, targets) with -100 on positions the model must not learn."""
    ids = np.asarray(ids, dtype=np.int64)[-max_len:]
    mask = np.asarray(mask, dtype=np.int64)[-max_len:]
    x, y = ids[:-1], ids[1:].copy()
    y[mask[1:] == 0] = -100
    return x, y


def collate(pairs):
    """List of (x, y) -> padded int64 arrays (B, L)."""
    L = max(len(x) for x, _ in pairs)
    X = np.zeros((len(pairs), L), dtype=np.int64)
    Y = np.full((len(pairs), L), -100, dtype=np.int64)
    for i, (x, y) in enumerate(pairs):
        X[i, :len(x)], Y[i, :len(y)] = x, y
    return X, Y
