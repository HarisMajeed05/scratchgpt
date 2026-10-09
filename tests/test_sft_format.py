import numpy as np

from scratchgpt.data.sft_format import Shard, collate, to_xy, write_shard


def test_shard_round_trip(tmp_path):
    rows = [([1, 2, 3, 4], [0, 0, 1, 1]), ([5, 6, 7], [0, 1, 1])]
    write_shard(tmp_path / "train_00000.npz", rows)
    s = Shard(tmp_path / "train_00000.npz")
    assert len(s) == 2
    ids, mask = s.row(1)
    assert list(ids) == [5, 6, 7] and list(mask) == [0, 1, 1]


def test_to_xy_masks_targets_that_are_not_learned():
    x, y = to_xy([10, 11, 12, 13], [0, 0, 1, 1])
    assert list(x) == [10, 11, 12]
    assert list(y) == [-100, 12, 13]


def test_collate_pads_with_ignore_index():
    X, Y = collate([(np.array([1, 2, 3]), np.array([2, 3, 4])), (np.array([1]), np.array([2]))])
    assert X.shape == (2, 3) and list(Y[1]) == [2, -100, -100] and list(X[1]) == [1, 0, 0]
