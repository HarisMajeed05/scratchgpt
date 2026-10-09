from scratchgpt import tokens as t


def test_ranges_do_not_overlap():
    assert t.TEXT_VOCAB <= t.IMG_BASE
    assert t.IMG_BASE + t.IMG_CODES <= t.AUD_BASE
    assert t.AUD_BASE + t.AUD_CODES <= t.SPECIAL_BASE


def test_specials_unique_and_inside_vocab():
    ids = list(t.SPECIAL_NAMES)
    assert len(set(ids)) == len(ids)
    assert all(t.SPECIAL_BASE <= i < t.VOCAB_SIZE for i in ids)


def test_vocab_fits_uint16_and_is_padded():
    assert t.VOCAB_SIZE <= 65536 and t.VOCAB_SIZE % 64 == 0


def test_ids_that_shards_on_disk_depend_on_never_change():
    assert (t.EOT, t.BOI, t.EOI) == (45056, 45057, 45058)
    assert t.IMG_BASE == 32768 and t.IMG_TOKENS == 256
