from scratchgpt import sources


def test_shares_sum_to_one():
    assert abs(sum(v["share"] for v in sources.PRETRAIN_SOURCES.values()) - 1) < 1e-9


def test_every_source_is_well_formed():
    for name, v in sources.PRETRAIN_SOURCES.items():
        assert v["kind"] in ("text", "img"), name
        assert "path" in v["kw"], name
        assert v["share"] > 0, name
        if v["kind"] == "text":
            assert v["field"], name
