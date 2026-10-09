from scratchgpt import cli


def test_help_and_unknown_command(capsys):
    assert cli.main(["--help"]) == 0
    assert "train-tokenizer" in capsys.readouterr().out
    assert cli.main(["nope"]) == 2


def test_every_command_points_at_a_module_with_main():
    import importlib.util
    for name, (module, _) in cli.COMMANDS.items():
        assert importlib.util.find_spec(module) is not None, name
