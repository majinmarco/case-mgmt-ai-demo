import pytest

from case_mgmt_ai_demo import main


@pytest.mark.parametrize("command", ["review", "ingest", "serve"])
def test_placeholder_subcommands_report_not_implemented(command, capsys):
    exit_code = main([command])

    assert exit_code == 1
    assert f"'{command}' is not implemented yet" in capsys.readouterr().err


def test_unknown_subcommand_exits_with_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["frobnicate"])

    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_no_subcommand_exits_with_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])

    assert exc.value.code == 2
