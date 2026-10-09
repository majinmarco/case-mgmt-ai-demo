import os
from pathlib import Path

import pytest

from case_mgmt_ai_demo.config import load_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("PRA_"):
            monkeypatch.delenv(key)
    # setenv first so monkeypatch also removes keys load_settings exports from .env.
    for key in ("OPENAI_API_KEY", "LANGSMITH_API_KEY"):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)


def test_defaults_use_design_models_and_thresholds():
    settings = load_settings(env_file=None)

    assert settings.decisions_api_model == "gpt-6-luna"
    assert settings.notes_model == "gpt-6-luna"
    assert settings.embedding_model == "text-embedding-3-large"
    assert settings.approval_cutoff == 0.5
    assert (settings.borderline_low, settings.borderline_high) == (0.4, 0.6)
    assert settings.data_dir == Path("data")
    assert settings.index_dir == Path("data/chroma")
    assert settings.db_path == Path("data/app.sqlite")


def test_paths_follow_overridden_data_dir(monkeypatch):
    monkeypatch.setenv("PRA_DATA_DIR", "/srv/pra")

    settings = load_settings(env_file=None)

    assert settings.index_dir == Path("/srv/pra/chroma")
    assert settings.db_path == Path("/srv/pra/app.sqlite")


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("PRA_DECISIONS_API_MODEL", "gpt-other")
    monkeypatch.setenv("PRA_APPROVAL_CUTOFF", "0.7")
    monkeypatch.setenv("PRA_BORDERLINE_LOW", "0.65")
    monkeypatch.setenv("PRA_BORDERLINE_HIGH", "0.75")
    monkeypatch.setenv("PRA_DATA_DIR", "/tmp/pra")

    settings = load_settings(env_file=None)

    assert settings.decisions_api_model == "gpt-other"
    assert settings.approval_cutoff == 0.7
    assert (settings.borderline_low, settings.borderline_high) == (0.65, 0.75)
    assert settings.data_dir == Path("/tmp/pra")


def test_env_file_overrides_defaults(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("PRA_NOTES_MODEL=gpt-notes\nOPENAI_API_KEY=sk-test\n")

    settings = load_settings(env_file=str(env_file))

    assert settings.notes_model == "gpt-notes"


def test_env_file_exposes_api_keys_to_sdks(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-from-file\nLANGSMITH_API_KEY=ls-from-file\n")

    load_settings(env_file=str(env_file))

    assert os.environ["OPENAI_API_KEY"] == "sk-from-file"
    assert os.environ["LANGSMITH_API_KEY"] == "ls-from-file"


def test_process_environment_wins_over_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-from-file\n")

    load_settings(env_file=str(env_file))

    assert os.environ["OPENAI_API_KEY"] == "sk-from-shell"


def test_rejects_borderline_band_that_excludes_cutoff(monkeypatch):
    monkeypatch.setenv("PRA_APPROVAL_CUTOFF", "0.8")

    with pytest.raises(ValueError, match="Borderline"):
        load_settings(env_file=None)
