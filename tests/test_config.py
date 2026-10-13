import os
from pathlib import Path

import pytest
from langsmith import utils as langsmith_utils

from case_mgmt_ai_demo.config import load_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("PRA_"):
            monkeypatch.delenv(key)
    # setenv first so monkeypatch also removes keys load_settings exports from .env.
    exported = ("OPENAI_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGSMITH_TRACING_V2")
    for key in exported:
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)


def test_defaults_use_design_models_and_thresholds():
    settings = load_settings(env_file=None)

    assert settings.decisions_api_model == "gpt-6-luna"
    assert settings.notes_model == "gpt-6-luna"
    assert settings.selector_model == "gpt-6-luna"
    assert settings.section_validator_model == "gpt-6-luna"
    assert settings.embedding_model == "text-embedding-3-large"
    assert settings.approval_cutoff == 0.5
    assert (settings.borderline_low, settings.borderline_high) == (0.4, 0.6)
    assert settings.data_dir == Path("data")
    assert settings.index_dir == Path("data/chroma")
    assert settings.db_path == Path("data/app.sqlite")
    assert settings.reglamento_pdf == Path("docs_to_use/RC-2023-E.pdf")
    assert settings.retriever == "index"
    assert settings.retrieval_top_k == 8
    assert settings.tessdata_dir is None


def test_paths_follow_overridden_data_dir(monkeypatch):
    monkeypatch.setenv("PRA_DATA_DIR", "/srv/pra")

    settings = load_settings(env_file=None)

    assert settings.index_dir == Path("/srv/pra/chroma")
    assert settings.db_path == Path("/srv/pra/app.sqlite")


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("PRA_DECISIONS_API_MODEL", "gpt-other")
    monkeypatch.setenv("PRA_SELECTOR_MODEL", "gpt-selector")
    monkeypatch.setenv("PRA_SECTION_VALIDATOR_MODEL", "gpt-validator")
    monkeypatch.setenv("PRA_APPROVAL_CUTOFF", "0.7")
    monkeypatch.setenv("PRA_BORDERLINE_LOW", "0.65")
    monkeypatch.setenv("PRA_BORDERLINE_HIGH", "0.75")
    monkeypatch.setenv("PRA_DATA_DIR", "/tmp/pra")

    settings = load_settings(env_file=None)

    assert settings.decisions_api_model == "gpt-other"
    assert settings.selector_model == "gpt-selector"
    assert settings.section_validator_model == "gpt-validator"
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


def langsmith_tracing() -> bool:
    """Whether LangSmith itself would trace a run now (it caches what it reads)."""
    langsmith_utils.get_env_var.cache_clear()  # pyright: ignore[reportFunctionMemberAccess]
    return langsmith_utils.tracing_is_enabled() is True


def test_langsmith_tracing_is_on_when_its_api_key_is_set(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LANGSMITH_API_KEY=ls-from-file\n")

    load_settings(env_file=str(env_file))

    assert langsmith_tracing()


def test_langsmith_tracing_is_off_without_its_api_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    load_settings(env_file=None)

    assert not langsmith_tracing()


def test_langsmith_tracing_can_be_switched_off_with_its_api_key_set(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-from-shell")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")

    load_settings(env_file=None)

    assert not langsmith_tracing()


def test_the_retriever_and_its_top_k_are_configurable(monkeypatch):
    monkeypatch.setenv("PRA_RETRIEVER", "fixture")
    monkeypatch.setenv("PRA_RETRIEVAL_TOP_K", "12")

    settings = load_settings(env_file=None)

    assert (settings.retriever, settings.retrieval_top_k) == ("fixture", 12)
