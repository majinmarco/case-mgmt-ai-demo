"""Single source of configuration: model IDs, thresholds and paths.

Every value has a default and can be overridden by a `PRA_`-prefixed
environment variable (or the same key in `.env`). Loading settings also
exports `.env` into the process environment so the OpenAI and LangSmith
SDKs find their API keys; variables already set in the shell win.
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PRA_", extra="ignore")

    decisions_api_model: str = "gpt-6-luna"
    notes_model: str = "gpt-6-luna"
    selector_model: str = "gpt-6-luna"
    embedding_model: str = "text-embedding-3-large"

    approval_cutoff: float = 0.5
    borderline_low: float = 0.4
    borderline_high: float = 0.6

    data_dir: Path = Path("data")

    @property
    def index_dir(self) -> Path:
        """Where the Chroma index of the Reglamento is persisted."""
        return self.data_dir / "chroma"

    @property
    def db_path(self) -> Path:
        """SQLite file for Permit Requests, Recommendations and Determinations."""
        return self.data_dir / "app.sqlite"

    @model_validator(mode="after")
    def _borderline_band_contains_cutoff(self) -> "Settings":
        if not (0 <= self.borderline_low <= self.approval_cutoff <= self.borderline_high <= 1):
            raise ValueError(
                "Borderline band must satisfy 0 <= low <= approval cutoff <= high <= 1, got "
                f"low={self.borderline_low}, cutoff={self.approval_cutoff}, high={self.borderline_high}"
            )
        return self


def load_settings(env_file: str | None = ".env") -> Settings:
    if env_file is not None:
        load_dotenv(env_file, override=False)
    # `_env_file` is a pydantic-settings runtime kwarg that type checkers can't see.
    return Settings(_env_file=env_file)  # pyright: ignore[reportCallIssue]
