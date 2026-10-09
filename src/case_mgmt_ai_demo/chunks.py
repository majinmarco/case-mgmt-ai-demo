"""The chunk contract: one citable Section of the Reglamento as stored for retrieval.

Ingestion produces these and every downstream component builds against them. Until real
ingestion exists, `load_fixture_chunks()` serves ~10 hand-typed RC-2023-E Sections.
"""

import json
from importlib.resources import files
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

RequestType = Literal["PCOC", "PERMISO_UNICO"]
Doc = Literal["RC-2023-E", "COBROS-OA-2022-07"]
Level = Literal["tomo", "capitulo", "regla", "seccion", "anejo_item", "glossary_term"]

SECTION_ID_PREFIX: dict[str, str] = {"RC-2023-E": "RC-2023-E:", "COBROS-OA-2022-07": "COBROS:"}


class Chunk(BaseModel):
    section_id: str = Field(description="Canonical Section ID, e.g. 'RC-2023-E:3.7.1.2'.")
    doc: Doc
    level: Level
    heading: str
    parents: list[str] = Field(description="e.g. ['Tomo III', 'Capítulo 3.7', 'Regla 3.7.1']")
    page_start: int = Field(ge=1, description="Source PDF page (1-based).")
    page_end: int = Field(ge=1)
    text: str = Field(min_length=1, description="Full Section text, incisos preserved.")
    part: str | None = Field(
        default=None, pattern=r"^\d+-of-\d+$", description="'1-of-N' for a split Section."
    )
    request_types: list[RequestType] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.page_end < self.page_start:
            raise ValueError(f"page_end {self.page_end} is before page_start {self.page_start}")
        if not self.section_id.startswith(SECTION_ID_PREFIX[self.doc]):
            raise ValueError(f"section_id {self.section_id!r} does not belong to {self.doc}")
        return self


def load_fixture_chunks() -> list[Chunk]:
    raw = files("case_mgmt_ai_demo.fixtures").joinpath("rc_2023_e_sections.json").read_text()
    return [Chunk.model_validate(item) for item in json.loads(raw)]
