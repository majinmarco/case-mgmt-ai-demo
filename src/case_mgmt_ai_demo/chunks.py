"""The chunk contract: one citable Section of the Reglamento as stored for retrieval.

Ingestion produces these and every downstream component builds against them. Until
retrieval reads the real index, `load_fixture_chunks()` serves ~10 hand-typed RC-2023-E
Sections.
"""

import json
import re
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
    text: str = Field(min_length=1, description="Full Section text, clauses preserved.")
    part: str | None = Field(
        default=None, pattern=r"^\d+-of-\d+$", description="'1-of-N' for a split Section."
    )
    request_types: list[RequestType] = Field(min_length=1)

    @property
    def part_number(self) -> int:
        """Position among the parts of its Section; 1 when the Section is not split."""
        return int(self.part.split("-")[0]) if self.part else 1

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.page_end < self.page_start:
            raise ValueError(f"page_end {self.page_end} is before page_start {self.page_start}")
        if not self.section_id.startswith(SECTION_ID_PREFIX[self.doc]):
            raise ValueError(f"section_id {self.section_id!r} does not belong to {self.doc}")
        return self


class Clause(BaseModel):
    """One lettered clause of a Section, with its text exactly as the chunk holds it."""

    ref: str = Field(description="Section ID plus letter, e.g. 'RC-2023-E:2.1.9.2(k)'.")
    section_id: str
    heading: str
    text: str = Field(description="Verbatim, from the letter through any numbered items.")


# A top-level clause starts a line with its letter and a period ("k. "); numbered items
# ("1. ") and parenthesised sub-items ("a) ") stay inside the clause above them.
_CLAUSE_START = re.compile(r"^(?=[a-zñ]\. )", re.MULTILINE)


def split_clauses(chunk: Chunk) -> list[Clause]:
    """Text before the first clause, or a Section without clauses, is cited by its Section ID."""
    return [
        Clause(
            ref=f"{chunk.section_id}({text[0]})" if _CLAUSE_START.match(text) else chunk.section_id,
            section_id=chunk.section_id,
            heading=chunk.heading,
            text=text.removesuffix("\n"),
        )
        for text in _CLAUSE_START.split(chunk.text)
        if text
    ]


def load_fixture_chunks() -> list[Chunk]:
    raw = files("case_mgmt_ai_demo.fixtures").joinpath("rc_2023_e_sections.json").read_text()
    return [Chunk.model_validate(item) for item in json.loads(raw)]
