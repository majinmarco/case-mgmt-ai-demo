"""Heuristic checks for badly extracted Section text.

Garbled text can still be retrieved, and it moves the Approval Score (see the CMAWD-2
spike), so retrieval metrics alone do not catch it. These checks only warn: each flag is
something for a person to look at in the PDF.
"""

import re
from collections.abc import Iterator, Sequence
from itertools import groupby, pairwise
from typing import Literal

from pydantic import BaseModel

from case_mgmt_ai_demo.chunking import LIST_ITEM, REPEAT_SUFFIX
from case_mgmt_ai_demo.chunks import Chunk, split_clauses
from case_mgmt_ai_demo.extraction import GAP

Check = Literal[
    "unread_text",
    "truncated_line",
    "running_header",
    "clause_lettering_gap",
    "section_out_of_sequence",
    "short_chunk",
]

SHORT_CHUNK_CHARS = 40

_RUNNING_HEADER = re.compile(
    r"REGLAMENTO CONJUNTO\s*\|\s*[\divxlc]+\b|\b[\divxlc]+\s*\|\s*REGLAMENTO CONJUNTO"
    r"|^\s*REGLAMENTO CONJUNTO\s*$",
    re.MULTILINE,
)
_CLAUSE_LETTERS = "abcdefghijklmnñopqrstuvwxyz"


class Flag(BaseModel):
    section_id: str
    part: str | None = None
    check: Check
    detail: str


def check_chunks(chunks: Sequence[Chunk]) -> list[Flag]:
    flags: list[Flag] = []
    for chunk in chunks:
        flags += [
            Flag(section_id=chunk.section_id, part=chunk.part, check=check, detail=detail)
            for check, detail in _text_problems(chunk.text)
        ]
    flags += _clause_lettering_gaps(chunks)
    flags += _sections_out_of_sequence(chunks)
    return flags


def _text_problems(text: str) -> Iterator[tuple[Check, str]]:
    if GAP in text:
        yield "unread_text", f"{text.count(GAP)} run(s) of text drawn as outlines were not read"
    lines = text.split("\n")
    for line, following in pairwise(lines):
        # A sentence does not stop on a bare word, and a new one does not start in lower case.
        if line[-1:].isalnum() and following[:1].islower() and not LIST_ITEM.match(following):
            yield "truncated_line", f"…{line[-40:]!r} is followed by {following[:40]!r}…"
    if remnant := _RUNNING_HEADER.search(text):
        yield "running_header", repr(remnant[0])
    if len(text.strip()) < SHORT_CHUNK_CHARS:
        yield "short_chunk", f"{len(text.strip())} characters: {text.strip()!r}"


def _clause_lettering_gaps(chunks: Sequence[Chunk]) -> Iterator[Flag]:
    for section_id, parts in groupby(chunks, key=lambda chunk: chunk.section_id):
        letters = [
            clause.ref[-2]
            for part in parts
            for clause in split_clauses(part)
            if clause.ref != section_id
        ]
        for previous, letter in zip(["", *letters], letters):
            if letter not in _letters_after(previous):
                came_after = f"after ({previous})" if previous else "first"
                yield Flag(
                    section_id=section_id,
                    check="clause_lettering_gap",
                    detail=f"({letter}) comes {came_after}",
                )


def _letters_after(letter: str) -> str:
    if not letter:
        return "a"
    following = _CLAUSE_LETTERS[_CLAUSE_LETTERS.index(letter) + 1 :]
    return following[:2] if letter == "n" else following[:1]  # ñ is often left out


def _sections_out_of_sequence(chunks: Sequence[Chunk]) -> Iterator[Flag]:
    """Within a Regla, Secciones are numbered <Regla>.1, <Regla>.2, … in order."""
    secciones = [chunk for chunk in chunks if chunk.level == "seccion" and chunk.part_number == 1]
    for parents, group in groupby(secciones, key=lambda chunk: chunk.parents):
        if not parents or not parents[-1].startswith("Regla "):
            continue
        regla = REPEAT_SUFFIX.sub("", parents[-1].removeprefix("Regla "))
        expected = 1
        for chunk in group:
            number = REPEAT_SUFFIX.sub("", chunk.section_id.split(":", 1)[1])
            if number != f"{regla}.{expected}":
                yield Flag(
                    section_id=chunk.section_id,
                    check="section_out_of_sequence",
                    detail=f"Sección {number} where {regla}.{expected} was expected",
                )
            last = number.rsplit(".", 1)[-1]
            in_regla = number.startswith(f"{regla}.") and last.isdigit()
            expected = int(last) + 1 if in_regla else expected + 1
