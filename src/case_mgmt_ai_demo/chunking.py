"""Chunking: the extracted lines of the Reglamento become Chunks, one per Section.

A Sección is a chunk; so is a Regla, Capítulo or Tomo for the text it holds directly (a
Regla without Secciones, or the text before its first Sección). The Reglamento's own
numbering mistakes are kept as printed: a number used twice gets a `-bis` suffix so Section
IDs stay unique, and the rest is left for the extraction checks to flag.
"""

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.resources import files

import yaml
from pydantic import BaseModel, Field, field_validator

from case_mgmt_ai_demo.chunks import SECTION_ID_PREFIX, Chunk, Doc, Level, RequestType
from case_mgmt_ai_demo.extraction import Line


class IngestionConfig(BaseModel):
    """How one Reglamento PDF is extracted and chunked; recorded with every ingestion."""

    doc: Doc
    first_page: int = Field(ge=1, description="First page of the body (physical, 1-based).")
    last_page: int = Field(ge=1, description="Last page of the body, inclusive.")
    footer_top: float = Field(description="Lines at or below this y are the running header.")
    max_chunk_chars: int = Field(gt=0, description="A longer Section is split into parts.")
    request_types: dict[str, list[RequestType]] = Field(
        description="Request Types by Section number prefix ('3.7' covers 3.7.1.2); the "
        "longest prefix wins and 'default' covers the rest."
    )

    @field_validator("request_types")
    @classmethod
    def _has_default(cls, rules: dict[str, list[RequestType]]) -> dict[str, list[RequestType]]:
        if "default" not in rules:
            raise ValueError("request_types needs a 'default' entry")
        return rules


def load_ingestion_config(name: str = "rc_2023_e") -> IngestionConfig:
    raw = files("case_mgmt_ai_demo.fixtures").joinpath(f"ingestion/{name}.yaml").read_text()
    return IngestionConfig.model_validate(yaml.safe_load(raw))


@dataclass(frozen=True)
class SkippedHeading:
    """A heading with no text of its own, so no chunk (most Tomos, Capítulos and Reglas)."""

    section_id: str
    level: Level
    heading: str
    page: int


@dataclass(frozen=True)
class Chunking:
    chunks: list[Chunk]
    skipped: list[SkippedHeading]
    repeated: list[str]  # Section IDs given a suffix because the Reglamento repeats a number


_HEADING = re.compile(
    r"^(TOMO|CAP[IÍ]TULO|REGLA|SECC+I[OÓ]N)\s+(\d+(?:\.\s?\d+)*|[IVXLC]+)\b\.?\s*(.*)$"
)
_HEADING_MAX_X = 100.0  # headings start at the left margin; body text is indented past this
_LEVELS: dict[str, Level] = {"T": "tomo", "C": "capitulo", "R": "regla", "S": "seccion"}
_LABELS: dict[Level, str] = {"tomo": "Tomo", "capitulo": "Capítulo", "regla": "Regla"}
REPEAT_SUFFIX = re.compile(r"-bis\d*$")

LIST_ITEM = re.compile(r"^\s*(?:[a-zñ]{1,2}|\d{1,3}|[ivxlc]+)[.)]\s")
_CLAUSE_MARKER = re.compile(r"^[a-zñ]\. ")
_NESTED_INDENT = "  "  # keeps a lettered sub-item from reading as a top-level Clause
_LINE_PITCH = 13.4
_INDENT_STEP = 9.0
_SHORT_LINE_X1 = 480.0  # a line ending left of this did not wrap


@dataclass
class _Paragraph:
    text: str
    x0: float
    page_start: int
    page_end: int
    in_table: bool = False


@dataclass
class _Section:
    level: Level
    number: str  # as printed
    suffix: str  # "-bis" when the number was already used
    heading: str
    parents: list[str]
    page: int
    lines: list[Line] = field(default_factory=list)


def chunk_lines(lines: Iterable[Line], config: IngestionConfig) -> Chunking:
    chunks: list[Chunk] = []
    skipped: list[SkippedHeading] = []
    repeated: list[str] = []
    for section in _sections(lines):
        number = section.number if section.level != "tomo" else f"Tomo-{section.number}"
        section_id = f"{SECTION_ID_PREFIX[config.doc]}{number}{section.suffix}"
        if section.suffix:
            repeated.append(section_id)
        paragraphs = _paragraphs(section.lines)
        if not paragraphs:
            skipped.append(SkippedHeading(section_id, section.level, section.heading, section.page))
            continue
        parts = _split(paragraphs, config.max_chunk_chars)
        for index, part in enumerate(parts, start=1):
            chunks.append(
                Chunk(
                    section_id=section_id,
                    doc=config.doc,
                    level=section.level,
                    heading=section.heading,
                    parents=section.parents,
                    page_start=section.page if index == 1 else part[0].page_start,
                    page_end=part[-1].page_end,
                    text="\n".join(paragraph.text for paragraph in part),
                    part=f"{index}-of-{len(parts)}" if len(parts) > 1 else None,
                    request_types=request_types_for(section.number, config.request_types),
                )
            )
    return Chunking(chunks=chunks, skipped=skipped, repeated=repeated)


def request_types_for(number: str, rules: Mapping[str, list[RequestType]]) -> list[RequestType]:
    prefixes = [p for p in rules if number == p or number.startswith(f"{p}.")]
    return rules[max(prefixes, key=len)] if prefixes else rules["default"]


def _sections(lines: Iterable[Line]) -> Iterator[_Section]:
    labels: dict[Level, str] = {}
    seen: Counter[tuple[Level, str]] = Counter()
    current: _Section | None = None
    in_heading = False
    previous: Line | None = None
    for line in lines:
        text = _clean(line.text)
        match = _HEADING.match(text) if line.x0 < _HEADING_MAX_X else None
        if match:
            if current:
                yield current
            level = _LEVELS[match[1][0]]
            number = match[2].replace(" ", "")
            for lower in list(_LABELS)[list(_LEVELS.values()).index(level) :]:
                labels.pop(lower, None)
            seen[level, number] += 1
            suffix = _repeat_suffix(seen[level, number])
            current = _Section(level, number, suffix, match[3], list(labels.values()), line.page)
            if level in _LABELS:
                labels[level] = f"{_LABELS[level]} {number}{suffix}"
            in_heading = True
        elif current:
            # A long title wraps onto bold lines directly under the heading.
            wraps = (
                in_heading
                and line.bold
                and line.cell is None
                and previous is not None
                and line.page == previous.page
                and line.y0 - previous.y0 < 1.4 * _LINE_PITCH
            )
            if wraps:
                current.heading = f"{current.heading} {text}".strip()
            else:
                in_heading = False
                current.lines.append(line)
        previous = line
    if current:
        yield current


def _repeat_suffix(use: int) -> str:
    """Nothing for the first use of a number, then -bis, -bis2, -bis3…"""
    return "" if use == 1 else "-bis" if use == 2 else f"-bis{use - 1}"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def _paragraphs(lines: Sequence[Line]) -> list[_Paragraph]:
    """Wrapped lines joined back into paragraphs; every list item is its own paragraph."""
    paragraphs: list[_Paragraph] = []
    previous: Line | None = None
    for line in lines:
        text = _clean(line.text)
        if not text:
            continue
        starts = (
            previous is None
            or bool(LIST_ITEM.match(text))
            or line.cell != previous.cell
            or (line.cell is not None and line.page != previous.page)
            or (line.page == previous.page and line.y0 - previous.y0 > 1.5 * _LINE_PITCH)
            or (previous.x1 < _SHORT_LINE_X1 and previous.text.rstrip()[-1:] in ".:;")
        )
        if starts:
            in_table = line.cell is not None
            paragraphs.append(_Paragraph(text, line.x0, line.page, line.page, in_table))
        else:
            paragraphs[-1].text += f" {text}"
            paragraphs[-1].page_end = line.page
        previous = line
    _indent_nested_letters(paragraphs)
    return paragraphs


def _indent_nested_letters(paragraphs: Sequence[_Paragraph]) -> None:
    """A lettered item under another list item ("e. … / a. … / b. …") or in a table cell is
    not a Clause of the Section, so it is indented: only a letter at the start of a line
    starts a Clause."""
    open_items: list[float] = []
    for paragraph in paragraphs:
        if not LIST_ITEM.match(paragraph.text):
            continue
        while open_items and open_items[-1] > paragraph.x0 - _INDENT_STEP:
            open_items.pop()
        nested = bool(open_items) or paragraph.in_table
        if nested and _CLAUSE_MARKER.match(paragraph.text):
            paragraph.text = _NESTED_INDENT + paragraph.text
        open_items.append(paragraph.x0)


def _split(paragraphs: list[_Paragraph], max_chars: int) -> list[list[_Paragraph]]:
    """Parts of at most `max_chars`, of similar length, cut where a Clause starts; a Clause
    too long on its own is cut between its paragraphs, and a paragraph between sentences."""
    total = _length(paragraphs)
    if total <= max_chars:
        return [paragraphs]
    units: list[list[_Paragraph]] = []
    for clause in _clauses(paragraphs):
        if _length(clause) <= max_chars:
            units.append(clause)
        else:
            units += [[piece] for p in clause for piece in _sentences(p, max_chars)]
    for count in range(math.ceil(total / max_chars), len(units) + 1):
        parts = _even_parts(units, count)
        if all(_length(part) <= max_chars for part in parts):
            return parts
    raise AssertionError("unreachable: every unit fits in a part of its own")


def _even_parts(units: Sequence[list[_Paragraph]], count: int) -> list[list[_Paragraph]]:
    """`count` runs of consecutive units, each cut as near as the units allow to an even share."""
    total = sum(_length(unit) + 1 for unit in units)
    parts: list[list[_Paragraph]] = [[]]
    done = 0
    for index, unit in enumerate(units):
        size = _length(unit) + 1
        share = total * len(parts) / count
        remaining = len(units) - index
        nearer_before = abs(done - share) <= abs(done + size - share)
        if parts[-1] and len(parts) < count and (nearer_before or remaining == count - len(parts)):
            parts.append([])
        parts[-1] += unit
        done += size
    return parts


def _length(paragraphs: Sequence[_Paragraph]) -> int:
    return sum(len(p.text) for p in paragraphs) + max(len(paragraphs) - 1, 0)


def _clauses(paragraphs: Sequence[_Paragraph]) -> list[list[_Paragraph]]:
    clauses: list[list[_Paragraph]] = []
    for paragraph in paragraphs:
        if not clauses or _CLAUSE_MARKER.match(paragraph.text):
            clauses.append([])
        clauses[-1].append(paragraph)
    return clauses


def _sentences(paragraph: _Paragraph, max_chars: int) -> list[_Paragraph]:
    if len(paragraph.text) <= max_chars:
        return [paragraph]
    pieces = [""]
    for sentence in re.split(r"(?<=[.;:]) ", paragraph.text):
        # A sentence longer than the limit (a table read as one run of text) is cut on words.
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces += [sentence[:cut], ""]
            sentence = sentence[cut:].lstrip()
        if pieces[-1] and len(pieces[-1]) + 1 + len(sentence) > max_chars:
            pieces.append("")
        pieces[-1] = f"{pieces[-1]} {sentence}".strip()
    return [
        _Paragraph(piece, paragraph.x0, paragraph.page_start, paragraph.page_end)
        for piece in pieces
        if piece
    ]
