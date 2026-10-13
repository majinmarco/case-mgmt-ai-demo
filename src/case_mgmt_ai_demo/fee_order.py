"""The fee order (COBROS OA-2022-07) becomes Chunks: one per ordinal of the order
(PRIMERO … NOVENO) and one per numbered item of Anejo A.

Anejo A and the exemption table of TERCERO are ruled grids whose cells wrap over many lines
and whose rows run across page breaks, so plain text extraction interleaves the columns.
Here a table is read from its rules: each row becomes one line of chunk text, its cells in
column order and named after the table's header row ("a. Aranceles | Derechos a cobrarse:
$5.00 | Concepto: …"). A lettered row starts a line, so it is a Clause of its item.
"""

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pymupdf

from case_mgmt_ai_demo.chunking import (
    Chunking,
    IngestionConfig,
    Paragraph,
    SkippedHeading,
    paragraphs,
    section_chunks,
)
from case_mgmt_ai_demo.chunks import SECTION_ID_PREFIX, Chunk, Level
from case_mgmt_ai_demo.extraction import Line

ORDER = "Orden Administrativa OGPe 2022-07"
ANEJO = "Anejo A"

_ORDINAL = re.compile(
    r"^(PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|SÉPTIMO|OCTAVO|NOVENO):\s*(.*)$"
)
_ANEJO_START = re.compile(r"^ANEJO A\b")
_ORDER_END = re.compile(r"^POR TODO LO CUAL\b")  # the signature block after the last ordinal
_ITEM_NUMBER = re.compile(r"^(\d+)\.$")
_FOOTNOTE_NUMBER = re.compile(r"^(\d{1,2})\s")

_RULE_MAX_THICKNESS = 3.0
_RULE_MIN_LENGTH = 8.0
_SAME_RULE = 3.0  # rules this close are one rule drawn in pieces
_LINE_TOLERANCE = 3.0  # words whose tops are this close share a line
_FOOTNOTE_RULE_MAX_WIDTH = 200.0
_UNDERLINE_GAP = 6.0  # an underline is no further than this below its text
_SUPERSCRIPT = 1  # pymupdf span flag

type _Word = tuple[float, float, float, float, str]


@dataclass
class _Row:
    """One row of a ruled table. Column 0 is the leftmost; a row whose cells are merged
    (an item's heading, a division's name) has fewer cells than the table has columns."""

    page_start: int
    page_end: int
    cells: dict[int, str]
    header: dict[int, str]  # the table's header row, by column
    merged: bool
    ruled_above: bool  # False for the rest of a row the page break cut

    @property
    def first_named_column(self) -> int:
        """Where the fee columns start: after the row's label, and after the marker column
        ("#") when the table has one."""
        return 2 if self.header.get(0) else 1

    def text(self, columns_from: int = 0) -> str:
        """The row as a line of chunk text: its cells from `columns_from` on, each fee
        column's cell under the name the header row gives that column."""
        cells = [
            f"{self.header[column]}: {cell}"
            if column >= self.first_named_column and self.header.get(column)
            else cell
            for column, cell in sorted(self.cells.items())
            if column >= columns_from
        ]
        if self.first_named_column == 2 and 0 in self.cells and columns_from == 0:
            return f"{cells[0]} {' | '.join(cells[1:])}".strip()
        return " | ".join(cells)


@dataclass(frozen=True)
class _Footnote:
    page: int
    text: str


type _Content = Line | _Row | _Footnote


def chunk_fee_order(pdf: Path, config: IngestionConfig) -> Chunking:
    """The chunks of the pages `config` names: an ordinal or an Anejo A item each, split
    into parts when longer than the configured length."""
    chunks: list[Chunk] = []
    skipped: list[SkippedHeading] = []
    prefix = SECTION_ID_PREFIX[config.doc]
    for section in _sections(_read(pdf, config)):
        section_id = f"{prefix}{section.number}"
        if not section.text:
            skipped.append(SkippedHeading(section_id, section.level, section.heading, section.page))
            continue
        chunks += section_chunks(
            section_id,
            section.level,
            section.heading,
            section.parents,
            section.page,
            section.text,
            section.number,
            config,
        )
    return Chunking(chunks=chunks, skipped=skipped, repeated=[])


@dataclass
class _Section:
    level: Level
    number: str  # the Section ID without its document prefix: "TERCERO", "AnejoA:6"
    heading: str
    parents: list[str]
    page: int
    text: list[Paragraph]


def _sections(items: Sequence[_Content]) -> Iterator[_Section]:
    """Everything before PRIMERO (the recitals) and between sections (the signature, the
    table's header row) belongs to no section and is dropped."""
    current: _Section | None = None
    prose: list[Line] = []
    in_anejo = False
    division: list[str] = []
    after_division = False  # the last row read named a division

    def close() -> Iterator[_Section]:
        nonlocal current
        if current:
            current.text += paragraphs(prose)
            yield current
        prose.clear()
        current = None

    for item in items:
        if isinstance(item, Line):
            ordinal = _ORDINAL.match(item.text)
            if ordinal and not in_anejo:
                yield from close()
                title = ordinal[2].rstrip(":") if ordinal[2].isupper() else ""
                current = _Section(
                    "seccion", ordinal[1], title or ordinal[1], [ORDER], item.page, []
                )
                if ordinal[2] and not title:
                    prose.append(replace(item, text=ordinal[2]))
            elif _ANEJO_START.match(item.text) or _ORDER_END.match(item.text):
                yield from close()
                in_anejo = in_anejo or bool(_ANEJO_START.match(item.text))
            elif current:
                prose.append(item)
            continue
        if current:
            current.text += paragraphs(prose)
            prose.clear()
        if isinstance(item, _Footnote):
            if current:
                current.text.append(Paragraph(item.text, 0.0, item.page, item.page))
            continue
        number = _ITEM_NUMBER.match(item.cells.get(0, ""))
        if in_anejo and number:
            yield from close()
            rest = [cell for column, cell in sorted(item.cells.items()) if column > 0]
            current = _Section(
                "anejo_item",
                f"AnejoA:{number[1]}",
                item.cells.get(1) or " ".join(rest),
                [ANEJO, *division],
                item.page_start,
                [],
            )
            if 1 in item.cells and len(rest) > 1:  # an item of one row: its fee is beside it
                current.text.append(_row_paragraph(item, columns_from=1))
        elif in_anejo and item.merged and len(item.cells) == 1:
            yield from close()
            name = " ".join(item.cells.values())
            # A division's name may be followed by a second row naming a part of it.
            division = [*division, name] if after_division else [name]
            after_division = True
            continue
        elif current:
            current.text.append(_row_paragraph(item))
        after_division = False
    yield from close()


def _row_paragraph(row: _Row, columns_from: int = 0) -> Paragraph:
    return Paragraph(row.text(columns_from), 0.0, row.page_start, row.page_end)


def _read(pdf: Path, config: IngestionConfig) -> list[_Content]:
    """The lines and table rows of the configured pages in reading order, without the
    running header and footer. A row cut by a page break is put back together, and a
    footnote follows the line or row that refers to it. A table is taken to start at the
    first row after a line of text, and that row to be its header; a page holds one table."""
    items: list[_Content] = []
    last_row: _Row | None = None  # None between tables: the next row is a header row
    with pymupdf.open(pdf) as document:
        for number in range(config.first_page, min(config.last_page, len(document)) + 1):
            page = document[number - 1]
            first_row = True
            for item in _page_items(page, number, config):
                if isinstance(item, _Row):
                    item.header = last_row.header if last_row else item.cells
                    is_header = last_row is None
                    if last_row and first_row and _continues(item):
                        for column, text in item.cells.items():
                            above = last_row.cells.get(column, "")
                            last_row.cells[column] = f"{above} {text}".strip()
                        last_row.page_end = number
                    else:
                        last_row = item
                        if not is_header:
                            items.append(item)
                    first_row = False
                else:
                    if isinstance(item, Line):
                        last_row = None
                    items.append(item)
    return items


def _continues(row: _Row) -> bool:
    """Whether a page's first row is the rest of the row the page break cut: it has no rule
    above it, or neither a label nor anything in the first fee column."""
    return not row.ruled_above or (
        not row.merged and 0 not in row.cells and row.first_named_column not in row.cells
    )


def _page_items(page: pymupdf.Page, number: int, config: IngestionConfig) -> list[_Content]:
    extracted: Any = page.get_text("words")
    words: list[_Word] = [
        (x0, y0, x1, y1, word)
        for x0, y0, x1, y1, word, *_ in extracted
        if config.header_bottom <= y0 < config.footer_top
    ]
    rects = [drawing["rect"] for drawing in page.get_drawings()]
    vertical = [r for r in rects if r.width < _RULE_MAX_THICKNESS and r.height > _RULE_MIN_LENGTH]
    horizontal = [r for r in rects if r.height < _RULE_MAX_THICKNESS and r.width > _RULE_MIN_LENGTH]
    top = min((r.y0 for r in vertical), default=0.0) - _SAME_RULE
    bottom = max((r.y1 for r in vertical), default=0.0) + _SAME_RULE
    in_table = [r for r in horizontal if vertical and top <= r.y0 <= bottom]
    rules = _merged(r.y0 for r in in_table)
    left = min((r.x0 for r in in_table), default=0.0)
    right = max((r.x1 for r in in_table), default=0.0)
    # The table's own border is no column boundary.
    columns = [
        x for x in _merged(r.x0 for r in vertical) if left + _SAME_RULE < x < right - _SAME_RULE
    ]

    footnotes_from = _footnotes_top(horizontal, words, bottom)
    body = [word for word in words if word[1] < footnotes_from]
    in_rows: dict[float | None, list[_Word]] = {}
    outside: list[_Word] = []
    for word in body:
        middle = (word[1] + word[3]) / 2
        if vertical and top <= middle <= bottom:
            above = [rule for rule in rules if rule <= middle]
            in_rows.setdefault(max(above) if above else None, []).append(word)
        else:
            outside.append(word)

    # (y, item, the y range a footnote reference must fall in)
    placed: list[tuple[float, _Content, tuple[float, float]]] = []
    for line in _lines(outside):
        y0, y1 = min(w[1] for w in line), max(w[3] for w in line)
        text = " ".join(w[4] for w in line)
        placed.append((y0, Line(number, line[0][0], y0, line[-1][2], text), (y0, y1)))
    for rule, members in in_rows.items():
        y0, y1 = min(w[1] for w in members), max(w[3] for w in members)
        middle = (y0 + y1) / 2
        boundaries = [
            x
            for x in columns
            if any(abs(r.x0 - x) <= _SAME_RULE and r.y0 <= middle <= r.y1 for r in vertical)
        ]
        cells: dict[int, list[_Word]] = {}
        for word in members:
            centre = (word[0] + word[2]) / 2
            before = [x for x in boundaries if x <= centre]
            column = columns.index(max(before)) + 1 if before else 0
            cells.setdefault(column, []).append(word)
        row = _Row(
            page_start=number,
            page_end=number,
            cells={
                column: " ".join(w[4] for line in _lines(cell) for w in line)
                for column, cell in sorted(cells.items())
            },
            header={},
            merged=len(boundaries) < len(columns),
            ruled_above=rule is not None,
        )
        placed.append((y0, row, (y0, y1)))
    placed.sort(key=lambda item: item[0])

    footnotes: list[str] = []
    for line in _lines([word for word in words if word[1] >= footnotes_from]):
        text = " ".join(w[4] for w in line)
        if footnotes and not _FOOTNOTE_NUMBER.match(text):
            footnotes[-1] += f" {text}"  # a footnote of several lines
        else:
            footnotes.append(text)
    references = _footnote_references(page)
    items: list[_Content] = [item for _, item, _ in placed]
    for text in footnotes:
        numbered = _FOOTNOTE_NUMBER.match(text)
        referring = [
            item
            for _, item, (y0, y1) in placed
            if numbered
            and any(mark == numbered[1] and y0 - 2 <= y <= y1 + 2 for mark, y in references)
        ]
        # Without a reference on the page, the footnote goes where the page prints it: last.
        after = items.index(referring[0]) + 1 if referring else len(items)
        items.insert(after, _Footnote(number, text))
    return items


def _footnotes_top(
    horizontal: Sequence[pymupdf.Rect], words: Sequence[_Word], table_bottom: float
) -> float:
    """The y of the short rule that sets the page's footnotes off: it stands clear of the
    text above it (an underline does not) and text starts at its left end right under it.
    Past the page's foot when there are no footnotes."""
    for rule in sorted(horizontal, key=lambda r: r.y0):
        below = _lines([word for word in words if word[1] > rule.y0])
        underlines = any(
            rule.y0 - _UNDERLINE_GAP < y1 <= rule.y0 + _SAME_RULE and x0 < rule.x1 and x1 > rule.x0
            for x0, _, x1, y1, _ in words
        )
        if (
            rule.y0 > table_bottom
            and rule.width < _FOOTNOTE_RULE_MAX_WIDTH
            and not underlines
            and below
            and abs(below[0][0][0] - rule.x0) < _SAME_RULE
        ):
            return rule.y0
    return float("inf")


def _footnote_references(page: pymupdf.Page) -> list[tuple[str, float]]:
    """Each superscript number on the page, with the y of its middle."""
    layout: Any = page.get_text("dict")
    return [
        (span["text"].strip(), (span["bbox"][1] + span["bbox"][3]) / 2)
        for block in layout["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
        if span["flags"] & _SUPERSCRIPT and span["text"].strip().isdigit()
    ]


def _lines(words: Sequence[_Word]) -> list[list[_Word]]:
    """Words grouped into lines top to bottom, each left to right."""
    lines: list[list[_Word]] = []
    top = 0.0
    for word in sorted(words, key=lambda w: w[1]):
        if lines and word[1] - top <= _LINE_TOLERANCE:
            lines[-1].append(word)
        else:
            lines.append([word])
            top = word[1]
    return [sorted(line, key=lambda w: w[0]) for line in lines]


def _merged(values: Iterator[float]) -> list[float]:
    merged: list[float] = []
    for value in sorted(values):
        if not merged or value - merged[-1] > _SAME_RULE:
            merged.append(value)
    return merged
