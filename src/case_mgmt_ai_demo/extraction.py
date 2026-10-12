"""Text extraction from a Reglamento PDF: the visual lines of each page, in reading order.

The RC-2023-E PDF was made with "Microsoft: Print To PDF", which drew some runs of text as
vector outlines instead of text (e.g. most of the last sentence of Sección 2.1.9.5(a) on
p. 80). Plain text extraction silently skips them. Here every outlined run is found from the
page's drawings and either read back with OCR (when Spanish tessdata is installed) or left
as a `[…]` marker, so missing text is never silent.
"""

import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

GAP = "[…]"
OCR_LANGUAGE = "spa"

_BOLD = 16  # pymupdf span flag
_LINE_TOLERANCE = 5.0  # fragments whose vertical centres are this close share a visual line
_GLYPH_MAX_SIZE = 14.0
_GLYPH_MIN_PATH_ITEMS = 4  # rules, boxes and square bullets are simpler paths than letters
_RUN_MIN_GLYPHS = 3
_RUN_MAX_LETTER_GAP = 15.0
_RULE_MIN_WIDTH = 100.0  # a table's ruled lines are at least this long
_GUTTER = 6.0  # the least clear space between two columns of a table
_TOUCHING = 1.5  # an outlined run this close to text continues the same word


@dataclass(frozen=True)
class Line:
    """One visual line of a page, left to right."""

    page: int  # physical, 1-based
    x0: float
    y0: float
    x1: float
    text: str
    bold: bool = False
    cell: int | None = None  # the table cell it is in, numbered in reading order on its page


@dataclass(frozen=True)
class OutlinedRun:
    """Text the PDF draws as outlines, with what extraction put in its place."""

    page: int
    x0: float
    y0: float
    text: str  # the OCR reading, or GAP when it could not be read

    @property
    def recovered(self) -> bool:
        return self.text != GAP


@dataclass(frozen=True)
class Extraction:
    lines: list[Line]
    outlined_runs: list[OutlinedRun]
    ocr: bool


@dataclass
class _Fragment:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    bold: bool = False
    outlined: bool = False


def find_tessdata(tessdata_dir: Path | None = None) -> Path | None:
    """The tessdata directory holding the Spanish model, or None when OCR is unavailable."""
    candidates = [tessdata_dir, os.environ.get("TESSDATA_PREFIX"), pymupdf.get_tessdata()]
    for candidate in candidates:
        if candidate and (Path(candidate) / f"{OCR_LANGUAGE}.traineddata").is_file():
            return Path(candidate)
    return None


def extract_lines(
    pdf: Path,
    first_page: int,
    last_page: int,
    footer_top: float,
    tessdata: Path | None = None,
) -> Extraction:
    """Lines of pages `first_page`..`last_page` (physical, 1-based, inclusive), without the
    running header, which this PDF prints in a band at the foot of the page (`footer_top`)."""
    lines: list[Line] = []
    runs: list[OutlinedRun] = []
    with pymupdf.open(pdf) as document:
        for number in range(first_page, min(last_page, len(document)) + 1):
            page = document[number - 1]
            fragments = [f for f in _text_fragments(page) if f.y0 < footer_top]
            if not fragments:
                # A Tomo's cover page: its title is all outlines, and the heading on the
                # next page says the same.
                continue
            drawings = [d for d in page.get_drawings() if d["rect"].y0 < footer_top]
            outlined = list(_outlined_fragments(drawings))
            words = _ocr_words(page, tessdata) if outlined and tessdata else None
            for cell, row in _rows_in_reading_order(fragments + outlined, _tables(drawings)):
                for index, fragment in enumerate(row):
                    if fragment.outlined:
                        before = row[index - 1] if index else None
                        after = row[index + 1] if index + 1 < len(row) else None
                        if words is not None:
                            fragment.text = _read_run(fragment, words, before, after) or GAP
                        runs.append(OutlinedRun(number, fragment.x0, fragment.y0, fragment.text))
                text = _join(row)
                if text:
                    printed = [f for f in row if not f.outlined]
                    lines.append(
                        Line(
                            page=number,
                            x0=row[0].x0,
                            y0=min(f.y0 for f in printed or row),
                            x1=row[-1].x1,
                            text=text,
                            bold=bool(printed) and all(f.bold for f in printed),
                            cell=cell,
                        )
                    )
    return Extraction(lines=lines, outlined_runs=runs, ocr=tessdata is not None)


def _text_fragments(page: pymupdf.Page) -> Iterator[_Fragment]:
    layout: Any = page.get_text("dict")
    for block in layout["blocks"]:
        for line in block.get("lines", []):
            spans = [span for span in line["spans"] if span["text"].strip()]
            if spans:
                yield _Fragment(
                    *line["bbox"],
                    text="".join(span["text"] for span in line["spans"]),
                    bold=all(span["flags"] & _BOLD for span in spans),
                )


@dataclass(frozen=True)
class _Table:
    """A table as its ruled lines show it: `rules` are the y of each horizontal rule."""

    x0: float
    x1: float
    rules: list[float]


def _tables(drawings: Sequence[dict[str, Any]]) -> list[_Table]:
    """Runs of three or more horizontal rules ending at the same right edge. The tables of
    the Reglamento rule off their rows but not always their columns."""
    rules = sorted(
        (d["rect"] for d in drawings if d["rect"].width > _RULE_MIN_WIDTH and d["rect"].height < 3),
        key=lambda rect: rect.y0,
    )
    tables: list[list[pymupdf.Rect]] = []
    for rule in rules:
        if tables and abs(tables[-1][-1].x1 - rule.x1) < 3:
            tables[-1].append(rule)
        else:
            tables.append([rule])
    return [
        _Table(min(r.x0 for r in table), table[0].x1, [r.y0 for r in table])
        for table in tables
        if len(table) >= 3
    ]


def _rows_in_reading_order(
    fragments: Sequence[_Fragment], tables: Sequence[_Table]
) -> list[tuple[int | None, list[_Fragment]]]:
    """Visual lines top to bottom, except that a table is read cell by cell: the lines of
    cells side by side share baselines, and read straight across they would interleave."""
    by_place: dict[tuple[float, float] | None, list[_Fragment]] = {}
    for table in tables:
        between_rules: dict[float, list[_Fragment]] = {}
        for fragment in fragments:
            middle = (fragment.y0 + fragment.y1) / 2
            if table.rules[0] <= middle < table.rules[-1]:
                row = max(y for y in table.rules if y <= middle)
                between_rules.setdefault(row, []).append(fragment)
        # Text sticking out of the table between two rules (a heading, a paragraph) means
        # those rules bound no row: one table has ended and another has yet to start.
        rows = {
            row: members
            for row, members in between_rules.items()
            if all(table.x0 - 3 <= f.x0 and f.x1 <= table.x1 + 3 for f in members)
        }
        columns = _column_starts([f for members in rows.values() for f in members])
        for row, members in rows.items():
            for fragment in members:
                column = max(x for x in columns if x <= fragment.x0)
                by_place.setdefault((row, column), []).append(fragment)
    in_tables = {id(f) for members in by_place.values() for f in members}
    by_place[None] = [f for f in fragments if id(f) not in in_tables]
    ordered: list[tuple[tuple[float, float, float], tuple[float, float] | None, list[_Fragment]]]
    ordered = []
    for place, members in by_place.items():
        for row in _rows(members):
            y = min(fragment.y0 for fragment in row)
            ordered.append(((y, 0.0, y) if place is None else (*place, y), place, row))
    ordered.sort(key=lambda item: item[0])
    cells = {place: number for number, place in enumerate(sorted(p for p in by_place if p))}
    return [(cells.get(place) if place else None, row) for _, place, row in ordered]


def _column_starts(fragments: Sequence[_Fragment]) -> list[float]:
    """Where each column of a table starts: the left edges after a gutter no text crosses."""
    starts: list[float] = []
    covered = float("-inf")
    for fragment in sorted(fragments, key=lambda f: f.x0):
        if fragment.x0 - covered > _GUTTER:
            starts.append(fragment.x0)
        covered = max(covered, fragment.x1)
    return starts


def _outlined_fragments(drawings: Sequence[dict[str, Any]]) -> Iterator[_Fragment]:
    """Runs of letter-sized filled paths: text drawn as outlines."""
    glyphs = sorted(
        (
            drawing["rect"]
            for drawing in drawings
            if drawing["type"] == "f"
            and len(drawing["items"]) >= _GLYPH_MIN_PATH_ITEMS
            and drawing["rect"].width <= _GLYPH_MAX_SIZE
            and drawing["rect"].height <= _GLYPH_MAX_SIZE
        ),
        key=lambda rect: rect.y0,
    )
    band: list[pymupdf.Rect] = []
    bottom = 0.0
    for glyph in [*glyphs, None]:
        if band and (glyph is None or glyph.y0 > bottom):
            # Taller than a line of text: a figure, not letters.
            if bottom - band[0].y0 <= _GLYPH_MAX_SIZE + 2:
                yield from _runs_in_band(band)
            band = []
        if glyph is not None:
            bottom = max(bottom, glyph.y1) if band else glyph.y1
            band.append(glyph)


def _runs_in_band(band: list[pymupdf.Rect]) -> Iterator[_Fragment]:
    run: list[pymupdf.Rect] = []
    for glyph in [*sorted(band, key=lambda rect: rect.x0), None]:
        if run and (glyph is None or glyph.x0 - max(g.x1 for g in run) > _RUN_MAX_LETTER_GAP):
            if len(run) >= _RUN_MIN_GLYPHS:
                yield _Fragment(
                    min(g.x0 for g in run),
                    min(g.y0 for g in run),
                    max(g.x1 for g in run),
                    max(g.y1 for g in run),
                    text=GAP,
                    outlined=True,
                )
            run = []
        if glyph is not None:
            run.append(glyph)


def _rows(fragments: Sequence[_Fragment]) -> list[list[_Fragment]]:
    """Fragments grouped into visual lines, top to bottom, each left to right."""
    rows: list[list[_Fragment]] = []
    centre = 0.0
    for fragment in sorted(fragments, key=lambda f: (f.y0 + f.y1) / 2):
        middle = (fragment.y0 + fragment.y1) / 2
        if rows and middle - centre <= _LINE_TOLERANCE:
            rows[-1].append(fragment)
        else:
            rows.append([fragment])
            centre = middle
    return [sorted(row, key=lambda f: f.x0) for row in rows]


def _join(row: Sequence[_Fragment]) -> str:
    text = ""
    for index, fragment in enumerate(row):
        touching = index > 0 and fragment.x0 - row[index - 1].x1 < _TOUCHING
        mid_word = touching and (fragment.outlined or row[index - 1].outlined)
        # An unread run is never glued to its neighbour: "Certi […]", not "Certi[…]".
        if text and not (mid_word and GAP not in (fragment.text, row[index - 1].text)):
            text += " "
        text += fragment.text
    return re.sub(r"\s+", " ", text).strip()


type _Word = tuple[float, float, float, float, str]


def _ocr_words(page: pymupdf.Page, tessdata: Path) -> list[_Word]:
    textpage = page.get_textpage_ocr(
        language=OCR_LANGUAGE, dpi=300, full=True, tessdata=str(tessdata)
    )
    words: Any = page.get_text("words", textpage=textpage)
    return [(x0, y0, x1, y1, word) for x0, y0, x1, y1, word, *_ in words]


def _read_run(
    run: _Fragment, words: Sequence[_Word], before: _Fragment | None, after: _Fragment | None
) -> str:
    """The OCR words lying on the run. A word that also covers the text beside the run (the
    PDF can switch to outlines mid-word) gives only the letters that text does not have."""
    middle = (run.y0 + run.y1) / 2
    read: list[str] = []
    for x0, y0, x1, y1, word in sorted(words):
        overlap = min(x1, run.x1) - max(x0, run.x0)
        if not (y0 <= middle <= y1) or overlap <= 0:
            continue
        if x0 < run.x0 - _TOUCHING and before is not None and not before.outlined:
            shared = before.text.split()[-1] if before.text.split() else ""
            if not before.text[-1:].isspace() and word.startswith(shared):
                word = word[len(shared) :]
            elif overlap < (x1 - x0) / 2:
                continue
        if x1 > run.x1 + _TOUCHING and after is not None and not after.outlined:
            shared = after.text.split()[0] if after.text.split() else ""
            if not after.text[:1].isspace() and word.endswith(shared):
                word = word[: len(word) - len(shared)]
            elif overlap < (x1 - x0) / 2:
                continue
        if word:
            read.append(word)
    return " ".join(read)
