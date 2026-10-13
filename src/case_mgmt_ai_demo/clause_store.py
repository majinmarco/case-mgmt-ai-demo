"""The Clause store: each Section's Clauses, extracted once and kept in SQLite by Section ID.

Extraction does not depend on the Permit Request, so a review reads a Section's Clauses from
the store instead of extracting them again. What does depend on the request (whether the
Section applies, how each Clause is classified) is never kept here.

The extractor is an injected adapter: the regex over lettered paragraphs today, and the seam
for a model-based one where a Reglamento's layout defeats the regex.
"""

import hashlib
import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter, ValidationError

from case_mgmt_ai_demo.chunks import Chunk, Clause, Doc, split_clauses
from case_mgmt_ai_demo.config import Settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS section_clauses (
    section_id TEXT PRIMARY KEY,
    doc TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    clauses TEXT NOT NULL
);
"""

_CLAUSES = TypeAdapter(list[Clause])


class ClauseExtractor(Protocol):
    """Turns a Section into its Clauses (fakeable in tests)."""

    # Names the extractor and its behaviour: Clauses stored under another version are
    # extracted again.
    version: str

    def extract(self, section: Sequence[Chunk]) -> list[Clause]:
        """The Clauses of one Section, given all its parts in order, in Reglamento order.
        A model-based extractor raises ModelError when its call fails."""
        ...


class RegexClauseExtractor:
    """The top-level lettered paragraphs (`a. `, `b. ` …) of each part, as `split_clauses`
    reads them."""

    version = "regex-1"

    def extract(self, section: Sequence[Chunk]) -> list[Clause]:
        return [clause for chunk in section for clause in split_clauses(chunk)]


class ClauseStore:
    """The SQLite file at `path`, created with its table when it does not exist."""

    def __init__(self, path: Path, extractor: ClauseExtractor):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._extractor = extractor
        with self._connection() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """One connection per operation, so the store can be used from any thread."""
        connection = sqlite3.connect(self._path)
        try:
            with connection:  # commits, or rolls back on an error
                yield connection
        finally:
            connection.close()

    def clauses(self, section: Sequence[Chunk]) -> list[Clause]:
        """The Clauses of one Section, given all its parts in order. Read from the store;
        extracted and stored first when the store has none for this Section text and this
        extractor. An extraction that fails raises and stores nothing."""
        section_ids = {chunk.section_id for chunk in section}
        if len(section_ids) != 1:
            raise ValueError(f"Expected the parts of one Section, got {sorted(section_ids)}")
        [section_id] = section_ids
        fingerprint = _fingerprint(section, self._extractor.version)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT clauses FROM section_clauses WHERE section_id = ? AND fingerprint = ?",
                (section_id, fingerprint),
            ).fetchone()
        if row:
            try:
                return _CLAUSES.validate_json(row[0])
            except ValidationError:
                pass  # stored by a version whose Clauses this one cannot read: extract again
        clauses = self._extractor.extract(section)
        with self._connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO section_clauses (section_id, doc, fingerprint, clauses) "
                "VALUES (?, ?, ?, ?)",
                (section_id, section[0].doc, fingerprint, _CLAUSES.dump_json(clauses).decode()),
            )
        return clauses

    def prune(self, doc: Doc, chunks: Sequence[Chunk]) -> int:
        """Forget the stored Clauses of the document's Sections that `chunks` (every chunk of
        the document, each Section's parts in order) no longer hold as stored: Sections that
        are gone or whose text changed. The rest are kept, so they are not extracted again.
        Returns how many Sections were forgotten."""
        by_section: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            by_section.setdefault(chunk.section_id, []).append(chunk)
        current = {
            section_id: _fingerprint(parts, self._extractor.version)
            for section_id, parts in by_section.items()
        }
        with self._connection() as connection:
            stored = connection.execute(
                "SELECT section_id, fingerprint FROM section_clauses WHERE doc = ?", (doc,)
            ).fetchall()
            stale = [(id_,) for id_, fingerprint in stored if current.get(id_) != fingerprint]
            connection.executemany("DELETE FROM section_clauses WHERE section_id = ?", stale)
        return len(stale)

    def clear(self, doc: Doc | None = None) -> int:
        """Forget the stored Clauses of one document, or of every one; each Section's are
        extracted again when next needed. Returns how many Sections were forgotten."""
        with self._connection() as connection:
            if doc is None:
                return connection.execute("DELETE FROM section_clauses").rowcount
            return connection.execute(
                "DELETE FROM section_clauses WHERE doc = ?", (doc,)
            ).rowcount


def _fingerprint(section: Sequence[Chunk], extractor_version: str) -> str:
    """Changes with anything a Clause is made of (the Section's text and heading, part by
    part) and with the extractor."""
    parts = [[chunk.part, chunk.heading, chunk.text] for chunk in section]
    return hashlib.sha256(json.dumps([extractor_version, parts]).encode()).hexdigest()


def build_clause_store(settings: Settings) -> ClauseStore:
    """The Clause store of the configured data directory, with the extractor in use."""
    return ClauseStore(settings.clause_store_path, RegexClauseExtractor())
