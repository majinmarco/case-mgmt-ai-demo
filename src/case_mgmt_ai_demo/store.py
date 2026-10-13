"""Permit Requests and their Recommendations, kept in SQLite so they survive a restart.

Each is stored whole, as the JSON of its model: nothing queries inside them yet.
"""

import json
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from case_mgmt_ai_demo.permit_request import (
    PcocPermitRequest,
    PermisoUnicoPermitRequest,
    parse_permit_request,
)
from case_mgmt_ai_demo.review import Recommendation

_SCHEMA = """
CREATE TABLE IF NOT EXISTS permit_requests (
    id INTEGER PRIMARY KEY,
    fields TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS recommendations (
    permit_request_id INTEGER PRIMARY KEY REFERENCES permit_requests (id),
    body TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


@dataclass(frozen=True)
class StoredPermitRequest:
    """A Permit Request with the ID it is stored under."""

    id: int
    request: PcocPermitRequest | PermisoUnicoPermitRequest


class PermitRequestStore:
    """The SQLite file at `path`, created with its tables when it does not exist."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        with self._connection() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """One connection per operation, so the store can be used from any thread."""
        connection = sqlite3.connect(self._path)
        try:
            with connection:  # commits, or rolls back on an error
                connection.execute("PRAGMA foreign_keys = ON")
                yield connection
        finally:
            connection.close()

    def add(self, request: PcocPermitRequest | PermisoUnicoPermitRequest) -> int:
        """Store a Permit Request and return its ID."""
        with self._connection() as connection:
            cursor = connection.execute(
                "INSERT INTO permit_requests (fields) VALUES (?)", (request.model_dump_json(),)
            )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def all(self) -> list[StoredPermitRequest]:
        """Every Permit Request, in the order stored."""
        with self._connection() as connection:
            rows = connection.execute("SELECT id, fields FROM permit_requests ORDER BY id")
            return [_stored(*row) for row in rows]

    def get(self, request_id: int) -> StoredPermitRequest | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, fields FROM permit_requests WHERE id = ?", (request_id,)
            ).fetchone()
        return _stored(*row) if row else None

    def save_recommendation(self, request_id: int, recommendation: Recommendation) -> None:
        """Store the Recommendation of a Permit Request, in place of any it already has."""
        with self._connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO recommendations (permit_request_id, body) VALUES (?, ?)",
                (request_id, recommendation.model_dump_json()),
            )

    def recommendation(self, request_id: int) -> Recommendation | None:
        """The stored Recommendation; None when there is none, or when it was stored by a
        version whose Recommendations this one cannot read, so the request is reviewed again."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT body FROM recommendations WHERE permit_request_id = ?", (request_id,)
            ).fetchone()
        if row is None:
            return None
        try:
            return Recommendation.model_validate_json(row[0])
        except ValidationError as error:
            print(
                f"Recomendación de la solicitud {request_id} ilegible, se ignora: "
                f"{error.error_count()} errores de validación",
                file=sys.stderr,
            )
            return None


def _stored(request_id: int, fields: str) -> StoredPermitRequest:
    return StoredPermitRequest(id=request_id, request=parse_permit_request(json.loads(fields)))
