"""The Reglamento index: Chunks and their embeddings in an embedded, persistent Chroma store.

Every field of a Chunk is kept in the index (the text as the document, the rest as
metadata), so a Section ID resolves to its PDF and pages without reading the PDF again.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import chromadb
from chromadb.config import Settings as ChromaSettings

from case_mgmt_ai_demo.chunks import Chunk, Doc, RequestType

COLLECTION = "reglamento"
EMBEDDED_TEXT = "parents joined by ' > ', then the heading, then the text, one per line"
SEARCHED_DOC: Doc = "RC-2023-E"
_BATCH = 100


class Embedder(Protocol):
    """Turns texts into vectors (fakeable in tests)."""

    model: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    similarity: float  # cosine similarity to the query: 1 for the same direction


@dataclass(frozen=True)
class SyncResult:
    embedded: int  # new or changed chunks
    unchanged: int
    removed: int


def chunk_id(chunk: Chunk) -> str:
    """Unique per chunk: the Section ID, plus the part number of a split Section."""
    return chunk.section_id if chunk.part is None else f"{chunk.section_id}#{chunk.part_number}"


def embedding_text(chunk: Chunk) -> str:
    """What is embedded: the Section's place in the Reglamento, its heading and its text."""
    return f"{' > '.join(chunk.parents)}\n{chunk.heading}\n{chunk.text}"


class ReglamentoIndex:
    def __init__(self, index_dir: Path):
        index_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(index_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        # Embeddings are always supplied, so Chroma's default embedding function is not used.
        self._collection = client.get_or_create_collection(
            COLLECTION, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
        )

    def sync(self, doc: Doc, chunks: Sequence[Chunk], embedder: Embedder) -> SyncResult:
        """Make the index hold exactly `chunks` for `doc`. Only new or changed chunks are
        embedded, so running it again with the same chunks changes nothing."""
        stored = self._collection.get(where={"doc": doc}, include=["metadatas"])
        fingerprints = {
            stored_id: metadata.get("fingerprint")
            for stored_id, metadata in zip(stored["ids"], stored["metadatas"] or [])
        }
        wanted = {chunk_id(chunk): chunk for chunk in chunks}
        if len(wanted) != len(chunks):
            raise ValueError("Two chunks share a Section ID and part number")
        changed = [
            (id_, chunk)
            for id_, chunk in wanted.items()
            if fingerprints.get(id_) != _fingerprint(chunk, embedder.model)
        ]
        for start in range(0, len(changed), _BATCH):
            batch = changed[start : start + _BATCH]
            self._collection.upsert(
                ids=[id_ for id_, _ in batch],
                embeddings=embedder.embed([embedding_text(chunk) for _, chunk in batch]),  # pyright: ignore[reportArgumentType]
                documents=[chunk.text for _, chunk in batch],
                metadatas=[_metadata(chunk, embedder.model) for _, chunk in batch],
            )
        stale = [id_ for id_ in fingerprints if id_ not in wanted]
        for start in range(0, len(stale), _BATCH):
            self._collection.delete(ids=stale[start : start + _BATCH])
        return SyncResult(
            embedded=len(changed), unchanged=len(wanted) - len(changed), removed=len(stale)
        )

    def section(self, section_id: str) -> list[Chunk]:
        """The chunk of a Section, or its parts in order; empty when the ID is unknown."""
        return self.sections([section_id])

    def sections(self, section_ids: Sequence[str]) -> list[Chunk]:
        """Every chunk of the given Sections, in the order asked for, each one's parts in
        order; unknown IDs give nothing."""
        if not section_ids:
            return []
        found = self._collection.get(
            where={"section_id": {"$in": list(section_ids)}}, include=["metadatas", "documents"]
        )
        chunks = [
            _chunk(metadata, text)
            for metadata, text in zip(found["metadatas"] or [], found["documents"] or [])
        ]
        asked = {section_id: position for position, section_id in enumerate(section_ids)}
        return sorted(chunks, key=lambda chunk: (asked[chunk.section_id], chunk.part_number))

    def search(
        self, embedding: Sequence[float], request_type: RequestType, top_k: int
    ) -> list[Hit]:
        """The `top_k` chunks nearest the query among those tagged with the Request Type,
        nearest first. Only the body of RC-2023-E is searched: a glossary term defines a word
        and the fee order sets charges (the Fee Check's), so neither states a requirement."""
        found = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            where={
                "$and": [
                    {"request_types": {"$contains": request_type}},
                    {"doc": SEARCHED_DOC},
                    {"level": {"$ne": "glossary_term"}},
                ]
            },
            include=["metadatas", "documents", "distances"],
        )
        metadatas = (found["metadatas"] or [[]])[0]
        documents = (found["documents"] or [[]])[0]
        distances = (found["distances"] or [[]])[0]
        return [
            Hit(chunk=_chunk(metadata, text), similarity=1 - distance)
            for metadata, text, distance in zip(metadatas, documents, distances)
        ]

    def count(self) -> int:
        return self._collection.count()

    def searchable(self) -> bool:
        """Whether the document `search` covers has been ingested."""
        found = self._collection.get(where={"doc": SEARCHED_DOC}, limit=1, include=[])
        return bool(found["ids"])


def _metadata(chunk: Chunk, model: str) -> dict[str, Any]:
    # Chroma rejects None and empty lists, so absent fields are left out.
    fields = chunk.model_dump(exclude={"text"})
    fields["fingerprint"] = _fingerprint(chunk, model)
    return {key: value for key, value in fields.items() if value not in (None, [])}


def _chunk(metadata: Any, text: str) -> Chunk:
    fields = {key: value for key, value in metadata.items() if key != "fingerprint"}
    return Chunk.model_validate({"parents": [], **fields, "text": text})


def _fingerprint(chunk: Chunk, model: str) -> str:
    return hashlib.sha256(f"{model}\n{chunk.model_dump_json()}".encode()).hexdigest()
