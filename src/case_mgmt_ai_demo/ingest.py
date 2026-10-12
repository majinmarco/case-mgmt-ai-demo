"""Ingestion: a Reglamento PDF becomes Chunks in the index, with a report beside it."""

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from case_mgmt_ai_demo.chunking import IngestionConfig, chunk_lines
from case_mgmt_ai_demo.extraction import extract_lines
from case_mgmt_ai_demo.extraction_checks import Flag, check_chunks
from case_mgmt_ai_demo.index import EMBEDDED_TEXT, Embedder, ReglamentoIndex, SyncResult


class OutlinedText(BaseModel):
    """Text the PDF draws as outlines: how much of it there is and how much OCR read."""

    ocr: bool
    read: int
    unread: int
    readings: list[str]  # "p. 80: …", to check against the PDF


class IngestionReport(BaseModel):
    """What one ingestion did, with the configuration that produced it. Nothing in it
    depends on when it ran, so the same PDF and configuration give the same report."""

    pdf: str
    pdf_sha256: str
    config: IngestionConfig
    embedding_model: str | None
    embedded_text: str
    chunks: int
    sections: int
    split_sections: int
    chunks_by_level: dict[str, int]
    headings_without_text: dict[str, int]
    repeated_section_ids: list[str]
    outlined_text: OutlinedText
    flags_by_check: dict[str, int]
    flags: list[Flag]


@dataclass(frozen=True)
class Ingestion:
    report: IngestionReport
    report_path: Path
    chunks_path: Path
    indexed: SyncResult | None


def ingest(
    pdf: Path,
    config: IngestionConfig,
    data_dir: Path,
    index_dir: Path,
    embedder: Embedder | None,
    tessdata: Path | None = None,
) -> Ingestion:
    """Without an `embedder` the chunks and the report are written but the index is left
    alone, so chunking can be reviewed without calling the embeddings API."""
    extraction = extract_lines(
        pdf, config.first_page, config.last_page, config.footer_top, tessdata
    )
    chunking = chunk_lines(extraction.lines, config)
    chunks = chunking.chunks
    flags = check_chunks(chunks)
    read = [run for run in extraction.outlined_runs if run.recovered]
    report = IngestionReport(
        pdf=pdf.name,
        pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
        config=config,
        embedding_model=embedder.model if embedder else None,
        embedded_text=EMBEDDED_TEXT,
        chunks=len(chunks),
        sections=len({chunk.section_id for chunk in chunks}),
        split_sections=len({chunk.section_id for chunk in chunks if chunk.part}),
        chunks_by_level=dict(Counter(chunk.level for chunk in chunks)),
        headings_without_text=dict(Counter(heading.level for heading in chunking.skipped)),
        repeated_section_ids=chunking.repeated,
        outlined_text=OutlinedText(
            ocr=extraction.ocr,
            read=len(read),
            unread=len(extraction.outlined_runs) - len(read),
            readings=[f"p. {run.page}: {run.text}" for run in read],
        ),
        flags_by_check=dict(Counter(flag.check for flag in flags)),
        flags=flags,
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    report_path = data_dir / f"{config.doc}.ingestion_report.json"
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    chunks_path = data_dir / f"{config.doc}.chunks.jsonl"
    chunks_path.write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in chunks), encoding="utf-8"
    )
    indexed = ReglamentoIndex(index_dir).sync(config.doc, chunks, embedder) if embedder else None
    return Ingestion(
        report=report, report_path=report_path, chunks_path=chunks_path, indexed=indexed
    )
