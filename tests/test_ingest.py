"""Ingestion against the real RC-2023-E PDF (no OCR, fake embeddings)."""

import json
import re
from collections import Counter
from typing import Any

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from case_mgmt_ai_demo.chunking import load_ingestion_config
from case_mgmt_ai_demo.chunks import Chunk, split_clauses
from case_mgmt_ai_demo.index import COLLECTION, ReglamentoIndex, chunk_id
from case_mgmt_ai_demo.ingest import ingest
from conftest import RC_PDF
from test_index import FakeEmbedder


def section(chunking, section_id: str) -> list[Chunk]:
    return [chunk for chunk in chunking.chunks if chunk.section_id == section_id]


def test_known_sections_exist_with_their_heading_and_parents(rc_chunking):
    [presentation, _] = section(rc_chunking, "RC-2023-E:2.1.9.2")
    [unico] = section(rc_chunking, "RC-2023-E:3.7.1.2")

    assert presentation.heading == "REQUISITOS DE PRESENTACIÓN"
    assert presentation.parents == ["Tomo II", "Capítulo 2.1", "Regla 2.1.9"]
    assert (unico.level, unico.heading) == ("seccion", "REQUISITOS PRESENTACIÓN PERMISO ÚNICO")
    assert unico.parents == ["Tomo III", "Capítulo 3.7", "Regla 3.7.1"]
    assert unico.request_types == ["PERMISO_UNICO"]


def test_every_section_id_and_part_is_unique(rc_chunking):
    ids = Counter(chunk_id(chunk) for chunk in rc_chunking.chunks)

    assert [id_ for id_, count in ids.items() if count > 1] == []


def test_a_regla_without_secciones_is_chunked_at_regla_level(rc_chunking):
    [extension] = section(rc_chunking, "RC-2023-E:2.2.6")

    assert extension.level == "regla"
    assert extension.parents == ["Tomo II", "Capítulo 2.2"]
    assert [clause.ref[-3:] for clause in split_clauses(extension)] == ["(a)", "(b)", "(c)"]


def test_no_chunk_contains_the_running_header(rc_chunking):
    remnants = [
        chunk.section_id
        for chunk in rc_chunking.chunks
        if re.search(r"REGLAMENTO CONJUNTO\s*\|\s*\d+|\d+\s*\|\s*REGLAMENTO CONJUNTO", chunk.text)
    ]

    assert remnants == []


def test_no_chunk_comes_from_the_table_of_contents_or_the_glossary(rc_chunking):
    config = load_ingestion_config()

    assert min(chunk.page_start for chunk in rc_chunking.chunks) == 54  # Tomo I opens on p. 54
    assert max(chunk.page_end for chunk in rc_chunking.chunks) <= config.last_page
    assert not [chunk.section_id for chunk in rc_chunking.chunks if "....." in chunk.text]


def test_a_long_section_is_split_into_parts_sharing_its_section_id(rc_chunking):
    config = load_ingestion_config()
    first, second = section(rc_chunking, "RC-2023-E:2.1.9.2")

    assert (first.part, second.part) == ("1-of-2", "2-of-2")
    assert first.text.startswith("a. El proponente de una solicitud")
    letters = [clause.ref[-2] for part in (first, second) for clause in split_clauses(part)]
    assert letters == list("abcdefghijklmnopqrstuvw")
    assert all(len(chunk.text) <= config.max_chunk_chars for chunk in rc_chunking.chunks)


@pytest.mark.parametrize(
    ("section_id", "page"),
    [
        ("RC-2023-E:2.1.9.2", 77),  # the running header on this page reads "| 25"
        ("RC-2023-E:2.1.9.5", 80),
        ("RC-2023-E:2.2.1.3", 92),
        ("RC-2023-E:3.7.1.1", 247),
        ("RC-2023-E:3.7.1.2", 249),
        ("RC-2023-E:4.1.4.1", 262),
    ],
)
def test_page_start_is_the_physical_page_holding_the_section_heading(rc_chunking, section_id, page):
    assert section(rc_chunking, section_id)[0].page_start == page


def test_each_part_of_a_split_section_has_its_own_page_range(rc_chunking):
    first, second = section(rc_chunking, "RC-2023-E:2.1.9.2")

    assert (first.page_start, first.page_end) == (77, 78)
    assert (second.page_start, second.page_end) == (78, 79)  # clause (k) starts on p. 78
    assert second.text.startswith("k. Todo plano radicado")


def test_section_text_matches_the_pdf_word_for_word(rc_chunking):
    [value] = section(rc_chunking, "RC-2023-E:2.1.9.3")

    assert value.text.split("\n")[0] == (
        "a. Para toda solicitud de permiso de construcción se someterá certificación del "
        "valor estimado de la obra."
    )
    assert (value.page_start, value.page_end) == (80, 80)


def test_a_districts_parameter_table_keeps_each_parameter_with_its_description(rc_chunking):
    [parameters] = section(rc_chunking, "RC-2023-E:6.1.3.4")

    assert "\nCabida mínima de solar\n275 metros cuadrados\nAncho mínimo de solar\n11 metros\n" in (
        parameters.text
    )
    assert (
        "\nDensidad poblacional\n"
        "  a. Solares menores a doscientos setenta y cinco (275) metros cuadrados, "
        "una unidad de vivienda.\n"
    ) in parameters.text
    assert [clause.ref for clause in split_clauses(parameters)] == ["RC-2023-E:6.1.3.4"]


def test_the_reglamentos_repeated_regla_6_1_16_keeps_both_districts(rc_chunking):
    [heavy_industry] = section(rc_chunking, "RC-2023-E:6.1.16.1")
    [rural] = section(rc_chunking, "RC-2023-E:6.1.16.1-bis")

    assert heavy_industry.parents[-1] == "Regla 6.1.16"
    assert rural.parents[-1] == "Regla 6.1.16-bis"
    assert rural.page_start == 472
    assert "RC-2023-E:6.1.16-bis" in rc_chunking.repeated


def test_a_capitulo_spelled_without_its_accent_still_holds_its_text(rc_chunking):
    [board] = section(rc_chunking, "RC-2023-E:1.2")

    assert (board.level, board.heading) == ("capitulo", "JUNTA DE PLANIFICACIÓN")


@pytest.fixture(scope="module")
def ingested(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    embedder = FakeEmbedder()
    result = ingest(RC_PDF, load_ingestion_config(), data_dir, data_dir / "chroma", embedder)
    return data_dir, result, embedder


def test_ingest_builds_a_persistent_index_holding_every_chunk(ingested):
    data_dir, result, embedder = ingested

    index = ReglamentoIndex(data_dir / "chroma")  # reopened from disk

    assert index.count() == result.report.chunks == len(embedder.embedded)
    assert result.indexed is not None and result.indexed.embedded == result.report.chunks
    [general] = index.section("RC-2023-E:3.7.1.1")
    assert (general.doc, general.page_start, general.page_end) == ("RC-2023-E", 247, 248)


def test_every_chunk_in_the_index_has_its_document_and_pages_in_the_metadata(ingested):
    data_dir, result, _ = ingested
    client = chromadb.PersistentClient(
        path=str(data_dir / "chroma"), settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection = client.get_collection(COLLECTION)

    metadatas: Any = collection.get(include=["metadatas"])["metadatas"]

    assert len(metadatas) == result.report.chunks
    for metadata in metadatas:
        assert metadata["doc"] == "RC-2023-E"
        assert 54 <= metadata["page_start"] <= metadata["page_end"] <= 824


def test_every_written_chunk_validates_against_the_chunk_contract(ingested):
    _, result, _ = ingested

    lines = result.chunks_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == result.report.chunks
    assert all(Chunk.model_validate_json(line) for line in lines)


def test_the_report_records_the_configuration_and_the_flagged_chunks(ingested):
    data_dir, result, _ = ingested

    report = json.loads(result.report_path.read_text(encoding="utf-8"))

    assert result.report_path.parent == (data_dir / "chroma").parent
    assert report["config"] == load_ingestion_config().model_dump()
    assert report["embedding_model"] == "fake-embedding"
    assert report["outlined_text"]["ocr"] is False
    assert report["outlined_text"]["unread"] > 300
    assert report["flags_by_check"]["unread_text"] > 100
    assert {
        "section_id": "RC-2023-E:2.1.9.5",
        "part": None,
        "check": "unread_text",
        "detail": "2 run(s) of text drawn as outlines were not read",
    } in report["flags"]
    assert "RC-2023-E:116.1.7" in {
        flag["section_id"] for flag in report["flags"] if flag["check"] == "section_out_of_sequence"
    }


def test_ingesting_again_embeds_nothing_and_leaves_the_same_report_and_index(ingested):
    data_dir, first, _ = ingested
    report_before = first.report_path.read_text(encoding="utf-8")
    again = FakeEmbedder()

    second = ingest(RC_PDF, load_ingestion_config(), data_dir, data_dir / "chroma", again)

    assert again.embedded == []
    assert second.indexed is not None
    assert (second.indexed.embedded, second.indexed.removed) == (0, 0)
    assert second.indexed.unchanged == first.report.chunks
    assert second.report_path.read_text(encoding="utf-8") == report_before
    assert ReglamentoIndex(data_dir / "chroma").count() == first.report.chunks
