import pytest
from pydantic import ValidationError

from case_mgmt_ai_demo.chunks import Chunk, load_fixture_chunks


def test_fixture_holds_about_ten_rc_sections_for_both_request_types():
    chunks = load_fixture_chunks()

    assert 8 <= len(chunks) <= 12
    assert {chunk.doc for chunk in chunks} == {"RC-2023-E"}
    assert all(chunk.section_id.startswith("RC-2023-E:") for chunk in chunks)
    tagged = {request_type for chunk in chunks for request_type in chunk.request_types}
    assert tagged == {"PCOC", "PERMISO_UNICO"}


def test_fixture_includes_the_pcoc_sections_used_in_the_spike():
    section_ids = {chunk.section_id for chunk in load_fixture_chunks()}

    assert {
        "RC-2023-E:2.1.9.2",
        "RC-2023-E:2.1.9.3",
        "RC-2023-E:2.1.9.5",
        "RC-2023-E:2.2.1.3",
    } <= section_ids


def test_a_split_section_keeps_one_section_id_across_its_parts():
    parts = [c for c in load_fixture_chunks() if c.section_id == "RC-2023-E:2.1.9.2"]

    assert [part.part for part in parts] == ["1-of-2", "2-of-2"]
    assert parts[0].text.startswith("a. ")
    assert "m. No se aceptará ninguna solicitud de servicio incompleta" in parts[1].text


def test_chunk_rejects_page_range_that_ends_before_it_starts():
    with pytest.raises(ValidationError):
        Chunk(
            section_id="RC-2023-E:2.1.9.3",
            doc="RC-2023-E",
            level="seccion",
            heading="VALOR ESTIMADO DE LA OBRA",
            parents=["Tomo II"],
            page_start=80,
            page_end=79,
            text="a. ...",
            request_types=["PCOC"],
        )


def test_chunk_rejects_section_id_from_another_document():
    with pytest.raises(ValidationError):
        Chunk(
            section_id="COBROS:AnejoA:6",
            doc="RC-2023-E",
            level="seccion",
            heading="X",
            parents=[],
            page_start=1,
            page_end=1,
            text="a. ...",
            request_types=["PCOC"],
        )
