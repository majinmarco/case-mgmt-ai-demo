import pytest
from pydantic import ValidationError

from case_mgmt_ai_demo.chunks import Chunk, load_fixture_chunks, split_clauses


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


def test_split_section_parts_give_clauses_a_to_k_and_l_to_w():
    first, second = [c for c in load_fixture_chunks() if c.section_id == "RC-2023-E:2.1.9.2"]

    assert [clause.ref for clause in split_clauses(first)] == [
        f"RC-2023-E:2.1.9.2({letter})" for letter in "abcdefghijk"
    ]
    assert [clause.ref for clause in split_clauses(second)] == [
        f"RC-2023-E:2.1.9.2({letter})" for letter in "lmnopqrstuvw"
    ]


def test_clause_text_is_copied_word_for_word_with_its_numbered_items():
    [standing] = [c for c in load_fixture_chunks() if c.section_id == "RC-2023-E:2.1.9.5"]

    clauses = {clause.ref: clause for clause in split_clauses(standing)}

    b = clauses["RC-2023-E:2.1.9.5(b)"]
    assert b.text.startswith("b. Tendrá que incluirse evidencia fehaciente")
    assert b.text.endswith("autorización para hacer negocios en Puerto Rico o Estados Unidos.")
    assert "1. Escritura pública." in b.text
    assert (b.section_id, b.heading) == ("RC-2023-E:2.1.9.5", "LEGITIMACIÓN ACTIVA (STANDING)")


def test_every_fixture_clause_is_a_verbatim_slice_and_together_they_cover_the_text():
    for chunk in load_fixture_chunks():
        clauses = split_clauses(chunk)

        assert clauses, chunk.section_id
        assert "\n".join(clause.text for clause in clauses) == chunk.text


def test_parenthesised_sub_items_are_not_split_into_clauses():
    [unico] = [c for c in load_fixture_chunks() if c.section_id == "RC-2023-E:3.7.1.2"]

    assert [clause.ref for clause in split_clauses(unico)] == [
        "RC-2023-E:3.7.1.2(a)",
        "RC-2023-E:3.7.1.2(b)",
    ]


def test_text_before_the_first_clause_or_a_section_without_clauses_is_cited_by_section():
    [value] = [c for c in load_fixture_chunks() if c.section_id == "RC-2023-E:2.1.9.3"]
    with_preamble = value.model_copy(update={"text": "Disposiciones generales:\n" + value.text})
    unlettered = value.model_copy(update={"text": "Texto sin incisos."})

    assert [c.ref for c in split_clauses(with_preamble)] == [
        "RC-2023-E:2.1.9.3",
        "RC-2023-E:2.1.9.3(a)",
        "RC-2023-E:2.1.9.3(b)",
        "RC-2023-E:2.1.9.3(c)",
    ]
    assert split_clauses(with_preamble)[0].text == "Disposiciones generales:"
    assert [(c.ref, c.text) for c in split_clauses(unlettered)] == [
        ("RC-2023-E:2.1.9.3", "Texto sin incisos.")
    ]
