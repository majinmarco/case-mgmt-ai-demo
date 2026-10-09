import pytest

from case_mgmt_ai_demo.chunks import load_fixture_chunks, split_clauses
from case_mgmt_ai_demo.criteria import Criterion, load_criteria

FIXTURE_CHUNKS = load_fixture_chunks()
FIXTURE_REFS = {clause.ref for chunk in FIXTURE_CHUNKS for clause in split_clauses(chunk)}


@pytest.mark.parametrize("request_type", ["PCOC", "PERMISO_UNICO"])
def test_each_request_type_has_five_to_eight_criteria(request_type):
    criteria = [c for c in load_criteria() if c.request_type == request_type]

    assert 5 <= len(criteria) <= 8


def test_criterion_ids_are_unique():
    ids = [criterion.id for criterion in load_criteria()]

    assert len(ids) == len(set(ids))


def test_every_pin_is_a_fixture_clause():
    for criterion in load_criteria():
        assert set(criterion.pinned) <= FIXTURE_REFS, criterion.id


def test_every_pinned_section_is_tagged_for_the_criterion_request_type():
    tags = {chunk.section_id: chunk.request_types for chunk in FIXTURE_CHUNKS}

    for criterion in load_criteria():
        for section_id in criterion.pinned_section_ids:
            assert criterion.request_type in tags[section_id], (criterion.id, section_id)


def test_criteria_cover_the_key_sections_of_each_request_type():
    pinned = {
        request_type: {
            s for c in load_criteria() if c.request_type == request_type for s in c.pinned_section_ids
        }
        for request_type in ("PCOC", "PERMISO_UNICO")
    }

    assert {
        "RC-2023-E:2.1.9.2",
        "RC-2023-E:2.1.9.3",
        "RC-2023-E:2.1.9.5",
        "RC-2023-E:2.2.1.3",
    } <= pinned["PCOC"]
    assert {
        "RC-2023-E:3.7.1.1",
        "RC-2023-E:3.7.1.2",
        "RC-2023-E:3.7.1.4",
        "RC-2023-E:4.1.4.1",
    } <= pinned["PERMISO_UNICO"]


def test_permiso_unico_has_criteria_for_new_requests_and_for_renewals():
    criteria = [c for c in load_criteria() if c.request_type == "PERMISO_UNICO"]

    assert any("renovacion" not in c.application_kinds for c in criteria)
    assert any("nueva" not in c.application_kinds for c in criteria)


def test_a_pin_names_its_section():
    criterion = Criterion(
        id="pu_renovacion",
        request_type="PERMISO_UNICO",
        text_es="…",
        text_en="…",
        pinned=["RC-2023-E:3.7.1.4(f)", "RC-2023-E:3.7.1.4(g)", "RC-2023-E:2.1.9.5"],
        instructions="¿…?",
    )

    assert criterion.pinned_section_ids == ["RC-2023-E:3.7.1.4", "RC-2023-E:2.1.9.5"]


@pytest.mark.parametrize("bad_id", ["PCOC-Legitimacion", "con espacio", ""])
def test_criterion_ids_are_usable_as_question_names(bad_id):
    with pytest.raises(ValueError):
        Criterion(
            id=bad_id,
            request_type="PCOC",
            text_es="…",
            text_en="…",
            pinned=["RC-2023-E:2.1.9.5(b)"],
            instructions="¿…?",
        )


def test_duplicate_criterion_ids_are_rejected(tmp_path):
    criterion = """
  - id: pcoc_legitimacion
    text_es: Legitimación activa.
    text_en: Standing.
    pinned: ["RC-2023-E:2.1.9.5(b)"]
    instructions: ¿Acredita la legitimación activa?
"""
    (tmp_path / "pcoc.yaml").write_text(
        f"request_type: PCOC\ncriteria:{criterion}{criterion}", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="pcoc_legitimacion"):
        load_criteria(tmp_path)
