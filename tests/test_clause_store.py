from collections.abc import Sequence

import pytest

from case_mgmt_ai_demo.chunks import Chunk, Clause, load_fixture_chunks, split_clauses
from case_mgmt_ai_demo.clause_store import ClauseStore, RegexClauseExtractor
from case_mgmt_ai_demo.review import ModelError

STANDING = "RC-2023-E:2.1.9.5"
SPLIT = "RC-2023-E:2.1.9.2"  # two parts in the fixture


def section(section_id: str) -> list[Chunk]:
    return [chunk for chunk in load_fixture_chunks() if chunk.section_id == section_id]


class CountingExtractor:
    """The regex extractor, counting the Sections it is asked for; with `fail`, it raises
    ModelError instead."""

    def __init__(self, version: str = "counting-1", fail: bool = False):
        self.version = version
        self.fail = fail
        self.extracted: list[str] = []

    def extract(self, section: Sequence[Chunk]) -> list[Clause]:
        self.extracted.append(section[0].section_id)
        if self.fail:
            raise ModelError("Clause extractor: 500")
        return RegexClauseExtractor().extract(section)


def test_the_regex_extractor_gives_a_sections_clauses_across_its_parts_in_order():
    first, second = section(SPLIT)

    clauses = RegexClauseExtractor().extract([first, second])

    assert clauses == split_clauses(first) + split_clauses(second)
    assert len(clauses) > len(split_clauses(first))


def test_a_sections_clauses_are_extracted_once_and_then_read_from_the_store(tmp_path):
    extractor = CountingExtractor()
    store = ClauseStore(tmp_path / "clauses.sqlite", extractor)

    first = store.clauses(section(STANDING))
    second = store.clauses(section(STANDING))

    assert first == second == RegexClauseExtractor().extract(section(STANDING))
    assert ("RC-2023-E:2.1.9.5(a)", STANDING, "LEGITIMACIÓN ACTIVA (STANDING)") in [
        (c.ref, c.section_id, c.heading) for c in first
    ]
    assert extractor.extracted == [STANDING]


def test_stored_clauses_survive_a_restart(tmp_path):
    ClauseStore(tmp_path / "clauses.sqlite", CountingExtractor()).clauses(section(SPLIT))
    extractor = CountingExtractor()

    clauses = ClauseStore(tmp_path / "clauses.sqlite", extractor).clauses(section(SPLIT))

    assert clauses == RegexClauseExtractor().extract(section(SPLIT))
    assert extractor.extracted == []


def test_each_section_has_its_own_entry(tmp_path):
    extractor = CountingExtractor()
    store = ClauseStore(tmp_path / "clauses.sqlite", extractor)

    store.clauses(section(STANDING))
    split = store.clauses(section(SPLIT))

    assert {clause.section_id for clause in split} == {SPLIT}
    assert extractor.extracted == [STANDING, SPLIT]


def test_changed_section_text_is_extracted_again(tmp_path):
    extractor = CountingExtractor()
    store = ClauseStore(tmp_path / "clauses.sqlite", extractor)
    [chunk] = section(STANDING)
    store.clauses([chunk])
    changed = chunk.model_copy(update={"text": chunk.text + "\nz. Inciso añadido."})

    clauses = store.clauses([changed])

    assert clauses[-1].ref == "RC-2023-E:2.1.9.5(z)"
    assert extractor.extracted == [STANDING, STANDING]
    assert store.clauses([changed]) == clauses  # the new text's Clauses are now the stored ones
    assert extractor.extracted == [STANDING, STANDING]


def test_a_changed_extractor_extracts_again(tmp_path):
    ClauseStore(tmp_path / "clauses.sqlite", CountingExtractor("regex-1")).clauses(
        section(STANDING)
    )
    newer = CountingExtractor("llm-1")

    ClauseStore(tmp_path / "clauses.sqlite", newer).clauses(section(STANDING))

    assert newer.extracted == [STANDING]


def test_a_failed_extraction_is_not_stored(tmp_path):
    path = tmp_path / "clauses.sqlite"
    with pytest.raises(ModelError, match="Clause extractor: 500"):
        ClauseStore(path, CountingExtractor(fail=True)).clauses(section(STANDING))
    working = CountingExtractor()

    clauses = ClauseStore(path, working).clauses(section(STANDING))

    assert working.extracted == [STANDING]
    assert clauses == RegexClauseExtractor().extract(section(STANDING))


def test_a_failed_extraction_leaves_the_clauses_already_stored(tmp_path):
    path = tmp_path / "clauses.sqlite"
    [chunk] = section(STANDING)
    ClauseStore(path, CountingExtractor()).clauses([chunk])
    changed = chunk.model_copy(update={"text": chunk.text + "\nz. Inciso añadido."})
    with pytest.raises(ModelError):
        ClauseStore(path, CountingExtractor(fail=True)).clauses([changed])
    reader = CountingExtractor()

    ClauseStore(path, reader).clauses([chunk])

    assert reader.extracted == []


def test_clearing_one_document_leaves_the_others_clauses(tmp_path):
    cobros = Chunk(
        section_id="COBROS:AnejoA:6",
        doc="COBROS-OA-2022-07",
        level="anejo_item",
        heading="PERMISO DE CONSTRUCCIÓN",
        parents=["Anejo A"],
        page_start=9,
        page_end=9,
        text="a. Primer cargo.\nb. Segundo cargo.",
        request_types=["PCOC"],
    )
    extractor = CountingExtractor()
    store = ClauseStore(tmp_path / "clauses.sqlite", extractor)
    store.clauses(section(STANDING))
    store.clauses(section(SPLIT))
    store.clauses([cobros])

    assert store.clear("RC-2023-E") == 2

    store.clauses([cobros])
    store.clauses(section(STANDING))
    assert extractor.extracted == [STANDING, SPLIT, "COBROS:AnejoA:6", STANDING]


def test_pruning_forgets_only_the_sections_that_are_gone_or_changed(tmp_path):
    extractor = CountingExtractor()
    store = ClauseStore(tmp_path / "clauses.sqlite", extractor)
    [standing] = section(STANDING)
    for section_id in (STANDING, SPLIT, "RC-2023-E:2.1.9.3"):
        store.clauses(section(section_id))
    changed = standing.model_copy(update={"text": standing.text + "\nz. Inciso añadido."})

    forgotten = store.prune("RC-2023-E", [changed, *section(SPLIT)])  # 2.1.9.3 is gone

    assert forgotten == 2
    extractor.extracted.clear()
    store.clauses(section(SPLIT))
    assert extractor.extracted == []
    store.clauses([changed])
    assert extractor.extracted == [STANDING]


def test_pruning_one_document_leaves_the_others_clauses(tmp_path):
    extractor = CountingExtractor()
    store = ClauseStore(tmp_path / "clauses.sqlite", extractor)
    store.clauses(section(STANDING))

    assert store.prune("COBROS-OA-2022-07", []) == 0

    store.clauses(section(STANDING))
    assert extractor.extracted == [STANDING]


def test_clearing_the_store_removes_every_sections_clauses(tmp_path):
    extractor = CountingExtractor()
    store = ClauseStore(tmp_path / "clauses.sqlite", extractor)
    store.clauses(section(STANDING))

    assert store.clear() == 1
    assert store.clear() == 0

    store.clauses(section(STANDING))
    assert extractor.extracted == [STANDING, STANDING]


def test_chunks_of_different_sections_are_not_one_section(tmp_path):
    store = ClauseStore(tmp_path / "clauses.sqlite", CountingExtractor())

    with pytest.raises(ValueError, match="one Section"):
        store.clauses(section(STANDING) + section(SPLIT))
