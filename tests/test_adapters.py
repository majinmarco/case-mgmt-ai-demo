from case_mgmt_ai_demo.adapters import load_pinned_clauses
from case_mgmt_ai_demo.chunks import load_fixture_chunks, split_clauses


def test_every_pinned_clause_is_a_fixture_clause_pinned_once():
    fixture_refs = {
        clause.ref for chunk in load_fixture_chunks() for clause in split_clauses(chunk)
    }

    pinned = [selection.ref for selection in load_pinned_clauses()]

    assert pinned and set(pinned) <= fixture_refs
    assert len(pinned) == len(set(pinned))


def test_the_general_completeness_clause_is_set_aside_by_configuration():
    pinned = {selection.ref: selection for selection in load_pinned_clauses()}

    assert pinned["RC-2023-E:2.1.9.2(m)"].category == "tramite_del_sistema"
