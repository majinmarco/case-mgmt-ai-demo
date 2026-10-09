from case_mgmt_ai_demo.adapters import FixtureRetriever, load_fixed_clause_categories
from case_mgmt_ai_demo.chunks import load_fixture_chunks, split_clauses


def test_every_fixed_clause_is_a_fixture_clause_fixed_once():
    fixture_refs = {
        clause.ref for chunk in load_fixture_chunks() for clause in split_clauses(chunk)
    }

    fixed = [selection.ref for selection in load_fixed_clause_categories()]

    assert fixed and set(fixed) <= fixture_refs
    assert len(fixed) == len(set(fixed))


def test_the_general_completeness_clause_is_set_aside_by_configuration():
    fixed = {selection.ref: selection for selection in load_fixed_clause_categories()}

    assert fixed["RC-2023-E:2.1.9.2(m)"].category == "tramite_del_sistema"


def test_sealed_construction_plans_do_not_apply_to_a_permiso_unico_by_configuration():
    [plans] = [s for s in load_fixed_clause_categories() if s.ref == "RC-2023-E:2.1.9.2(k)"]

    assert plans.category == "no_aplica"
    assert plans.request_types == ["PERMISO_UNICO"]


def test_fixture_retriever_fetches_every_part_of_a_section_by_id():
    chunks = FixtureRetriever().fetch({"RC-2023-E:2.1.9.2", "RC-2023-E:3.7.1.1"})

    assert [(c.section_id, c.part) for c in chunks] == [
        ("RC-2023-E:2.1.9.2", "1-of-2"),
        ("RC-2023-E:2.1.9.2", "2-of-2"),
        ("RC-2023-E:3.7.1.1", None),
    ]
