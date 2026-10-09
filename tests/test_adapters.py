from types import SimpleNamespace

import openai
import pytest
from pydantic import BaseModel

from case_mgmt_ai_demo.adapters import (
    FixtureRetriever,
    OpenAIDecisions,
    OpenAINotesLLM,
    OpenAIRequirementSelector,
    OpenAISectionValidator,
    load_fixed_clause_categories,
)
from case_mgmt_ai_demo.chunks import load_fixture_chunks, split_clauses
from case_mgmt_ai_demo.permit_request import parse_permit_request
from case_mgmt_ai_demo.review import ModelError
from test_review import GOOD_PCOC


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


def failing(*_args, **_kwargs):
    raise openai.OpenAIError("Connection error.")


class _Strict(BaseModel):
    n: int


def malformed(*_args, **_kwargs):
    _Strict.model_validate({"n": "no es un número"})


def unparsed(*_args, **_kwargs):
    return SimpleNamespace(output_parsed=None)


FAILING_CLIENT = SimpleNamespace(
    decisions=SimpleNamespace(create=failing), responses=SimpleNamespace(parse=failing)
)
UNPARSED_CLIENT = SimpleNamespace(responses=SimpleNamespace(parse=unparsed))
MALFORMED_CLIENT = SimpleNamespace(responses=SimpleNamespace(parse=malformed))
SECTION = [c for c in load_fixture_chunks() if c.section_id == "RC-2023-E:2.1.9.3"]
CALLS = {
    "decisions": lambda client: OpenAIDecisions(client, "m").predicates("x", {"q": "¿?"}),
    "notes": lambda client: OpenAINotesLLM(client, "m").write_notes("x"),
    "selector": lambda client: OpenAIRequirementSelector(client, "m").select(
        split_clauses(SECTION[0]), parse_permit_request(GOOD_PCOC)
    ),
    "validator": lambda client: OpenAISectionValidator(client, "m").validate(
        SECTION, parse_permit_request(GOOD_PCOC)
    ),
}


@pytest.mark.parametrize("adapter", CALLS)
def test_an_openai_error_is_raised_as_a_model_error(adapter):
    with pytest.raises(ModelError, match="Connection error") as error:
        CALLS[adapter](FAILING_CLIENT)

    assert isinstance(error.value.__cause__, openai.OpenAIError)


@pytest.mark.parametrize("adapter", ["notes", "selector", "validator"])
def test_a_response_with_nothing_parsed_is_a_model_error(adapter):
    with pytest.raises(ModelError, match="no parsed"):
        CALLS[adapter](UNPARSED_CLIENT)


@pytest.mark.parametrize("adapter", ["notes", "selector", "validator"])
def test_malformed_structured_output_is_a_model_error(adapter):
    with pytest.raises(ModelError, match="validation error"):
        CALLS[adapter](MALFORMED_CLIENT)
