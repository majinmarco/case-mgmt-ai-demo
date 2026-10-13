from collections.abc import Sequence
from types import SimpleNamespace

import openai
import pytest
from openai import OpenAI
from pydantic import BaseModel

from case_mgmt_ai_demo.adapters import (
    FixtureRetriever,
    IndexNotBuilt,
    IndexRetriever,
    OpenAIDecisions,
    OpenAINotesLLM,
    OpenAIRequirementSelector,
    OpenAISectionValidator,
    build_retriever,
    build_review_pipeline,
    load_fixed_clause_categories,
)
from case_mgmt_ai_demo.chunks import load_fixture_chunks, split_clauses
from case_mgmt_ai_demo.config import Settings
from case_mgmt_ai_demo.index import ReglamentoIndex
from case_mgmt_ai_demo.permit_request import parse_permit_request
from case_mgmt_ai_demo.review import ModelError
from test_index import AxisEmbedder
from test_review import GOOD_PCOC, GOOD_PERMISO_UNICO


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


def index_retriever(tmp_path, *keywords: str, top_k: int = 1) -> IndexRetriever:
    """A retriever over a real Chroma index of the fixture Sections, in which a request and a
    Section are near each other when they share the first of `keywords` either holds."""
    embedder = AxisEmbedder(*keywords)
    index = ReglamentoIndex(tmp_path / "chroma")
    index.sync("RC-2023-E", load_fixture_chunks(), embedder)
    return IndexRetriever(index, embedder, top_k)


def test_index_retriever_gives_the_section_nearest_the_request_with_its_similarity(tmp_path):
    retriever = index_retriever(tmp_path, "valor estimado de la obra")

    [found] = retriever.retrieve(parse_permit_request(GOOD_PCOC))

    assert found.chunk.section_id == "RC-2023-E:2.1.9.3"
    assert found.similarity == pytest.approx(1.0)


def test_index_retriever_gives_every_part_of_a_section_found_by_one_part(tmp_path):
    retriever = index_retriever(tmp_path, "catastro")  # only in part 1 of Sección 2.1.9.2

    found = retriever.retrieve(parse_permit_request(GOOD_PCOC))

    assert [(r.chunk.section_id, r.chunk.part, r.similarity) for r in found] == [
        ("RC-2023-E:2.1.9.2", "1-of-2", pytest.approx(1.0)),
        ("RC-2023-E:2.1.9.2", "2-of-2", pytest.approx(1.0)),
    ]


def test_index_retriever_gives_sections_nearest_first(tmp_path):
    retriever = index_retriever(tmp_path, "croquis", top_k=3)

    found = retriever.retrieve(parse_permit_request(GOOD_PERMISO_UNICO))

    similarities = [r.similarity or 0.0 for r in found]
    assert found[0].chunk.section_id == "RC-2023-E:3.7.1.2"
    assert similarities == sorted(similarities, reverse=True)
    assert similarities[0] > similarities[-1]


def test_index_retriever_gives_only_sections_of_the_request_type(tmp_path):
    chunks = load_fixture_chunks()
    retriever = index_retriever(tmp_path, "croquis", top_k=len(chunks))

    found = retriever.retrieve(parse_permit_request(GOOD_PERMISO_UNICO))

    assert {r.chunk.section_id for r in found} == {
        c.section_id for c in chunks if "PERMISO_UNICO" in c.request_types
    }


def test_index_retriever_fetches_every_part_of_a_section_by_id(tmp_path):
    chunks = index_retriever(tmp_path).fetch({"RC-2023-E:3.7.1.1", "RC-2023-E:2.1.9.2"})

    assert [(c.section_id, c.part) for c in chunks] == [
        ("RC-2023-E:2.1.9.2", "1-of-2"),
        ("RC-2023-E:2.1.9.2", "2-of-2"),
        ("RC-2023-E:3.7.1.1", None),
    ]


class FailingEmbedder:
    model = "text-embedding-3-large"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise openai.OpenAIError("Connection error.")


def test_an_embedding_error_during_retrieval_is_a_model_error(tmp_path):
    retriever = IndexRetriever(ReglamentoIndex(tmp_path / "chroma"), FailingEmbedder(), top_k=1)

    with pytest.raises(ModelError, match="text-embedding-3-large"):
        retriever.retrieve(parse_permit_request(GOOD_PCOC))


def test_the_configuration_chooses_the_fixture_retriever(tmp_path):
    settings = Settings(retriever="fixture", data_dir=tmp_path)

    assert isinstance(build_retriever(settings, OpenAI(api_key="sk-test")), FixtureRetriever)


def test_the_index_retriever_is_the_default_and_reads_the_configured_index(tmp_path):
    settings = Settings(data_dir=tmp_path)
    ReglamentoIndex(settings.index_dir).sync("RC-2023-E", load_fixture_chunks(), AxisEmbedder())

    retriever = build_retriever(settings, OpenAI(api_key="sk-test"))

    assert isinstance(retriever, IndexRetriever)
    assert [c.section_id for c in retriever.fetch({"RC-2023-E:2.1.9.5"})] == ["RC-2023-E:2.1.9.5"]


def test_an_index_that_holds_nothing_is_reported_rather_than_searched(tmp_path):
    with pytest.raises(IndexNotBuilt, match=str(tmp_path)):
        build_retriever(Settings(data_dir=tmp_path), OpenAI(api_key="sk-test"))


def test_an_index_holding_only_the_fee_order_is_reported_rather_than_searched(tmp_path):
    settings = Settings(data_dir=tmp_path)
    fee_item = load_fixture_chunks()[0].model_copy(
        update={"section_id": "COBROS:AnejoA:6", "doc": "COBROS-OA-2022-07", "part": None}
    )
    ReglamentoIndex(settings.index_dir).sync("COBROS-OA-2022-07", [fee_item], AxisEmbedder())

    with pytest.raises(IndexNotBuilt):
        build_retriever(settings, OpenAI(api_key="sk-test"))


def test_the_review_pipeline_keeps_its_clause_store_in_the_data_directory(tmp_path):
    settings = Settings(retriever="fixture", data_dir=tmp_path)

    build_review_pipeline(settings, OpenAI(api_key="sk-test"))

    assert settings.clause_store_path == tmp_path / "clauses.sqlite"
    assert settings.clause_store_path.is_file()
