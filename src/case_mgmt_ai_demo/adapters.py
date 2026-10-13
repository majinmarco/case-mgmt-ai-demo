"""Real implementations of the review pipeline's adapters."""

import json
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib.resources import files
from typing import TYPE_CHECKING

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from case_mgmt_ai_demo.chunks import Chunk, Clause, load_fixture_chunks
from case_mgmt_ai_demo.config import Settings
from case_mgmt_ai_demo.criteria import load_criteria
from case_mgmt_ai_demo.fee_check import load_fee_rules
from case_mgmt_ai_demo.permit_request import PermitRequest
from case_mgmt_ai_demo.review import (
    ClauseSelection,
    FixedCategory,
    ModelError,
    Note,
    Retrieved,
    ReviewPipeline,
    SectionRelevance,
    SectionRetriever,
    relevance_prompt,
    retrieval_query,
    selection_prompt,
)


if TYPE_CHECKING:  # imported when used: Chroma is slow to load
    from case_mgmt_ai_demo.index import Embedder, ReglamentoIndex


@contextmanager
def _model_errors(model: str) -> Iterator[None]:
    """Raise any OpenAI error or malformed output as a ModelError, so the pipeline records it
    and carries on."""
    try:
        yield
    except (OpenAIError, ValidationError) as error:
        raise ModelError(f"{model}: {error}") from error


def _parse[T: BaseModel](client: OpenAI, model: str, prompt: str, output: type[T], what: str) -> T:
    """One structured-output call; raises ModelError when it fails or nothing is parsed."""
    with _model_errors(model):
        response = client.responses.parse(model=model, input=prompt, text_format=output)
    if response.output_parsed is None:
        raise ModelError(f"{model} returned no parsed {what}")
    return response.output_parsed


class OpenAIDecisions:
    """Asks named predicate questions in one `decisions.create` call."""

    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def predicates(self, input: str, questions: Mapping[str, str]) -> dict[str, float | None]:
        with _model_errors(self._model):
            decision = self._client.decisions.create(
                model=self._model,
                input=input,
                questions=[
                    {"type": "predicate", "name": name, "instructions": instructions}
                    for name, instructions in questions.items()
                ],
            )
        # Refusals carry no probability; questions are named, so match answers by name.
        answered = {
            answer.name: answer.probability
            for answer in decision.answers
            if answer.type == "predicate"
        }
        return {name: answered.get(name) for name in questions}


class _NotesOutput(BaseModel):
    notes: list[Note]


class OpenAINotesLLM:
    """Writes Notes with structured output `{text, section_ids, curable}`."""

    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def write_notes(self, prompt: str) -> list[Note]:
        return _parse(self._client, self._model, prompt, _NotesOutput, "Notes").notes


class _SelectionOutput(BaseModel):
    classifications: list[ClauseSelection]


class OpenAIRequirementSelector:
    """Classifies every clause with structured output `{ref, category, reason}`."""

    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def select(self, clauses: list[Clause], request: PermitRequest) -> list[ClauseSelection]:
        prompt = selection_prompt(clauses, request)
        output = _parse(self._client, self._model, prompt, _SelectionOutput, "selection")
        return output.classifications


class _RelevanceOutput(BaseModel):
    applies: bool
    reason: str


class OpenAISectionValidator:
    """Judges whether one Section applies, with structured output `{applies, reason}`."""

    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def validate(self, chunks: list[Chunk], request: PermitRequest) -> SectionRelevance:
        prompt = relevance_prompt(chunks, request)
        verdict = _parse(self._client, self._model, prompt, _RelevanceOutput, "verdict")
        return SectionRelevance(
            section_id=chunks[0].section_id, applies=verdict.applies, reason=verdict.reason
        )


class OpenAIEmbedder:
    """Embeds chunk texts for the Reglamento index."""

    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self.model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in response.data]


class FixtureRetriever:
    """Stands in for Chroma retrieval: every fixture Section tagged with the Request Type,
    without a similarity."""

    def __init__(self):
        self._chunks = load_fixture_chunks()

    def retrieve(self, request: PermitRequest) -> list[Retrieved]:
        return [
            Retrieved(chunk=chunk)
            for chunk in self._chunks
            if request.request_type in chunk.request_types
        ]

    def fetch(self, section_ids: Collection[str]) -> list[Chunk]:
        return [chunk for chunk in self._chunks if chunk.section_id in section_ids]


class IndexRetriever:
    """Retrieval from the Reglamento index: the Sections nearest the request, among those of
    its Request Type."""

    def __init__(self, index: "ReglamentoIndex", embedder: "Embedder", top_k: int):
        self._index = index
        self._embedder = embedder
        self._top_k = top_k

    def retrieve(self, request: PermitRequest) -> list[Retrieved]:
        with _model_errors(self._embedder.model):
            [embedding] = self._embedder.embed([retrieval_query(request)])
        # Nearest first, so the first hit of a split Section is its best part.
        similarity: dict[str, float] = {}
        for hit in self._index.search(embedding, request.request_type, self._top_k):
            similarity.setdefault(hit.chunk.section_id, hit.similarity)
        # Every part of a Section, not only the part found: its Clauses are judged together.
        return [
            Retrieved(chunk=chunk, similarity=similarity[chunk.section_id])
            for chunk in self._index.sections(list(similarity))
        ]

    def fetch(self, section_ids: Collection[str]) -> list[Chunk]:
        return self._index.sections(sorted(section_ids))


class IndexNotBuilt(RuntimeError):
    """`review` is set to read the Reglamento index, and RC-2023-E is not in it."""


def build_retriever(settings: Settings, client: OpenAI) -> SectionRetriever:
    """The retriever the configuration names (`PRA_RETRIEVER`)."""
    if settings.retriever == "fixture":
        return FixtureRetriever()
    from case_mgmt_ai_demo.index import ReglamentoIndex

    index = ReglamentoIndex(settings.index_dir)
    if not index.searchable():
        raise IndexNotBuilt(str(settings.index_dir))
    embedder = OpenAIEmbedder(client, settings.embedding_model)
    return IndexRetriever(index, embedder, settings.retrieval_top_k)


def load_fixed_clause_categories() -> list[FixedCategory]:
    """General clauses whose category is fixed by configuration instead of asked each time."""
    raw = files("case_mgmt_ai_demo.fixtures").joinpath("fixed_clause_categories.json").read_text()
    return [FixedCategory.model_validate(item) for item in json.loads(raw)]


def build_review_pipeline(settings: Settings, client: OpenAI) -> ReviewPipeline:
    return ReviewPipeline(
        OpenAIDecisions(client, settings.decisions_api_model),
        OpenAINotesLLM(client, settings.notes_model),
        build_retriever(settings, client),
        OpenAIRequirementSelector(client, settings.selector_model),
        OpenAISectionValidator(client, settings.section_validator_model),
        settings,
        fixed=load_fixed_clause_categories(),
        criteria=load_criteria(),
        fee_rules=load_fee_rules(),
    )
