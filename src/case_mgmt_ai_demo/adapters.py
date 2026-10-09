"""Real implementations of the review pipeline's adapters."""

import json
from collections.abc import Collection, Mapping
from importlib.resources import files

from openai import OpenAI
from pydantic import BaseModel

from case_mgmt_ai_demo.chunks import Chunk, Clause, load_fixture_chunks
from case_mgmt_ai_demo.config import Settings
from case_mgmt_ai_demo.criteria import load_criteria
from case_mgmt_ai_demo.permit_request import PermitRequest
from case_mgmt_ai_demo.review import (
    ClauseSelection,
    DraftNote,
    FixedCategory,
    ReviewPipeline,
    SectionRelevance,
    relevance_prompt,
    selection_prompt,
)


class OpenAIDecisions:
    """Asks named predicate questions in one `decisions.create` call."""

    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def predicates(self, input: str, questions: Mapping[str, str]) -> dict[str, float | None]:
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
    notes: list[DraftNote]


class OpenAINotesLLM:
    """Writes Notes with structured output `{text, section_ids, curable}`."""

    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def write_notes(self, prompt: str) -> list[DraftNote]:
        response = self._client.responses.parse(
            model=self._model, input=prompt, text_format=_NotesOutput
        )
        if response.output_parsed is None:
            raise RuntimeError(f"Notes model {self._model} returned no parsed Notes")
        return response.output_parsed.notes


class _SelectionOutput(BaseModel):
    classifications: list[ClauseSelection]


class OpenAIRequirementSelector:
    """Classifies every clause with structured output `{ref, category, reason}`."""

    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def select(self, clauses: list[Clause], request: PermitRequest) -> list[ClauseSelection]:
        response = self._client.responses.parse(
            model=self._model,
            input=selection_prompt(clauses, request),
            text_format=_SelectionOutput,
        )
        if response.output_parsed is None:
            raise RuntimeError(f"Selector model {self._model} returned no parsed selection")
        return response.output_parsed.classifications


class _RelevanceOutput(BaseModel):
    applies: bool
    reason: str


class OpenAISectionValidator:
    """Judges whether one Section applies, with structured output `{applies, reason}`."""

    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def validate(self, chunks: list[Chunk], request: PermitRequest) -> SectionRelevance:
        response = self._client.responses.parse(
            model=self._model,
            input=relevance_prompt(chunks, request),
            text_format=_RelevanceOutput,
        )
        if response.output_parsed is None:
            raise RuntimeError(f"Section validator {self._model} returned no parsed verdict")
        verdict = response.output_parsed
        return SectionRelevance(
            section_id=chunks[0].section_id, applies=verdict.applies, reason=verdict.reason
        )


class FixtureRetriever:
    """Stands in for Chroma retrieval: every fixture Section tagged with the Request Type."""

    def __init__(self):
        self._chunks = load_fixture_chunks()

    def retrieve(self, request: PermitRequest) -> list[Chunk]:
        return [chunk for chunk in self._chunks if request.request_type in chunk.request_types]

    def fetch(self, section_ids: Collection[str]) -> list[Chunk]:
        return [chunk for chunk in self._chunks if chunk.section_id in section_ids]


def load_fixed_clause_categories() -> list[FixedCategory]:
    """General clauses whose category is fixed by configuration instead of asked each time."""
    raw = files("case_mgmt_ai_demo.fixtures").joinpath("fixed_clause_categories.json").read_text()
    return [FixedCategory.model_validate(item) for item in json.loads(raw)]


def build_review_pipeline(settings: Settings, client: OpenAI) -> ReviewPipeline:
    return ReviewPipeline(
        OpenAIDecisions(client, settings.decisions_api_model),
        OpenAINotesLLM(client, settings.notes_model),
        FixtureRetriever(),
        OpenAIRequirementSelector(client, settings.selector_model),
        OpenAISectionValidator(client, settings.section_validator_model),
        settings,
        fixed=load_fixed_clause_categories(),
        criteria=load_criteria(),
    )
