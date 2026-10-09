"""Real implementations of the review pipeline's adapters."""

from collections.abc import Mapping

from openai import OpenAI
from pydantic import BaseModel

from case_mgmt_ai_demo.chunks import Chunk, load_fixture_chunks
from case_mgmt_ai_demo.config import Settings
from case_mgmt_ai_demo.permit_request import PermitRequest
from case_mgmt_ai_demo.review import DraftNote, ReviewPipeline


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


class FixtureRetriever:
    """Stands in for Chroma retrieval: every fixture Section tagged with the Request Type."""

    def __init__(self):
        self._chunks = load_fixture_chunks()

    def retrieve(self, request: PermitRequest) -> list[Chunk]:
        return [chunk for chunk in self._chunks if request.request_type in chunk.request_types]


def build_review_pipeline(settings: Settings, client: OpenAI) -> ReviewPipeline:
    return ReviewPipeline(
        OpenAIDecisions(client, settings.decisions_api_model),
        OpenAINotesLLM(client, settings.notes_model),
        FixtureRetriever(),
        settings,
    )
