"""The review pipeline: one Permit Request in, one Recommendation out.

A LangGraph graph `intake → retrieve → decide → write_notes → validate_citations`. The Decisions
API client, the Notes LLM and the Section retriever are injected adapters, so tests can fake them.
"""

from collections.abc import Mapping
from typing import Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError
from pydantic_core import ErrorDetails

from case_mgmt_ai_demo.chunks import Chunk
from case_mgmt_ai_demo.config import Settings
from case_mgmt_ai_demo.permit_request import PermitRequest

APPROVAL_QUESTION = "aprobacion"
# The compliance wording scored best in the CMAWD-2 spike (docs/spikes/cmawd-2-decisions-api.md).
APPROVAL_INSTRUCTIONS = (
    "¿Cumple esta Solicitud de Permiso de Construcción con todos los requisitos de las "
    "Secciones del Reglamento citadas?"
)

NOTES_INSTRUCTIONS = """\
Eres un asistente que apoya a un Revisor de solicitudes de permisos en Puerto Rico.
Escribe de 2 a 5 Notas en español que apoyen la Recomendación para la Solicitud de Permiso.
- Cada Nota tiene una o dos oraciones y se basa únicamente en las Secciones del Reglamento \
proporcionadas y en los campos de la solicitud.
- Cada Nota cita al menos una Sección en `section_ids`, usando exactamente el identificador \
entre corchetes (p. ej. RC-2023-E:2.1.9.5). Puede precisar el inciso: RC-2023-E:2.1.9.5(b).
- `curable` es verdadero cuando la Nota describe una deficiencia que el Solicitante puede \
subsanar (requerimiento de subsanación), y falso en otro caso."""

Verdict = Literal["approve", "not_approve", "manual_review"]


class DraftNote(BaseModel):
    """A Note as the Notes LLM writes it, before its citations are checked."""

    text: str
    section_ids: list[str]
    curable: bool


class Note(DraftNote):
    unretrieved_section_ids: list[str] = Field(
        default_factory=list, description="Cited Sections that were not retrieved."
    )

    @property
    def flagged(self) -> bool:
        return bool(self.unretrieved_section_ids)


class Recommendation(BaseModel):
    request: PermitRequest
    approval_score: float | None = Field(description="None when the Decisions API refused.")
    verdict: Verdict
    borderline: bool
    notes: list[Note]
    retrieved_section_ids: list[str]


class InvalidPermitRequest(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


class DecisionsClient(Protocol):
    def predicates(self, input: str, questions: Mapping[str, str]) -> dict[str, float | None]:
        """Probability per named predicate question; None for a refused question."""
        ...


class NotesLLM(Protocol):
    def write_notes(self, prompt: str) -> list[DraftNote]: ...


class SectionRetriever(Protocol):
    def retrieve(self, request: PermitRequest) -> list[Chunk]: ...


class ReviewInput(TypedDict):
    raw: Mapping[str, Any]


class ReviewState(ReviewInput):
    """Graph state; each node returns a partial update of it."""

    request: PermitRequest
    sections: list[Chunk]
    approval_score: float | None
    draft_notes: list[DraftNote]
    notes: list[Note]


class ReviewPipeline:
    def __init__(
        self,
        decisions: DecisionsClient,
        notes_llm: NotesLLM,
        retriever: SectionRetriever,
        settings: Settings,
    ):
        self._decisions = decisions
        self._notes_llm = notes_llm
        self._retriever = retriever
        self._settings = settings
        graph = StateGraph(ReviewState, input_schema=ReviewInput)
        graph.add_node("intake", self._intake)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("decide", self._decide)
        graph.add_node("write_notes", self._write_notes)
        graph.add_node("validate_citations", self._validate_citations)
        graph.add_edge(START, "intake")
        graph.add_edge("intake", "retrieve")
        graph.add_edge("retrieve", "decide")
        graph.add_edge("decide", "write_notes")
        graph.add_edge("write_notes", "validate_citations")
        graph.add_edge("validate_citations", END)
        self._graph = graph.compile()

    def review(self, raw: Mapping[str, Any]) -> Recommendation:
        """Review a Permit Request given as raw fields; raises InvalidPermitRequest."""
        state = ReviewState(**self._graph.invoke({"raw": raw}))
        score = state["approval_score"]
        return Recommendation(
            request=state["request"],
            approval_score=score,
            verdict=self._verdict(score),
            borderline=score is not None
            and self._settings.borderline_low <= score <= self._settings.borderline_high,
            notes=state["notes"],
            retrieved_section_ids=list(dict.fromkeys(c.section_id for c in state["sections"])),
        )

    def _verdict(self, score: float | None) -> Verdict:
        if score is None:
            return "manual_review"
        return "approve" if score >= self._settings.approval_cutoff else "not_approve"

    def _intake(self, state: ReviewState) -> dict[str, Any]:
        try:
            return {"request": PermitRequest.model_validate(state["raw"])}
        except ValidationError as error:
            raise InvalidPermitRequest([_error_es(e) for e in error.errors()]) from error

    def _retrieve(self, state: ReviewState) -> dict[str, Any]:
        return {"sections": self._retriever.retrieve(state["request"])}

    def _decide(self, state: ReviewState) -> dict[str, Any]:
        answers = self._decisions.predicates(
            _context(state["sections"], state["request"]),
            {APPROVAL_QUESTION: APPROVAL_INSTRUCTIONS},
        )
        return {"approval_score": answers[APPROVAL_QUESTION]}

    def _write_notes(self, state: ReviewState) -> dict[str, Any]:
        score = state["approval_score"]
        shown = "sin puntuación (pregunta rechazada)" if score is None else f"{score:.2f}"
        prompt = (
            f"{NOTES_INSTRUCTIONS}\n\n"
            f"{_context(state['sections'], state['request'])}\n\n"
            f"PUNTUACIÓN DE APROBACIÓN (0-1): {shown}"
        )
        return {"draft_notes": self._notes_llm.write_notes(prompt)}

    def _validate_citations(self, state: ReviewState) -> dict[str, Any]:
        retrieved = {chunk.section_id for chunk in state["sections"]}
        return {
            "notes": [
                Note(
                    **draft.model_dump(),
                    unretrieved_section_ids=[
                        cited for cited in draft.section_ids if not _was_retrieved(cited, retrieved)
                    ],
                )
                for draft in state["draft_notes"]
            ]
        }


# Spanish messages for the Pydantic errors a Permit Request file is likely to hit.
INTAKE_ERRORS_ES = {
    "missing": "campo obligatorio",
    "extra_forbidden": "campo no reconocido",
    "string_pattern_mismatch": "formato no válido",
    "string_too_short": "no puede estar vacío",
    "literal_error": "valor no permitido (se espera {expected})",
    "greater_than": "debe ser mayor que {gt}",
    "greater_than_equal": "debe ser mayor o igual que {ge}",
    "int_parsing": "debe ser un número entero",
    "decimal_parsing": "debe ser un número",
    "model_type": "la solicitud debe ser un objeto JSON",
}


def _error_es(error: ErrorDetails) -> str:
    template = INTAKE_ERRORS_ES.get(error["type"])
    message = template.format(**error.get("ctx", {})) if template else error["msg"]
    return f"{'.'.join(map(str, error['loc'])) or '(solicitud)'}: {message}"


def _was_retrieved(cited: str, retrieved: set[str]) -> bool:
    """A citation may go down to inciso: 'RC-2023-E:2.1.9.2(m)' matches 'RC-2023-E:2.1.9.2'."""
    return cited in retrieved or cited.split("(", 1)[0] in retrieved


def _context(sections: list[Chunk], request: PermitRequest) -> str:
    """The shared Spanish context: Section texts, then the request fields."""
    section_texts = "\n\n".join(
        f"[{chunk.section_id}] {chunk.heading}\n{chunk.text}" for chunk in sections
    )
    fields = "\n".join(f"- {label}: {value}" for label, value in request.fields_es().items())
    return f"SECCIONES DEL REGLAMENTO\n\n{section_texts}\n\nSOLICITUD DE PERMISO\n{fields}"
