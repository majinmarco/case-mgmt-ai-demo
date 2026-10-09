"""The review pipeline: one Permit Request in, one Recommendation out.

A LangGraph graph
`intake → retrieve → select_requirements → decide → write_notes → validate_citations`. The Section
retriever, the requirement selector, the Decisions API client and the Notes LLM are injected
adapters, so tests can fake them. The Criteria of the request's Request Type pin the Clauses
that are always retrieved, always scored against, and each assessed on its own.
"""

from collections.abc import Collection, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Protocol, TypedDict, get_args

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError
from pydantic_core import ErrorDetails

from case_mgmt_ai_demo.chunks import Chunk, Clause, RequestType, split_clauses
from case_mgmt_ai_demo.config import Settings
from case_mgmt_ai_demo.criteria import Criterion
from case_mgmt_ai_demo.permit_request import PermitRequest, parse_permit_request

APPROVAL_QUESTION = "aprobacion"
# The compliance wording scored best in the CMAWD-2 spike (docs/spikes/cmawd-2-decisions-api.md).
APPROVAL_INSTRUCTIONS = (
    "¿Cumple esta Solicitud de {type_name} con todos los requisitos de las "
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
subsanar (requerimiento de subsanación), y falso en otro caso.
- La evaluación por Criterio indica qué requisitos se cumplen (probabilidad alta) y cuáles no \
(probabilidad baja); apóyate en ella para elegir qué comentar."""

SELECTION_INSTRUCTIONS = """\
Eres un asistente que prepara la evaluación de una Solicitud de Permiso en Puerto Rico.
Clasifica CADA inciso del Reglamento listado abajo en una de tres categorías, para esta solicitud:
- "aplica": un requisito que el Solicitante o su solicitud deben cumplir para este tipo de \
solicitud. Clasifícalo como "aplica" AUNQUE la solicitud no lo cumpla o no lo mencione: \
un requisito incumplido u omitido sigue aplicando y es una deficiencia.
- "tramite_del_sistema": un paso procesal que realiza o verifica el sistema o la agencia \
(presentación y carga de documentos en el SUI, validación, plazos de la agencia, archivo), y \
que no se acredita con los datos de la solicitud. Los pagos de derechos y aranceles son \
"tramite_del_sistema" aunque el inciso se los imponga al solicitante: se verifican aparte.
- "no_aplica": no aplica a esta solicitud (p. ej. asuntos discrecionales cuando la solicitud \
es ministerial, proyectos públicos cuando el solicitante es privado, prórrogas cuando no se \
solicita una, requisitos de renovación cuando la solicitud es nueva y de solicitudes nuevas \
cuando es una renovación, obligaciones que surgen después de expedido el permiso, \
disposiciones meramente definitorias o declarativas sin requisito).
Devuelve una clasificación por inciso, con `ref` exactamente como aparece entre corchetes, y \
una razón breve en español."""

Verdict = Literal["approve", "not_approve", "manual_review"]
ClauseCategory = Literal["aplica", "tramite_del_sistema", "no_aplica"]


class ClauseSelection(BaseModel):
    """Whether one clause is a requirement this Permit Request must meet, and why."""

    ref: str = Field(description="Clause reference, e.g. 'RC-2023-E:2.1.9.2(k)'.")
    category: ClauseCategory
    reason: str

    @property
    def set_aside(self) -> bool:
        """Only `aplica` clauses are requirements the request is scored against."""
        return self.category != "aplica"


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


class CriterionAssessment(BaseModel):
    criterion: Criterion
    probability: float | None = Field(description="None when the Decisions API refused.")


class Recommendation(BaseModel):
    request: PermitRequest
    approval_score: float | None = Field(description="None when the Decisions API refused.")
    verdict: Verdict
    borderline: bool
    criteria: list[CriterionAssessment] = Field(
        description="The Criteria of the request's Request Type, in curated order."
    )
    notes: list[Note]
    retrieved_section_ids: list[str]
    selection: list[ClauseSelection] = Field(
        description="Every retrieved clause the selector classified, in Reglamento order."
    )
    unclassified_clause_refs: list[str] = Field(
        description="Retrieved clauses the selector left out; kept for decide."
    )
    unknown_clause_refs: list[str] = Field(
        description="References the selector returned that match no retrieved clause; dropped."
    )

    @property
    def set_aside_count(self) -> int:
        return sum(1 for s in self.selection if s.set_aside)


class InvalidPermitRequest(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


class MissingPinnedClause(LookupError):
    """A Criterion pins a Clause the retrieved Sections do not hold: a configuration error."""


class DecisionsClient(Protocol):
    def predicates(self, input: str, questions: Mapping[str, str]) -> dict[str, float | None]:
        """Probability per named predicate question; None for a refused question."""
        ...


class NotesLLM(Protocol):
    def write_notes(self, prompt: str) -> list[DraftNote]: ...


class SectionRetriever(Protocol):
    def retrieve(self, request: PermitRequest) -> list[Chunk]:
        """The Sections most relevant to the request."""
        ...

    def fetch(self, section_ids: Collection[str]) -> list[Chunk]:
        """Every chunk (all parts) of the given Sections."""
        ...


class RequirementSelector(Protocol):
    def select(self, clauses: list[Clause], request: PermitRequest) -> list[ClauseSelection]:
        """One classification per clause; may omit clauses or return unknown references."""
        ...


class ReviewInput(TypedDict):
    raw: Mapping[str, Any]


class ReviewState(ReviewInput):
    """Graph state; each node returns a partial update of it."""

    request: PermitRequest
    criteria: list[Criterion]
    sections: list[Chunk]
    clauses: list[Clause]
    selection: list[ClauseSelection]
    unclassified_clause_refs: list[str]
    unknown_clause_refs: list[str]
    approval_score: float | None
    criterion_probabilities: dict[str, float | None]
    draft_notes: list[DraftNote]
    notes: list[Note]


class ReviewPipeline:
    def __init__(
        self,
        decisions: DecisionsClient,
        notes_llm: NotesLLM,
        retriever: SectionRetriever,
        selector: RequirementSelector,
        settings: Settings,
        fixed: Iterable[ClauseSelection] = (),
        criteria: Iterable[Criterion] = (),
    ):
        """`fixed` fixes the category of general clauses the selector is unsteady on;
        `criteria` holds the Criteria of every Request Type."""
        self._decisions = decisions
        self._notes_llm = notes_llm
        self._retriever = retriever
        self._selector = selector
        self._settings = settings
        self._fixed = {selection.ref: selection for selection in fixed}
        self._criteria = list(criteria)
        graph = StateGraph(ReviewState, input_schema=ReviewInput)
        graph.add_node("intake", self._intake)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("select_requirements", self._select_requirements)
        graph.add_node("decide", self._decide)
        graph.add_node("write_notes", self._write_notes)
        graph.add_node("validate_citations", self._validate_citations)
        graph.add_edge(START, "intake")
        graph.add_edge("intake", "retrieve")
        graph.add_edge("retrieve", "select_requirements")
        graph.add_edge("select_requirements", "decide")
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
            criteria=[
                CriterionAssessment(
                    criterion=criterion,
                    probability=state["criterion_probabilities"][criterion.id],
                )
                for criterion in state["criteria"]
            ],
            notes=state["notes"],
            retrieved_section_ids=list(dict.fromkeys(c.section_id for c in state["sections"])),
            selection=state["selection"],
            unclassified_clause_refs=state["unclassified_clause_refs"],
            unknown_clause_refs=state["unknown_clause_refs"],
        )

    def _verdict(self, score: float | None) -> Verdict:
        if score is None:
            return "manual_review"
        return "approve" if score >= self._settings.approval_cutoff else "not_approve"

    def _intake(self, state: ReviewState) -> dict[str, Any]:
        try:
            request = parse_permit_request(state["raw"])
        except ValidationError as error:
            raise InvalidPermitRequest([_error_es(e) for e in error.errors()]) from error
        return {
            "request": request,
            "criteria": [c for c in self._criteria if c.applies_to(request)],
        }

    def _retrieve(self, state: ReviewState) -> dict[str, Any]:
        sections = self._retriever.retrieve(state["request"])
        pinned = [s for c in state["criteria"] for s in c.pinned_section_ids]
        found = {(chunk.section_id, chunk.part) for chunk in sections}
        sections += [
            chunk
            for chunk in self._retriever.fetch(set(pinned))
            if (chunk.section_id, chunk.part) not in found
        ]
        clauses = [c for s in sections for c in split_clauses(s)]
        for criterion in state["criteria"]:
            if unmatched := criterion.unmatched_pins(clauses):
                raise MissingPinnedClause(
                    f"Criterion {criterion.id} pins {', '.join(unmatched)}, "
                    "which no retrieved Section holds"
                )
        return {"sections": sections, "clauses": clauses}

    def _select_requirements(self, state: ReviewState) -> dict[str, Any]:
        clauses = state["clauses"]
        # Clauses a Criterion pins are requirements by definition, so the selector is not asked.
        required = {
            clause.ref: ClauseSelection(
                ref=clause.ref,
                category="aplica",
                reason=f"Fijado por el Criterio {', '.join(ids)}: siempre se evalúa.",
            )
            for clause in clauses
            if (ids := [c.id for c in state["criteria"] if c.pins(clause)])
        }
        asked = [c for c in clauses if c.ref not in self._fixed and c.ref not in required]
        returned: dict[str, ClauseSelection] = {}
        for selection in self._selector.select(asked, state["request"]):
            returned.setdefault(selection.ref, selection)
        returned |= {c.ref: self._fixed[c.ref] for c in clauses if c.ref in self._fixed}
        returned |= required
        known = {clause.ref for clause in clauses}
        return {
            "selection": [returned[c.ref] for c in clauses if c.ref in returned],
            "unclassified_clause_refs": [c.ref for c in clauses if c.ref not in returned],
            "unknown_clause_refs": [ref for ref in returned if ref not in known],
        }

    def _decide(self, state: ReviewState) -> dict[str, Any]:
        """One overall predicate over the selected Clauses, and one per Criterion over its own
        pinned Clauses; each has its own input, so each is its own call, made concurrently."""
        request, clauses = state["request"], state["clauses"]
        # An unclassified clause may be a requirement, so it stays rather than being set aside.
        set_aside = {s.ref for s in state["selection"] if s.set_aside}
        questions = {
            APPROVAL_QUESTION: (
                [c for c in clauses if c.ref not in set_aside],
                APPROVAL_INSTRUCTIONS.format(type_name=request.type_name_es),
            )
        } | {
            criterion.id: ([c for c in clauses if criterion.pins(c)], criterion.instructions)
            for criterion in state["criteria"]
        }

        def ask(name: str) -> float | None:
            requirements, instructions = questions[name]
            context = _requirements_context(requirements, request)
            return self._decisions.predicates(context, {name: instructions})[name]

        with ThreadPoolExecutor(max_workers=len(questions)) as pool:
            answers = dict(zip(questions, pool.map(ask, questions)))
        return {
            "approval_score": answers.pop(APPROVAL_QUESTION),
            "criterion_probabilities": answers,
        }

    def _write_notes(self, state: ReviewState) -> dict[str, Any]:
        score = state["approval_score"]
        shown = "sin puntuación (pregunta rechazada)" if score is None else f"{score:.2f}"
        prompt = (
            f"{NOTES_INSTRUCTIONS}\n\n"
            f"{_context(state['sections'], state['request'])}\n\n"
            f"PUNTUACIÓN DE APROBACIÓN (0-1): {shown}"
        )
        if state["criteria"]:
            assessments = "\n".join(
                f"- [{c.id}] {c.text_es} ({', '.join(c.pinned)}): "
                + _probability_es(state["criterion_probabilities"][c.id])
                for c in state["criteria"]
            )
            prompt += f"\n\nEVALUACIÓN POR CRITERIO (probabilidad de cumplimiento, 0-1)\n{assessments}"
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
    "less_than_equal": "debe ser menor o igual que {le}",
    "model_type": "la solicitud debe ser un objeto JSON",
    "model_attributes_type": "la solicitud debe ser un objeto JSON",
    "union_tag_not_found": "campo obligatorio",
    "union_tag_invalid": "valor no permitido (se espera {expected_tags})",
    "date_from_datetime_parsing": "debe ser una fecha (AAAA-MM-DD)",
}


def _error_es(error: ErrorDetails) -> str:
    template = INTAKE_ERRORS_ES.get(error["type"])
    message = template.format(**error.get("ctx", {})) if template else error["msg"]
    loc = list(error["loc"])
    if loc[:1] and loc[0] in get_args(RequestType):
        loc = loc[1:]  # the Request Type the fields were validated as, not a field
    elif error["type"].startswith("union_tag_"):
        loc = ["request_type"]
    return f"{'.'.join(map(str, loc)) or '(solicitud)'}: {message}"


def _probability_es(probability: float | None) -> str:
    return "no evaluado (pregunta rechazada)" if probability is None else f"{probability:.2f}"


def _was_retrieved(cited: str, retrieved: set[str]) -> bool:
    """A citation may go down to a clause: 'RC-2023-E:2.1.9.2(m)' matches 'RC-2023-E:2.1.9.2'."""
    return cited in retrieved or cited.split("(", 1)[0] in retrieved


def selection_prompt(clauses: list[Clause], request: PermitRequest) -> str:
    """The Spanish prompt asking a model to classify every clause for this request."""
    listing = "\n\n".join(f"[{c.ref}] {c.heading}\n{c.text}" for c in clauses)
    return (
        f"{SELECTION_INSTRUCTIONS}\n\nINCISOS DEL REGLAMENTO\n\n{listing}\n\n"
        f"SOLICITUD DE PERMISO\n{_fields(request)}"
    )


def _context(sections: list[Chunk], request: PermitRequest) -> str:
    """The Notes context: whole Section texts, then the request fields."""
    section_texts = "\n\n".join(
        f"[{chunk.section_id}] {chunk.heading}\n{chunk.text}" for chunk in sections
    )
    return (
        f"SECCIONES DEL REGLAMENTO\n\n{section_texts}\n\nSOLICITUD DE PERMISO\n{_fields(request)}"
    )


def _requirements_context(clauses: list[Clause], request: PermitRequest) -> str:
    """The decide context: the selected clauses grouped under their Section, then the fields."""
    by_section: dict[str, list[Clause]] = {}
    for clause in clauses:
        by_section.setdefault(clause.section_id, []).append(clause)
    section_texts = "\n\n".join(
        f"[{section_id}] {group[0].heading}\n" + "\n".join(c.text for c in group)
        for section_id, group in by_section.items()
    )
    return (
        f"SECCIONES DEL REGLAMENTO\n\n{section_texts}\n\nSOLICITUD DE PERMISO\n{_fields(request)}"
    )


def _fields(request: PermitRequest) -> str:
    return "\n".join(f"- {label}: {value}" for label, value in request.fields_es().items())
