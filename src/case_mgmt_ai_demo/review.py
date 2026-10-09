"""The review pipeline: one Permit Request in, one Recommendation out.

A LangGraph graph `intake → retrieve → validate_sections → select_requirements → decide →
write_notes → validate_citations`. The Section retriever, the Section validator, the requirement
selector, the Decisions API client and the Notes LLM are injected adapters, so tests can fake them.

The Approval Score is not one question over everything retrieved: each Requirement is assessed on
its own, and the score is the lowest probability. A Requirement is a Criterion (its pinned
Clauses) or a Section that applies (its other applicable Clauses). Sections that do not apply to
the request are set aside before any Clause is selected or scored.

A model adapter that fails raises ModelError; the pipeline records it in the Recommendation's
`errors` and carries on with the safest reading (see each node) rather than aborting the review.
"""

import operator
import re
from collections.abc import Collection, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Literal, Protocol, TypedDict, get_args

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError, computed_field
from pydantic_core import ErrorDetails

from case_mgmt_ai_demo.chunks import Chunk, Clause, RequestType, split_clauses
from case_mgmt_ai_demo.config import Settings
from case_mgmt_ai_demo.criteria import Criterion
from case_mgmt_ai_demo.permit_request import PermitRequest, parse_permit_request

# The compliance wording scored best in the CMAWD-2 spike (docs/spikes/cmawd-2-decisions-api.md).
SECTION_REQUIREMENT_INSTRUCTIONS = (
    "¿Cumple esta Solicitud de {type_name} con los requisitos citados de la Sección {number} "
    "del Reglamento?"
)

NOTES_INSTRUCTIONS = """\
Eres un asistente que apoya a un Revisor de solicitudes de permisos en Puerto Rico.
Escribe Notas en español que apoyen la Recomendación para la Solicitud de Permiso.
- La evaluación por requisito da, para cada requisito, la probabilidad de que la solicitud lo \
cumpla. Escribe una Nota por cada requisito con probabilidad menor que 0.5 o no evaluado, que \
explique qué falta o qué incumple la solicitud; después, de 1 a 3 Notas que confirmen los \
requisitos principales que sí se cumplen.
- Cada Nota tiene una o dos oraciones y se basa únicamente en las Secciones del Reglamento \
proporcionadas y en los campos de la solicitud.
- Cada Nota cita al menos una Sección en `section_ids`, usando exactamente el identificador \
entre corchetes (p. ej. RC-2023-E:2.1.9.5). Puede precisar el inciso: RC-2023-E:2.1.9.5(b).
- `curable` es verdadero cuando la Nota describe una deficiencia que el Solicitante puede \
subsanar (requerimiento de subsanación), y falso en otro caso."""

RELEVANCE_INSTRUCTIONS = """\
Eres un asistente que prepara la evaluación de una Solicitud de Permiso en Puerto Rico.
Una búsqueda recuperó la Sección del Reglamento que aparece abajo. Decide si esta Sección \
aplica a esta solicitud: si establece al menos un requisito que el Solicitante o su solicitud \
deban cumplir, dado el tipo de solicitud (y si es nueva o una renovación). Una Sección aplica \
AUNQUE la solicitud no cumpla o no mencione sus requisitos. No aplica si trata de otro tipo de \
trámite (p. ej. prórrogas cuando no se solicita una, renovaciones cuando la solicitud es nueva, \
solicitudes nuevas cuando es una renovación) o solo regula a la agencia. Da una razón breve en \
español."""

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
ModelStep = Literal["validate_sections", "select_requirements", "decide", "write_notes"]


class ClauseSelection(BaseModel):
    """Whether one clause is a requirement this Permit Request must meet, and why."""

    ref: str = Field(description="Clause reference, e.g. 'RC-2023-E:2.1.9.2(k)'.")
    category: ClauseCategory
    reason: str

    @property
    def set_aside(self) -> bool:
        """Only `aplica` clauses are requirements the request is scored against."""
        return self.category != "aplica"


class FixedCategory(ClauseSelection):
    """A clause category fixed by configuration instead of asked of the selector."""

    request_types: list[RequestType] | None = Field(
        default=None, description="The Request Types it applies to; None for every one."
    )

    def applies_to(self, request_type: RequestType) -> bool:
        return self.request_types is None or request_type in self.request_types


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


class SectionRelevance(BaseModel):
    """Whether one retrieved Section applies to this Permit Request, and why."""

    section_id: str
    applies: bool
    reason: str


class RequirementAssessment(BaseModel):
    """One requirement asked of the Decisions API: a Criterion, or what applies of a Section."""

    id: str = Field(description="Criterion ID, or 'seccion_' plus the Section number.")
    text_es: str
    clause_refs: list[str]
    probability: float | None = Field(description="None when the Decisions API refused.")
    criterion: Criterion | None = Field(
        default=None, description="None for a Section's Requirement."
    )


class ModelFailure(BaseModel):
    """A model call that failed; the review carried on without its answer."""

    step: ModelStep
    subject: str | None = Field(
        default=None, description="The Section or Requirement asked about; None for the step."
    )
    message: str


class Recommendation(BaseModel):
    request: PermitRequest
    approval_score: float | None = Field(
        description="The lowest Requirement probability; None when one was refused or "
        "nothing applied."
    )
    verdict: Verdict
    borderline: bool
    requirements: list[RequirementAssessment] = Field(
        description="Criteria in curated order, then Sections' Requirements in retrieval order."
    )
    notes: list[Note]
    retrieved_section_ids: list[str]
    section_relevance: list[SectionRelevance] = Field(
        description="Every retrieved Section, whether it applies, and why."
    )
    selection: list[ClauseSelection] = Field(
        description="Every retrieved clause the selector classified, in Reglamento order."
    )
    unclassified_clause_refs: list[str] = Field(
        description="Retrieved clauses the selector left out; kept for decide."
    )
    unknown_clause_refs: list[str] = Field(
        description="References the selector returned that match no retrieved clause; dropped."
    )
    errors: list[ModelFailure] = Field(
        default_factory=list, description="Model calls that failed, in pipeline order."
    )

    @property
    def set_aside_count(self) -> int:
        return sum(1 for s in self.selection if s.set_aside)

    @property
    def criterion_requirements(self) -> list[RequirementAssessment]:
        return [r for r in self.requirements if r.criterion]

    @computed_field
    @property
    def weakest_requirement(self) -> RequirementAssessment | None:
        """The Requirement that set the Approval Score (the first, on a tie)."""
        assessed = [r for r in self.requirements if r.probability is not None]
        if self.approval_score is None or not assessed:
            return None
        return min(assessed, key=lambda r: r.probability or 0.0)

    @property
    def sections_not_applicable(self) -> list[SectionRelevance]:
        return [r for r in self.section_relevance if not r.applies]


class InvalidPermitRequest(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


class MissingPinnedClause(LookupError):
    """A Criterion pins a Clause the retrieved Sections do not hold: a configuration error."""


class ModelError(RuntimeError):
    """A model adapter's call failed (API error or unusable response)."""


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


class SectionValidator(Protocol):
    def validate(self, chunks: list[Chunk], request: PermitRequest) -> SectionRelevance:
        """Whether the Section these chunks (all its parts) hold applies to the request."""
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
    section_relevance: list[SectionRelevance]
    applicable_sections: list[Chunk]
    clauses: list[Clause]
    selection: list[ClauseSelection]
    unclassified_clause_refs: list[str]
    unknown_clause_refs: list[str]
    requirements: list[RequirementAssessment]
    approval_score: float | None
    draft_notes: list[DraftNote]
    notes: list[Note]
    errors: Annotated[list[ModelFailure], operator.add]


class ReviewPipeline:
    def __init__(
        self,
        decisions: DecisionsClient,
        notes_llm: NotesLLM,
        retriever: SectionRetriever,
        selector: RequirementSelector,
        validator: SectionValidator,
        settings: Settings,
        fixed: Iterable[FixedCategory] = (),
        criteria: Iterable[Criterion] = (),
    ):
        """`fixed` fixes the category of general clauses the selector is unsteady on;
        `criteria` holds the Criteria of every Request Type."""
        self._decisions = decisions
        self._notes_llm = notes_llm
        self._retriever = retriever
        self._selector = selector
        self._validator = validator
        self._settings = settings
        self._fixed = list(fixed)
        self._criteria = list(criteria)
        graph = StateGraph(ReviewState, input_schema=ReviewInput)
        graph.add_node("intake", self._intake)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("validate_sections", self._validate_sections)
        graph.add_node("select_requirements", self._select_requirements)
        graph.add_node("decide", self._decide)
        graph.add_node("write_notes", self._write_notes)
        graph.add_node("validate_citations", self._validate_citations)
        graph.add_edge(START, "intake")
        graph.add_edge("intake", "retrieve")
        graph.add_edge("retrieve", "validate_sections")
        graph.add_edge("validate_sections", "select_requirements")
        graph.add_edge("select_requirements", "decide")
        graph.add_edge("decide", "write_notes")
        graph.add_edge("write_notes", "validate_citations")
        graph.add_edge("validate_citations", END)
        self._graph = graph.compile()

    def review(self, raw: Mapping[str, Any]) -> Recommendation:
        """Review a Permit Request given as raw fields; raises InvalidPermitRequest."""
        state = ReviewState(**self._graph.invoke({"raw": raw}))
        score = state["approval_score"]
        errors = state.get("errors", [])
        verdict = self._verdict(score, errors)
        return Recommendation(
            request=state["request"],
            approval_score=score,
            verdict=verdict,
            borderline=verdict != "manual_review"
            and score is not None
            and self._settings.borderline_low <= score <= self._settings.borderline_high,
            requirements=state["requirements"],
            notes=state["notes"],
            retrieved_section_ids=list(dict.fromkeys(c.section_id for c in state["sections"])),
            section_relevance=state["section_relevance"],
            selection=state["selection"],
            unclassified_clause_refs=state["unclassified_clause_refs"],
            unknown_clause_refs=state["unknown_clause_refs"],
            errors=errors,
        )

    def _verdict(self, score: float | None, errors: list[ModelFailure]) -> Verdict:
        # A failed model call leaves the score resting on a guess, so a Reviewer decides.
        if score is None or errors:
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
        return {"sections": sections}

    def _validate_sections(self, state: ReviewState) -> dict[str, Any]:
        """Ask once per Section, concurrently, whether it applies; a pinned Section always does."""
        by_section: dict[str, list[Chunk]] = {}
        for chunk in state["sections"]:
            by_section.setdefault(chunk.section_id, []).append(chunk)
        pinned_by = {
            section_id: [c.id for c in state["criteria"] if section_id in c.pinned_section_ids]
            for section_id in by_section
        }

        def validate(section_id: str) -> tuple[SectionRelevance, ModelFailure | None]:
            if criteria := pinned_by[section_id]:
                reason = f"Fijada por el Criterio {', '.join(criteria)}: siempre aplica."
                return SectionRelevance(section_id=section_id, applies=True, reason=reason), None
            try:
                return self._validator.validate(by_section[section_id], state["request"]), None
            except ModelError as error:
                # Kept, so its requirements are still scored rather than silently dropped.
                reason = "No se pudo validar (error del modelo): se mantiene."
                return (
                    SectionRelevance(section_id=section_id, applies=True, reason=reason),
                    ModelFailure(step="validate_sections", subject=section_id, message=str(error)),
                )

        with ThreadPoolExecutor(max_workers=max(len(by_section), 1)) as pool:
            validated = list(pool.map(validate, by_section))
        relevance = [r for r, _ in validated]
        applies = {r.section_id for r in relevance if r.applies}
        sections = [chunk for chunk in state["sections"] if chunk.section_id in applies]
        return {
            "section_relevance": relevance,
            "applicable_sections": sections,
            "clauses": [c for s in sections for c in split_clauses(s)],
            "errors": [error for _, error in validated if error],
        }

    def _select_requirements(self, state: ReviewState) -> dict[str, Any]:
        clauses = state["clauses"]
        request_type = state["request"].request_type
        fixed = {f.ref: f for f in self._fixed if f.applies_to(request_type)}
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
        asked = [c for c in clauses if c.ref not in fixed and c.ref not in required]
        returned: dict[str, ClauseSelection] = {}
        errors: list[ModelFailure] = []
        try:
            selections = self._selector.select(asked, state["request"])
        except ModelError as error:
            # Every asked clause is then unclassified, and so kept for decide.
            selections = []
            errors.append(ModelFailure(step="select_requirements", message=str(error)))
        for selection in selections:
            returned.setdefault(selection.ref, selection)
        returned |= {c.ref: fixed[c.ref] for c in clauses if c.ref in fixed}
        returned |= required
        known = {clause.ref for clause in clauses}
        return {
            "selection": [returned[c.ref] for c in clauses if c.ref in returned],
            "unclassified_clause_refs": [c.ref for c in clauses if c.ref not in returned],
            "unknown_clause_refs": [ref for ref in returned if ref not in known],
            "errors": errors,
        }

    def _decide(self, state: ReviewState) -> dict[str, Any]:
        """One predicate per Requirement, each over its own Clauses; each has its own
        input, so each is its own call, made concurrently."""
        request, clauses = state["request"], state["clauses"]
        # An unclassified clause may be a requirement, so it stays rather than being set aside.
        set_aside = {s.ref for s in state["selection"] if s.set_aside}
        questions: list[_Question] = [
            _Question(
                id=criterion.id,
                text_es=criterion.text_es,
                instructions=criterion.instructions,
                clauses=[c for c in clauses if criterion.pins(c)],
                criterion=criterion,
            )
            for criterion in state["criteria"]
        ]
        by_section: dict[str, list[Clause]] = {}
        for clause in clauses:
            if clause.ref not in set_aside and not any(c.pins(clause) for c in state["criteria"]):
                by_section.setdefault(clause.section_id, []).append(clause)
        for section_id, section_clauses in by_section.items():
            number = section_id.split(":", 1)[1]
            questions.append(
                _Question(
                    id="seccion_" + re.sub(r"[^a-z0-9]+", "_", number.lower()),
                    text_es=f"Requisitos aplicables de la Sección {number} "
                    f"({section_clauses[0].heading})",
                    instructions=SECTION_REQUIREMENT_INSTRUCTIONS.format(
                        type_name=request.type_name_es, number=number
                    ),
                    clauses=section_clauses,
                )
            )

        def ask(question: _Question) -> tuple[RequirementAssessment, ModelFailure | None]:
            context = _requirements_context(question.clauses, request)
            error = None
            try:
                answers = self._decisions.predicates(context, {question.id: question.instructions})
            except ModelError as failure:
                # Not assessed, as for a refusal: no score rather than a fabricated one.
                answers = {question.id: None}
                error = ModelFailure(step="decide", subject=question.id, message=str(failure))
            requirement = RequirementAssessment(
                id=question.id,
                text_es=question.text_es,
                clause_refs=[c.ref for c in question.clauses],
                probability=answers[question.id],
                criterion=question.criterion,
            )
            return requirement, error

        with ThreadPoolExecutor(max_workers=max(len(questions), 1)) as pool:
            answered = list(pool.map(ask, questions))
        requirements = [requirement for requirement, _ in answered]
        return {
            "requirements": requirements,
            "approval_score": _approval_score(requirements),
            "errors": [error for _, error in answered if error],
        }

    def _write_notes(self, state: ReviewState) -> dict[str, Any]:
        score = state["approval_score"]
        shown = "sin puntuación" if score is None else f"{score:.2f}"
        requirements = "\n".join(
            f"- [{r.id}] {r.text_es} ({', '.join(r.clause_refs)}): " + _probability_es(r.probability)
            for r in state["requirements"]
        )
        prompt = (
            f"{NOTES_INSTRUCTIONS}\n\n"
            f"{_context(state['applicable_sections'], state['request'])}\n\n"
            f"EVALUACIÓN POR REQUISITO (probabilidad de cumplimiento, 0-1)\n{requirements}\n\n"
            f"PUNTUACIÓN DE APROBACIÓN (0-1, la del requisito más débil): {shown}"
        )
        try:
            return {"draft_notes": self._notes_llm.write_notes(prompt)}
        except ModelError as error:
            # The score and verdict stand on their own; the Reviewer sees no Notes and the error.
            return {
                "draft_notes": [],
                "errors": [ModelFailure(step="write_notes", message=str(error))],
            }

    def _validate_citations(self, state: ReviewState) -> dict[str, Any]:
        # Only Sections that apply were shown to the Notes LLM, so only they may be cited.
        retrieved = {chunk.section_id for chunk in state["applicable_sections"]}
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


class _Question(BaseModel):
    """A Requirement before it is asked."""

    id: str
    text_es: str
    instructions: str
    clauses: list[Clause]
    criterion: Criterion | None = None


def _approval_score(requirements: list[RequirementAssessment]) -> float | None:
    """Approval needs every requirement met, so the score is the weakest requirement's."""
    probabilities = [r.probability for r in requirements]
    if not probabilities or None in probabilities:
        return None
    return min(p for p in probabilities if p is not None)


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


def relevance_prompt(chunks: list[Chunk], request: PermitRequest) -> str:
    """The Spanish prompt asking a model whether one Section applies to this request."""
    text = "\n".join(chunk.text for chunk in chunks)
    return (
        f"{RELEVANCE_INSTRUCTIONS}\n\nSECCIÓN DEL REGLAMENTO\n[{chunks[0].section_id}] "
        f"{chunks[0].heading}\n{text}\n\nSOLICITUD DE PERMISO\n{_fields(request)}"
    )


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
