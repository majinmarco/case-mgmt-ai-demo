"""The review pipeline: one Permit Request in, one Recommendation out.

A LangGraph graph `intake → retrieve → validate_sections → select_requirements → decide →
write_notes → validate_citations`; Notes citing a Section that does not apply go back to
write_notes once, and are dropped if they still do. The Section retriever, the Section validator,
the requirement selector, the Decisions API client and the Notes LLM are injected adapters, so
tests can fake them.

The Fee Check runs beside all of that, straight after intake. It is deterministic, never changes
the Approval Score, and adds its own curable Note when the declared fee is not the fee owed.

The Approval Score is not one question over everything retrieved: each Requirement is assessed on
its own, and the score is the lowest probability. A Requirement is a Criterion (its pinned
Clauses) or a Section that applies (its other applicable Clauses). Sections that do not apply to
the request are set aside before any Clause is selected or scored.

A model adapter that fails raises ModelError; the pipeline records it in the Recommendation's
`errors` and carries on with the safest reading (see each node) rather than aborting the review.

The Recommendation carries a pipeline trace: what retrieval returned, with similarity scores, and
what each node added to the graph state, in the order the nodes ran.
"""

import operator
import re
from collections.abc import Collection, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Literal, Protocol, TypedDict, get_args

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError, computed_field
from pydantic_core import ErrorDetails, to_jsonable_python

from case_mgmt_ai_demo.chunks import Chunk, Clause, RequestType, split_clauses
from case_mgmt_ai_demo.config import Settings
from case_mgmt_ai_demo.criteria import Criterion
from case_mgmt_ai_demo.fee_check import FeeCheck, FeeRules, check_fee
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

CITATION_CORRECTION_INSTRUCTIONS = """\
CORRECCIÓN DE CITAS
En el intento anterior, estas Notas no citaron ninguna Sección o citaron Secciones que no están \
entre las SECCIONES DEL REGLAMENTO proporcionadas. Escribe de nuevo todas las Notas y cita \
únicamente identificadores que aparecen entre corchetes en esas Secciones:"""

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
Clasifica CADA inciso del Reglamento listado abajo en una de cuatro categorías, para esta \
solicitud. Los CAMPOS DEL FORMULARIO son todo lo que un Solicitante puede declarar o adjuntar: \
no existe otro medio para acreditar un requisito.
- "aplica": un requisito que el Solicitante o su solicitud deben cumplir para este tipo de \
solicitud, y que algún campo del formulario permite acreditar. Clasifícalo como "aplica" \
AUNQUE la solicitud no lo cumpla o haya dejado ese campo vacío, sin declarar o sin el \
documento: un requisito incumplido u omitido sigue aplicando y es una deficiencia.
- "no_verificable": un requisito que sí aplica a esta solicitud, pero que NINGÚN campo del \
formulario permite acreditar: pide un documento, estudio, certificación, plano o dato que no \
figura entre los campos ni entre los documentos que el formulario admite. El Solicitante no \
puede cumplirlo por este medio, así que no se evalúa: lo verifica el Revisor. No lo uses \
cuando el campo existe y la solicitud lo dejó vacío o incompleto: eso es "aplica".
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
una razón breve en español; para "no_verificable", la razón nombra lo que el formulario no \
permite declarar."""

Verdict = Literal["approve", "not_approve", "manual_review"]
# `no_verificable`: a requirement that applies, but that no field of the form could evidence.
ClauseCategory = Literal["aplica", "no_verificable", "tramite_del_sistema", "no_aplica"]
# write_notes runs at most twice: once, then once more if a citation is invalid.
NOTES_ATTEMPTS = 2
ModelStep = Literal[
    "retrieve", "validate_sections", "select_requirements", "decide", "write_notes"
]
# How the graph is named in LangSmith, when tracing is on.
RUN_NAME = "permit_review"


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


class Note(BaseModel):
    """A Note as the Notes LLM writes it; it reaches the Recommendation if its citations hold."""

    text: str
    section_ids: list[str]
    curable: bool


class DroppedNote(Note):
    """A Note left out of the Recommendation: its citations were still invalid after the retry."""

    unretrieved_section_ids: list[str] = Field(
        description="Cited Sections that are not among those that apply; empty when it cited none."
    )

    def problem_es(self) -> str:
        """What is wrong with its citations, for the retry prompt and `review` output."""
        if self.unretrieved_section_ids:
            cited = ", ".join(self.unretrieved_section_ids)
            return f"{cited} no está entre las Secciones que aplican"
        return "no cita ninguna Sección"


class SectionRelevance(BaseModel):
    """Whether one retrieved Section applies to this Permit Request, and why."""

    section_id: str
    applies: bool
    reason: str


class Retrieved(BaseModel):
    """A chunk semantic retrieval returned, with how near the request it is."""

    chunk: Chunk
    similarity: float | None = Field(
        default=None,
        description="Cosine similarity to the request, the best of its Section's parts; "
        "None when the retriever does not score.",
    )


class RetrievedSection(BaseModel):
    """One Section the review worked from, and how it got there."""

    section_id: str
    heading: str
    similarity: float | None = Field(
        description="None when only a pin brought it, or the retriever does not score."
    )
    pinned_by: list[str] = Field(description="IDs of the applicable Criteria that pin it.")
    clause_refs: list[str] = Field(
        default_factory=list,
        description="Every Clause it holds; none is selected or scored when it does not apply.",
    )


class TraceStep(BaseModel):
    """What one run of a node added to the graph state."""

    node: str
    output: dict[str, Any] = Field(
        description="The state update as JSON; Sections, Clauses and Criteria by their IDs."
    )


class PipelineTrace(BaseModel):
    retrieved: list[RetrievedSection] = Field(
        default_factory=list, description="Semantic retrieval's Sections, nearest first, then pins."
    )
    steps: list[TraceStep] = Field(
        default_factory=list,
        description="In the order the nodes ran; write_notes and validate_citations twice "
        "after a retry.",
    )


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
    notes: list[Note] = Field(
        description="Each cites at least one Section that applies; then the Fee Check's Note, "
        "if any, citing its COBROS Sections."
    )
    notes_retried: bool = Field(
        default=False, description="write_notes was sent back once over invalid citations."
    )
    dropped_notes: list[DroppedNote] = Field(
        default_factory=list, description="Notes whose citations were still invalid after it."
    )
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
    fee_check: FeeCheck | None = Field(
        default=None, description="None when the pipeline has no fee rules."
    )
    trace: PipelineTrace = Field(default_factory=PipelineTrace)

    @property
    def set_aside_count(self) -> int:
        return sum(1 for s in self.selection if s.set_aside)

    @property
    def unverifiable(self) -> list[ClauseSelection]:
        """Requirements no field of the form could evidence: not scored, left to the Reviewer."""
        return [s for s in self.selection if s.category == "no_verificable"]

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
    def write_notes(self, prompt: str) -> list[Note]: ...


class SectionRetriever(Protocol):
    def retrieve(self, request: PermitRequest) -> list[Retrieved]:
        """The Sections most relevant to the request (all their parts), nearest first, among
        those of its Request Type."""
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
    retrieved: list[RetrievedSection]
    section_relevance: list[SectionRelevance]
    applicable_sections: list[Chunk]
    clauses: list[Clause]
    selection: list[ClauseSelection]
    unclassified_clause_refs: list[str]
    unknown_clause_refs: list[str]
    requirements: list[RequirementAssessment]
    approval_score: float | None
    notes_attempts: int
    draft_notes: list[Note]
    rejected_notes: list[DroppedNote]
    notes: list[Note]
    dropped_notes: list[DroppedNote]
    fee_check: FeeCheck | None
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
        fee_rules: FeeRules | None = None,
    ):
        """`fixed` fixes the category of general clauses the selector is unsteady on;
        `criteria` holds the Criteria of every Request Type; without `fee_rules` there is
        no Fee Check."""
        self._decisions = decisions
        self._notes_llm = notes_llm
        self._retriever = retriever
        self._selector = selector
        self._validator = validator
        self._settings = settings
        self._fixed = list(fixed)
        self._criteria = list(criteria)
        self._fee_rules = fee_rules
        graph = StateGraph(ReviewState, input_schema=ReviewInput)
        graph.add_node("intake", self._intake)
        graph.add_node("fee_check", self._fee_check)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("validate_sections", self._validate_sections)
        graph.add_node("select_requirements", self._select_requirements)
        graph.add_node("decide", self._decide)
        graph.add_node("write_notes", self._write_notes)
        graph.add_node("validate_citations", self._validate_citations)
        graph.add_edge(START, "intake")
        graph.add_edge("intake", "fee_check")  # in parallel with retrieval
        graph.add_edge("fee_check", END)
        graph.add_edge("intake", "retrieve")
        graph.add_edge("retrieve", "validate_sections")
        graph.add_edge("validate_sections", "select_requirements")
        graph.add_edge("select_requirements", "decide")
        graph.add_edge("decide", "write_notes")
        graph.add_edge("write_notes", "validate_citations")
        graph.add_conditional_edges("validate_citations", _after_citations, ["write_notes", END])
        self._graph = graph.compile()

    def review(self, raw: Mapping[str, Any]) -> Recommendation:
        """Review a Permit Request given as raw fields; raises InvalidPermitRequest."""
        steps: list[TraceStep] = []
        values: dict[str, Any] = {}
        payload: Any  # a node's update, or the whole state, by mode
        stream = self._graph.stream(
            {"raw": raw}, {"run_name": RUN_NAME}, stream_mode=["updates", "values"]
        )
        for mode, payload in stream:
            if mode == "values":
                values = payload
                continue
            steps += [
                TraceStep(node=node, output=_step_output(update))
                for node, update in payload.items()
            ]
        state = ReviewState(**values)
        score = state["approval_score"]
        errors = state.get("errors", [])
        verdict = self._verdict(score, errors)
        fee_check = state["fee_check"]
        return Recommendation(
            request=state["request"],
            approval_score=score,
            verdict=verdict,
            borderline=verdict != "manual_review"
            and score is not None
            and self._settings.borderline_low <= score <= self._settings.borderline_high,
            requirements=state["requirements"],
            notes=state["notes"] + _fee_notes(fee_check),
            notes_retried=state["notes_attempts"] > 1,  # the first attempt is not a retry
            dropped_notes=state["dropped_notes"],
            retrieved_section_ids=list(dict.fromkeys(c.section_id for c in state["sections"])),
            section_relevance=state["section_relevance"],
            selection=state["selection"],
            unclassified_clause_refs=state["unclassified_clause_refs"],
            unknown_clause_refs=state["unknown_clause_refs"],
            errors=errors,
            fee_check=fee_check,
            trace=PipelineTrace(retrieved=state["retrieved"], steps=steps),
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
            raise InvalidPermitRequest(
                [f"{field}: {message}" for field, message in intake_errors_es(error)]
            ) from error
        return {
            "request": request,
            "criteria": [c for c in self._criteria if c.applies_to(request)],
        }

    def _fee_check(self, state: ReviewState) -> dict[str, Any]:
        if self._fee_rules is None:
            return {"fee_check": None}
        return {"fee_check": check_fee(state["request"], self._fee_rules)}

    def _retrieve(self, state: ReviewState) -> dict[str, Any]:
        errors: list[ModelFailure] = []
        try:
            retrieved = self._retriever.retrieve(state["request"])
        except ModelError as error:
            # Pinned Sections are fetched by ID, so the Criteria are still assessed.
            retrieved = []
            errors.append(ModelFailure(step="retrieve", message=str(error)))
        sections = [r.chunk for r in retrieved]
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
        similarity = {r.chunk.section_id: r.similarity for r in retrieved}
        headings: dict[str, str] = {}
        for chunk in sections:
            headings.setdefault(chunk.section_id, chunk.heading)
        return {
            "sections": sections,
            "retrieved": [
                RetrievedSection(
                    section_id=section_id,
                    heading=heading,
                    similarity=similarity.get(section_id),
                    pinned_by=[c.id for c in state["criteria"] if section_id in c.pinned_section_ids],
                    clause_refs=[c.ref for c in clauses if c.section_id == section_id],
                )
                for section_id, heading in headings.items()
            ],
            "errors": errors,
        }

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
        if rejected := state.get("rejected_notes"):
            corrections = "\n".join(f"- «{note.text}»: {note.problem_es()}" for note in rejected)
            prompt += f"\n\n{CITATION_CORRECTION_INSTRUCTIONS}\n{corrections}"
        attempts = state.get("notes_attempts", 0) + 1
        try:
            return {"notes_attempts": attempts, "draft_notes": self._notes_llm.write_notes(prompt)}
        except ModelError as error:
            # The score and verdict stand on their own. A failed retry leaves the first
            # attempt's Notes, whose invalid ones are then dropped; a failed first attempt, none.
            return {
                "notes_attempts": attempts,
                "draft_notes": state.get("draft_notes", []),
                "errors": [ModelFailure(step="write_notes", message=str(error))],
            }

    def _validate_citations(self, state: ReviewState) -> dict[str, Any]:
        """Keep the Notes that cite only Sections that apply; after the retry, drop the rest."""
        # Only Sections that apply were shown to the Notes LLM, so only they may be cited.
        retrieved = {chunk.section_id for chunk in state["applicable_sections"]}
        valid: list[Note] = []
        rejected: list[DroppedNote] = []
        for draft in state["draft_notes"]:
            unretrieved = [c for c in draft.section_ids if not _was_retrieved(c, retrieved)]
            if draft.section_ids and not unretrieved:
                valid.append(draft)
            else:
                rejected.append(
                    DroppedNote(**draft.model_dump(), unretrieved_section_ids=unretrieved)
                )
        final = not _retry_notes(rejected, state["notes_attempts"])
        return {
            "rejected_notes": rejected,
            "notes": valid,
            "dropped_notes": rejected if final else [],
        }


class _Question(BaseModel):
    """A Requirement before it is asked."""

    id: str
    text_es: str
    instructions: str
    clauses: list[Clause]
    criterion: Criterion | None = None


def _step_output(update: Mapping[str, Any] | None) -> dict[str, Any]:
    """A node's state update for the trace. Sections, Clauses and Criteria go by their IDs:
    their text is in the index and the configuration."""
    output: dict[str, Any] = {}
    for key, value in (update or {}).items():
        if key in ("sections", "applicable_sections"):
            value = list(dict.fromkeys(chunk.section_id for chunk in value))
        elif key == "clauses":
            value = [clause.ref for clause in value]
        elif key == "criteria":
            value = [criterion.id for criterion in value]
        elif key == "requirements":
            value = [r.model_dump(exclude={"criterion"}) for r in value]
        output[key] = to_jsonable_python(value)
    return output


def _retry_notes(rejected: list[DroppedNote], attempts: int) -> bool:
    """Notes with invalid citations go back to write_notes once."""
    return bool(rejected) and attempts < NOTES_ATTEMPTS


def _after_citations(state: ReviewState) -> str:
    return "write_notes" if _retry_notes(state["rejected_notes"], state["notes_attempts"]) else END


def _fee_notes(fee_check: FeeCheck | None) -> list[Note]:
    """The Fee Check's Note: curable, since the Applicant can pay or declare the right fee."""
    if fee_check is None or (problem := fee_check.problem_es()) is None:
        return []
    return [Note(text=problem, section_ids=fee_check.section_ids, curable=True)]


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


def intake_errors_es(error: ValidationError) -> list[tuple[str, str]]:
    """Why raw fields are not a Permit Request: each field at fault, as its dotted path, with
    a Spanish message."""
    return [_error_es(details) for details in error.errors()]


def _error_es(error: ErrorDetails) -> tuple[str, str]:
    template = INTAKE_ERRORS_ES.get(error["type"])
    message = template.format(**error.get("ctx", {})) if template else error["msg"]
    loc = list(error["loc"])
    if loc[:1] and loc[0] in get_args(RequestType):
        loc = loc[1:]  # the Request Type the fields were validated as, not a field
    elif error["type"].startswith("union_tag_"):
        loc = ["request_type"]
    return ".".join(map(str, loc)) or "(solicitud)", message


def _probability_es(probability: float | None) -> str:
    return "no evaluado (pregunta rechazada)" if probability is None else f"{probability:.2f}"


def _was_retrieved(cited: str, retrieved: set[str]) -> bool:
    """A citation may go down to a clause: 'RC-2023-E:2.1.9.2(m)' matches 'RC-2023-E:2.1.9.2'."""
    return cited in retrieved or cited.split("(", 1)[0] in retrieved


def retrieval_query(request: PermitRequest) -> str:
    """What semantic retrieval looks for: the request, as its labelled Spanish fields."""
    return _fields(request)


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
    form = "\n".join(f"- {label}: {declares}" for label, declares in request.form_es().items())
    return (
        f"{SELECTION_INSTRUCTIONS}\n\nINCISOS DEL REGLAMENTO\n\n{listing}\n\n"
        f"CAMPOS DEL FORMULARIO (lo que cada uno permite declarar)\n{form}\n\n"
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
