import copy
from collections.abc import Collection, Mapping

import pytest

from case_mgmt_ai_demo.chunks import Chunk, Clause, load_fixture_chunks, split_clauses
from case_mgmt_ai_demo.config import Settings
from case_mgmt_ai_demo.criteria import Criterion
from case_mgmt_ai_demo.permit_request import PermitRequest
from case_mgmt_ai_demo.review import (
    ClauseCategory,
    ClauseSelection,
    DraftNote,
    FixedCategory,
    InvalidPermitRequest,
    MissingPinnedClause,
    ReviewPipeline,
    SectionRelevance,
)

GOOD_PCOC = {
    "request_type": "PCOC",
    "applicant_name": "María Isabel Rivera Ortiz",
    "applicant_type": "privado",
    "applicant_role": "dueno",
    "standing_document": "escritura_publica",
    "municipio": "Caguas",
    "catastro": "254-087-123-45",
    "calificacion": "R-I (Residencial Intermedio); uso residencial unifamiliar permitido; altura máxima 2 plantas",
    "attached_documents": ["planos_sellados", "certificacion_valor_obra"],
    "work_description": "Construcción de vivienda unifamiliar de 2 plantas",
    "estimated_work_value": 185000,
    "floors": 2,
    "height_m": 7.5,
    "gross_floor_area_m2": 210,
    "declared_fee": 1850,
}

GOOD_PERMISO_UNICO = {
    "request_type": "PERMISO_UNICO",
    "applicant_name": "Ana Lucía Torres Rivera",
    "applicant_type": "privado",
    "applicant_role": "arrendatario",
    "standing_document": "contrato_arrendamiento",
    "municipio": "Ponce",
    "catastro": "315-042-218-09",
    "calificacion": "C-L (Comercial Liviano); cafetería permitida ministerialmente",
    "attached_documents": [
        "memorial_explicativo",
        "croquis",
        "fotos_instalacion",
        "exclusion_categorica",
    ],
    "proposed_use": "Cafetería con servicio de mesas, sin venta de bebidas alcohólicas",
    "use_authorization": "ministerial",
    "employees": 6,
    "parking_spaces": 8,
    "operating_hours": "lunes a sábado, de 6:00 a. m. a 3:00 p. m.",
    "licenses_requested": ["sanitaria", "prevencion_incendios"],
    "validity_years": 3,
    "declared_fee": 150,
}
RENEWAL = {
    "permit_number": "2022-498765-PU-000987",
    "expiry_date": "2023-11-30",
    "filing_date": "2026-09-15",
    "use_conforms_certification": False,
}


class FakeDecisions:
    """Answers each named question from `probabilities`, else with `default`."""

    def __init__(
        self,
        probabilities: Mapping[str, float | None] | None = None,
        default: float | None = 0.9,
    ):
        self.probabilities = probabilities or {}
        self.default = default
        self.calls: list[tuple[str, dict[str, str]]] = []

    def predicates(self, input: str, questions: Mapping[str, str]) -> dict[str, float | None]:
        self.calls.append((input, dict(questions)))
        return {name: self.probabilities.get(name, self.default) for name in questions}


def all_inputs(decisions: FakeDecisions) -> str:
    """Every input the Decisions API was given, joined: what the Requirements were assessed on."""
    return "\n\n".join(input for input, _ in decisions.calls)


def call_for(decisions: FakeDecisions, question: str) -> tuple[str, dict[str, str]]:
    [call] = [call for call in decisions.calls if question in call[1]]
    return call


class FakeSectionValidator:
    """Every Section applies unless listed in `not_applicable` (Section ID -> reason)."""

    def __init__(self, not_applicable: Mapping[str, str] | None = None):
        self.not_applicable = not_applicable or {}
        self.calls: list[tuple[list[Chunk], PermitRequest]] = []

    def validate(self, chunks: list[Chunk], request: PermitRequest) -> SectionRelevance:
        self.calls.append((chunks, request))
        section_id = chunks[0].section_id
        reason = self.not_applicable.get(section_id)
        return SectionRelevance(
            section_id=section_id,
            applies=reason is None,
            reason=reason or f"Aplica: {section_id}",
        )


class FakeNotesLLM:
    def __init__(self, *batches: list[DraftNote]):
        self.batches = list(batches)
        self.prompts: list[str] = []

    def write_notes(self, prompt: str) -> list[DraftNote]:
        self.prompts.append(prompt)
        return self.batches.pop(0)


class FakeSelector:
    """Classifies every clause it is shown: `aplica` unless listed in `categories`."""

    def __init__(
        self,
        categories: Mapping[str, ClauseCategory] | None = None,
        omit: frozenset[str] = frozenset(),
        extra: tuple[ClauseSelection, ...] = (),
    ):
        self.categories = categories or {}
        self.omit = omit
        self.extra = extra
        self.calls: list[tuple[list[Clause], PermitRequest]] = []

    def select(self, clauses: list[Clause], request: PermitRequest) -> list[ClauseSelection]:
        self.calls.append((clauses, request))
        return [
            ClauseSelection(
                ref=clause.ref,
                category=self.categories.get(clause.ref, "aplica"),
                reason=f"razón para {clause.ref}",
            )
            for clause in clauses
            if clause.ref not in self.omit
        ] + list(self.extra)


class FakeRetriever:
    """Semantic retrieval returns `section_ids`; any fixture Section can be fetched by ID."""

    def __init__(self, section_ids: set[str]):
        self.chunks = [c for c in load_fixture_chunks() if c.section_id in section_ids]
        self.requests: list[PermitRequest] = []

    def retrieve(self, request: PermitRequest) -> list[Chunk]:
        self.requests.append(request)
        return self.chunks

    def fetch(self, section_ids: Collection[str]) -> list[Chunk]:
        return [c for c in load_fixture_chunks() if c.section_id in section_ids]


STANDING_NOTE = DraftNote(
    text="La solicitante es dueña y presenta copia de la escritura pública.",
    section_ids=["RC-2023-E:2.1.9.5(b)"],
    curable=False,
)
VALUE_NOTE = DraftNote(
    text="Se incluye la certificación del valor estimado de la obra.",
    section_ids=["RC-2023-E:2.1.9.3"],
    curable=False,
)


def pipeline(
    decisions, notes_llm, retriever, selector=None, fixed=(), criteria=(), validator=None
) -> ReviewPipeline:
    settings = Settings(approval_cutoff=0.5, borderline_low=0.4, borderline_high=0.6)
    return ReviewPipeline(
        decisions,
        notes_llm,
        retriever,
        selector or FakeSelector(),
        validator or FakeSectionValidator(),
        settings,
        fixed=fixed,
        criteria=criteria,
    )


def clause_text(ref: str) -> str:
    """The fixture text of one clause, as an independent check on what reaches the models."""
    section_id, letter = ref.removesuffix(")").split("(")
    for chunk in load_fixture_chunks():
        if chunk.section_id == section_id:
            for line in chunk.text.splitlines():
                if line.startswith(f"{letter}. "):
                    return line
    raise KeyError(ref)


@pytest.fixture
def retriever():
    return FakeRetriever({"RC-2023-E:2.1.9.3", "RC-2023-E:2.1.9.5"})


def test_good_request_gets_approve_recommendation_with_cited_notes(retriever):
    decisions = FakeDecisions(default=0.86)
    notes_llm = FakeNotesLLM([STANDING_NOTE, VALUE_NOTE])

    recommendation = pipeline(decisions, notes_llm, retriever).review(GOOD_PCOC)

    assert recommendation.approval_score == 0.86
    assert recommendation.verdict == "approve"
    assert not recommendation.borderline
    assert [note.text for note in recommendation.notes] == [STANDING_NOTE.text, VALUE_NOTE.text]
    assert recommendation.notes[0].section_ids == ["RC-2023-E:2.1.9.5(b)"]
    assert not any(note.flagged for note in recommendation.notes)
    assert recommendation.retrieved_section_ids == ["RC-2023-E:2.1.9.3", "RC-2023-E:2.1.9.5"]


def test_each_section_is_asked_over_its_own_applicable_clauses_and_the_request(retriever):
    decisions = FakeDecisions(default=0.86)

    pipeline(decisions, FakeNotesLLM([VALUE_NOTE]), retriever).review(GOOD_PCOC)

    assert [list(questions) for _, questions in decisions.calls] == [
        ["seccion_2_1_9_3"],
        ["seccion_2_1_9_5"],
    ]
    input, questions = call_for(decisions, "seccion_2_1_9_5")
    assert questions["seccion_2_1_9_5"] == (
        "¿Cumple esta Solicitud de Permiso de Construcción con los requisitos citados de la "
        "Sección 2.1.9.5 del Reglamento?"
    )
    assert "[RC-2023-E:2.1.9.5]" in input
    assert "Tendrá que incluirse evidencia fehaciente" in input
    assert clause_text("RC-2023-E:2.1.9.3(a)") not in input
    assert "María Isabel Rivera Ortiz" in input
    assert "254-087-123-45" in input


def test_notes_prompt_carries_sections_request_and_score(retriever):
    notes_llm = FakeNotesLLM([VALUE_NOTE])

    pipeline(FakeDecisions(default=0.86), notes_llm, retriever).review(GOOD_PCOC)

    [prompt] = notes_llm.prompts
    assert "[RC-2023-E:2.1.9.3]" in prompt
    assert "Caguas" in prompt
    assert "0.86" in prompt


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("catastro", "254-087"),
        ("applicant_type", "extranjero"),
        ("floors", 0),
        ("request_type", "LICENCIA_DE_PESCA"),
    ],
)
def test_invalid_intake_is_rejected_before_any_model_is_called(retriever, field, value):
    raw = copy.deepcopy(GOOD_PCOC) | {field: value}
    decisions = FakeDecisions(default=0.86)
    notes_llm = FakeNotesLLM([VALUE_NOTE])

    with pytest.raises(InvalidPermitRequest) as error:
        pipeline(decisions, notes_llm, retriever).review(raw)

    assert any(message.startswith(f"{field}:") for message in error.value.errors)
    assert decisions.calls == [] and notes_llm.prompts == [] and retriever.requests == []


def test_missing_required_field_is_invalid_intake(retriever):
    raw = {k: v for k, v in GOOD_PCOC.items() if k != "work_description"}

    with pytest.raises(InvalidPermitRequest, match="work_description"):
        pipeline(FakeDecisions(), FakeNotesLLM(), retriever).review(raw)


def test_note_citing_a_section_that_was_not_retrieved_is_flagged(retriever):
    stray = DraftNote(
        text="El permiso puede prorrogarse hasta dos veces.",
        section_ids=["RC-2023-E:2.1.9.3", "RC-2023-E:2.2.6(c)"],
        curable=False,
    )
    notes_llm = FakeNotesLLM([STANDING_NOTE, stray])

    recommendation = pipeline(FakeDecisions(default=0.86), notes_llm, retriever).review(
        GOOD_PCOC
    )

    standing, flagged = recommendation.notes
    assert not standing.flagged
    assert flagged.flagged
    assert flagged.unretrieved_section_ids == ["RC-2023-E:2.2.6(c)"]


def test_citing_a_sibling_section_with_a_shared_prefix_is_flagged(retriever):
    # 2.1.9.3 was retrieved; 2.1.9.30 is a different Section, not a clause of it.
    sibling = DraftNote(text="...", section_ids=["RC-2023-E:2.1.9.30"], curable=False)

    recommendation = pipeline(
        FakeDecisions(default=0.86), FakeNotesLLM([sibling]), retriever
    ).review(GOOD_PCOC)

    assert recommendation.notes[0].unretrieved_section_ids == ["RC-2023-E:2.1.9.30"]


def test_every_requirement_refused_means_manual_review_but_notes_are_still_written(retriever):
    notes_llm = FakeNotesLLM([VALUE_NOTE])

    recommendation = pipeline(FakeDecisions(default=None), notes_llm, retriever).review(
        GOOD_PCOC
    )

    assert recommendation.approval_score is None
    assert recommendation.verdict == "manual_review"
    assert not recommendation.borderline
    assert len(recommendation.notes) == 1
    assert "sin puntuación" in notes_llm.prompts[0]


def test_one_refused_requirement_means_no_score_and_the_others_are_kept(retriever):
    decisions = FakeDecisions({"seccion_2_1_9_3": None}, default=0.9)

    recommendation = pipeline(decisions, FakeNotesLLM([]), retriever).review(GOOD_PCOC)

    assert recommendation.approval_score is None
    assert recommendation.verdict == "manual_review"
    assert [(c.id, c.probability) for c in recommendation.requirements] == [
        ("seccion_2_1_9_3", None),
        ("seccion_2_1_9_5", 0.9),
    ]


def test_approval_score_is_the_lowest_requirement_probability(pcoc_retriever):
    decisions = FakeDecisions(
        {"pcoc_parametros": 0.62, "seccion_2_1_9_3": 0.55, "pcoc_legitimacion": 0.97},
        default=0.9,
    )

    recommendation = pipeline(
        decisions, FakeNotesLLM([]), pcoc_retriever, criteria=CRITERIA
    ).review(GOOD_PCOC)

    assert recommendation.approval_score == 0.55
    assert recommendation.weakest_requirement is not None
    assert recommendation.weakest_requirement.id == "seccion_2_1_9_3"
    assert recommendation.verdict == "approve"
    assert recommendation.borderline


@pytest.mark.parametrize(
    ("score", "verdict", "borderline"),
    [
        (0.0, "not_approve", False),
        (0.39, "not_approve", False),
        (0.45, "not_approve", True),
        (0.5, "approve", True),
        (0.6, "approve", True),
        (0.61, "approve", False),
    ],
)
def test_verdict_follows_cutoff_and_borderline_band(retriever, score, verdict, borderline):
    recommendation = pipeline(
        FakeDecisions(default=score), FakeNotesLLM([VALUE_NOTE]), retriever
    ).review(GOOD_PCOC)

    assert (recommendation.verdict, recommendation.borderline) == (verdict, borderline)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            {k: v for k, v in GOOD_PCOC.items() if k != "work_description"},
            "work_description: campo obligatorio",
        ),
        (GOOD_PCOC | {"catastro": "254-087"}, "catastro: formato no válido"),
        (GOOD_PCOC | {"applicant_type": "extranjero"}, "applicant_type: valor no permitido"),
        (GOOD_PCOC | {"floors": 0}, "floors: debe ser mayor o igual que 1"),
        (GOOD_PCOC | {"colour": "azul"}, "colour: campo no reconocido"),
        (
            {k: v for k, v in GOOD_PCOC.items() if k != "request_type"},
            "request_type: campo obligatorio",
        ),
        (
            GOOD_PCOC | {"request_type": "LICENCIA_DE_PESCA"},
            "request_type: valor no permitido (se espera 'PCOC', 'PERMISO_UNICO')",
        ),
        (GOOD_PERMISO_UNICO | {"validity_years": 5}, "validity_years: debe ser menor o igual que 3"),
        (GOOD_PCOC | {"request_type": "PERMISO_UNICO"}, "proposed_use: campo obligatorio"),
        (
            GOOD_PERMISO_UNICO | {"renewal": RENEWAL | {"filing_date": "ayer"}},
            "renewal.filing_date:",
        ),
    ],
)
def test_invalid_intake_errors_are_in_spanish(retriever, raw, message):
    with pytest.raises(InvalidPermitRequest) as error:
        pipeline(FakeDecisions(), FakeNotesLLM(), retriever).review(raw)

    assert any(m.startswith(message) for m in error.value.errors), error.value.errors


PROCEDURAL: dict[str, ClauseCategory] = {
    "RC-2023-E:2.1.9.2(a)": "tramite_del_sistema",
    "RC-2023-E:2.1.9.2(n)": "tramite_del_sistema",
    "RC-2023-E:2.1.9.2(l)": "no_aplica",
    "RC-2023-E:2.1.9.5(d)": "no_aplica",
}


@pytest.fixture
def pcoc_retriever():
    return FakeRetriever({"RC-2023-E:2.1.9.2", "RC-2023-E:2.1.9.3", "RC-2023-E:2.1.9.5"})


def test_selector_is_shown_every_retrieved_clause_and_the_request(pcoc_retriever):
    selector = FakeSelector()

    pipeline(FakeDecisions(default=0.7), FakeNotesLLM([]), pcoc_retriever, selector).review(
        GOOD_PCOC
    )

    [(clauses, request)] = selector.calls
    assert len(clauses) == 23 + 3 + 6
    assert clauses[0].ref == "RC-2023-E:2.1.9.2(a)"
    assert clauses[-1].ref == "RC-2023-E:2.1.9.5(f)"
    assert request.applicant_name == "María Isabel Rivera Ortiz"


def test_only_aplica_clauses_reach_decide_word_for_word_under_their_section(pcoc_retriever):
    decisions = FakeDecisions(default=0.7)

    pipeline(decisions, FakeNotesLLM([]), pcoc_retriever, FakeSelector(PROCEDURAL)).review(
        GOOD_PCOC
    )

    input = all_inputs(decisions)
    for ref in PROCEDURAL:
        assert clause_text(ref) not in input
    for ref in ["RC-2023-E:2.1.9.2(k)", "RC-2023-E:2.1.9.2(m)", "RC-2023-E:2.1.9.5(b)"]:
        assert clause_text(ref) in input
    assert "1. Escritura pública.\n2. Contrato de arrendamiento." in input
    # Both parts of 2.1.9.2 sit under one heading in one Requirement, ahead of the request fields.
    filing, _ = call_for(decisions, "seccion_2_1_9_2")
    assert filing.count("[RC-2023-E:2.1.9.2] REQUISITOS DE PRESENTACIÓN") == 1
    assert clause_text("RC-2023-E:2.1.9.2(k)") in filing
    assert filing.index(clause_text("RC-2023-E:2.1.9.2(w)")) < filing.index("SOLICITUD DE PERMISO")


def test_an_applicable_requirement_the_request_omits_stays_in_the_decide_input(pcoc_retriever):
    # No value certification attached and no value declared: 2.1.9.3(a) is a deficiency.
    raw = GOOD_PCOC | {"attached_documents": ["planos_sellados"], "estimated_work_value": None}
    decisions = FakeDecisions(default=0.2)

    pipeline(decisions, FakeNotesLLM([]), pcoc_retriever, FakeSelector(PROCEDURAL)).review(raw)

    input = all_inputs(decisions)
    assert clause_text("RC-2023-E:2.1.9.3(a)") in input


def test_write_notes_receives_the_full_applicable_sections(pcoc_retriever):
    notes_llm = FakeNotesLLM([])

    pipeline(
        FakeDecisions(default=0.7), notes_llm, pcoc_retriever, FakeSelector(PROCEDURAL)
    ).review(GOOD_PCOC)

    [prompt] = notes_llm.prompts
    for ref in PROCEDURAL:
        assert clause_text(ref) in prompt


def test_recommendation_carries_the_selection_with_reasons_in_reglamento_order(pcoc_retriever):
    recommendation = pipeline(
        FakeDecisions(default=0.7),
        FakeNotesLLM([]),
        pcoc_retriever,
        FakeSelector(PROCEDURAL),
    ).review(GOOD_PCOC)

    selection = {s.ref: s for s in recommendation.selection}
    assert len(recommendation.selection) == 32
    assert recommendation.selection[0].ref == "RC-2023-E:2.1.9.2(a)"
    assert selection["RC-2023-E:2.1.9.2(n)"].category == "tramite_del_sistema"
    assert selection["RC-2023-E:2.1.9.2(l)"].reason == "razón para RC-2023-E:2.1.9.2(l)"
    assert selection["RC-2023-E:2.1.9.2(k)"].category == "aplica"
    assert recommendation.set_aside_count == 4


def test_unknown_clause_refs_are_dropped_and_reported(pcoc_retriever):
    invented = ClauseSelection(ref="RC-2023-E:2.1.9.2(z)", category="aplica", reason="inventado")
    selector = FakeSelector(PROCEDURAL, extra=(invented,))

    recommendation = pipeline(
        FakeDecisions(default=0.7), FakeNotesLLM([]), pcoc_retriever, selector
    ).review(GOOD_PCOC)

    assert "RC-2023-E:2.1.9.2(z)" not in {s.ref for s in recommendation.selection}
    assert recommendation.unknown_clause_refs == ["RC-2023-E:2.1.9.2(z)"]


def test_a_clause_the_selector_left_out_is_reported_and_kept_for_decide(pcoc_retriever):
    decisions = FakeDecisions(default=0.7)
    selector = FakeSelector(PROCEDURAL, omit=frozenset({"RC-2023-E:2.1.9.3(b)"}))

    recommendation = pipeline(decisions, FakeNotesLLM([]), pcoc_retriever, selector).review(
        GOOD_PCOC
    )

    assert recommendation.unclassified_clause_refs == ["RC-2023-E:2.1.9.3(b)"]
    input = all_inputs(decisions)
    assert clause_text("RC-2023-E:2.1.9.3(b)") in input


def test_a_clause_classified_twice_keeps_its_first_classification(pcoc_retriever):
    again = ClauseSelection(ref="RC-2023-E:2.1.9.2(a)", category="aplica", reason="otra vez")
    decisions = FakeDecisions(default=0.7)

    recommendation = pipeline(
        decisions, FakeNotesLLM([]), pcoc_retriever, FakeSelector(PROCEDURAL, extra=(again,))
    ).review(GOOD_PCOC)

    assert [s.category for s in recommendation.selection if s.ref == "RC-2023-E:2.1.9.2(a)"] == [
        "tramite_del_sistema"
    ]
    assert clause_text("RC-2023-E:2.1.9.2(a)") not in all_inputs(decisions)


FIXED_COMPLETENESS = FixedCategory(
    ref="RC-2023-E:2.1.9.2(m)",
    category="tramite_del_sistema",
    reason="Fijado: la completitud se juzga por los requisitos específicos.",
)


def test_a_fixed_clause_is_not_shown_to_the_selector_and_keeps_its_configured_category(
    pcoc_retriever,
):
    selector = FakeSelector()
    decisions = FakeDecisions(default=0.7)

    recommendation = pipeline(
        decisions, FakeNotesLLM([]), pcoc_retriever, selector, fixed=[FIXED_COMPLETENESS]
    ).review(GOOD_PCOC)

    [(clauses, _)] = selector.calls
    assert "RC-2023-E:2.1.9.2(m)" not in {clause.ref for clause in clauses}
    selection = {s.ref: s for s in recommendation.selection}
    assert selection["RC-2023-E:2.1.9.2(m)"] == FIXED_COMPLETENESS
    assert [s.ref for s in recommendation.selection][11:14] == [
        "RC-2023-E:2.1.9.2(l)",
        "RC-2023-E:2.1.9.2(m)",
        "RC-2023-E:2.1.9.2(n)",
    ]
    assert recommendation.unclassified_clause_refs == []
    assert clause_text("RC-2023-E:2.1.9.2(m)") not in all_inputs(decisions)


def test_a_fixed_category_wins_over_the_selector(pcoc_retriever):
    stray = ClauseSelection(ref="RC-2023-E:2.1.9.2(m)", category="aplica", reason="del modelo")

    recommendation = pipeline(
        FakeDecisions(default=0.7),
        FakeNotesLLM([]),
        pcoc_retriever,
        FakeSelector(extra=(stray,)),
        fixed=[FIXED_COMPLETENESS],
    ).review(GOOD_PCOC)

    assert [s for s in recommendation.selection if s.ref == "RC-2023-E:2.1.9.2(m)"] == [
        FIXED_COMPLETENESS
    ]
    assert recommendation.unknown_clause_refs == []


def test_a_fixed_category_for_a_clause_that_was_not_retrieved_is_ignored(retriever):
    # `retriever` holds 2.1.9.3 and 2.1.9.5 only, so the 2.1.9.2(m) category has nothing to apply to.
    recommendation = pipeline(
        FakeDecisions(default=0.7),
        FakeNotesLLM([]),
        retriever,
        fixed=[FIXED_COMPLETENESS],
    ).review(GOOD_PCOC)

    assert "RC-2023-E:2.1.9.2(m)" not in {s.ref for s in recommendation.selection}


# Criteria are built here rather than loaded, so these tests do not depend on the curated YAML.
PCOC_STANDING = Criterion(
    id="pcoc_legitimacion",
    request_type="PCOC",
    text_es="El proponente acredita legitimación activa.",
    text_en="The Applicant shows standing.",
    pinned=["RC-2023-E:2.1.9.5(b)"],
    instructions="¿Acredita la solicitud la legitimación activa del proponente?",
)
PCOC_PARAMETERS = Criterion(
    id="pcoc_parametros",
    request_type="PCOC",
    text_es="La obra cumple los parámetros del distrito.",
    text_en="The work meets the district parameters.",
    pinned=["RC-2023-E:2.2.1.3(d)"],
    instructions="¿Cumple la obra cada parámetro de la Sección 2.2.1.3(d)?",
)
PU_USE = Criterion(
    id="pu_uso",
    request_type="PERMISO_UNICO",
    text_es="El uso está permitido ministerialmente.",
    text_en="The use is allowed as of right.",
    pinned=["RC-2023-E:3.7.1.1(f)"],
    application_kinds=["nueva"],
    instructions="¿Es el uso propuesto un uso ministerial?",
)
PU_RENEWAL_DEADLINE = Criterion(
    id="pu_renovacion_plazo",
    request_type="PERMISO_UNICO",
    text_es="La renovación se presentó a tiempo.",
    text_en="The renewal was filed in time.",
    pinned=["RC-2023-E:3.7.1.4(f)", "RC-2023-E:3.7.1.4(g)"],
    application_kinds=["renovacion"],
    instructions="¿Se presentó la renovación dentro del plazo?",
)
CRITERIA = [PCOC_STANDING, PCOC_PARAMETERS, PU_USE, PU_RENEWAL_DEADLINE]


def criterion_probabilities(**probabilities: float | None) -> FakeDecisions:
    return FakeDecisions(probabilities, default=0.7)


def test_pinned_sections_are_retrieved_even_when_the_retriever_misses_them():
    decisions = criterion_probabilities(pcoc_legitimacion=0.9, pcoc_parametros=0.8)

    recommendation = pipeline(
        decisions,
        FakeNotesLLM([]),
        FakeRetriever({"RC-2023-E:2.1.9.3"}),
        criteria=CRITERIA,
    ).review(GOOD_PCOC)

    assert recommendation.retrieved_section_ids == [
        "RC-2023-E:2.1.9.3",
        "RC-2023-E:2.1.9.5",
        "RC-2023-E:2.2.1.3",
    ]
    scored = all_inputs(decisions)
    assert clause_text("RC-2023-E:2.1.9.5(b)") in scored


def test_a_pinned_section_the_retriever_found_is_not_repeated(pcoc_retriever):
    recommendation = pipeline(
        criterion_probabilities(pcoc_legitimacion=0.9, pcoc_parametros=0.8),
        FakeNotesLLM([]),
        pcoc_retriever,
        criteria=CRITERIA,
    ).review(GOOD_PCOC)

    assert recommendation.retrieved_section_ids == [
        "RC-2023-E:2.1.9.2",
        "RC-2023-E:2.1.9.3",
        "RC-2023-E:2.1.9.5",
        "RC-2023-E:2.2.1.3",
    ]
    assert [s.ref for s in recommendation.selection].count("RC-2023-E:2.1.9.5(b)") == 1


def test_pinned_clauses_reach_decide_even_when_the_selector_would_set_them_aside(pcoc_retriever):
    selector = FakeSelector(
        {"RC-2023-E:2.1.9.5(b)": "no_aplica", "RC-2023-E:2.2.1.3(d)": "tramite_del_sistema"}
    )
    decisions = criterion_probabilities(pcoc_legitimacion=0.9, pcoc_parametros=0.8)

    recommendation = pipeline(
        decisions, FakeNotesLLM([]), pcoc_retriever, selector, criteria=CRITERIA
    ).review(GOOD_PCOC)

    scored = all_inputs(decisions)
    assert clause_text("RC-2023-E:2.1.9.5(b)") in scored
    assert clause_text("RC-2023-E:2.2.1.3(d)") in scored
    selection = {s.ref: s for s in recommendation.selection}
    assert selection["RC-2023-E:2.1.9.5(b)"].category == "aplica"
    assert "pcoc_legitimacion" in selection["RC-2023-E:2.1.9.5(b)"].reason
    [(shown, _)] = selector.calls
    assert "RC-2023-E:2.1.9.5(b)" not in {clause.ref for clause in shown}


def test_a_pinned_clause_wins_over_a_fixed_category(pcoc_retriever):
    fixed = FixedCategory(
        ref="RC-2023-E:2.2.1.3(d)", category="tramite_del_sistema", reason="Fijado."
    )
    decisions = criterion_probabilities(pcoc_legitimacion=0.9, pcoc_parametros=0.8)

    recommendation = pipeline(
        decisions, FakeNotesLLM([]), pcoc_retriever, fixed=[fixed], criteria=CRITERIA
    ).review(GOOD_PCOC)

    scored = all_inputs(decisions)
    assert clause_text("RC-2023-E:2.2.1.3(d)") in scored
    [parameters] = [s for s in recommendation.selection if s.ref == "RC-2023-E:2.2.1.3(d)"]
    assert parameters.category == "aplica"


def test_each_criterion_is_asked_over_its_own_pinned_clauses_and_the_request(pcoc_retriever):
    decisions = criterion_probabilities(pcoc_legitimacion=0.9, pcoc_parametros=0.8)

    pipeline(decisions, FakeNotesLLM([]), pcoc_retriever, criteria=CRITERIA).review(GOOD_PCOC)

    input, questions = call_for(decisions, "pcoc_legitimacion")
    assert questions == {"pcoc_legitimacion": PCOC_STANDING.instructions}
    assert "[RC-2023-E:2.1.9.5] LEGITIMACIÓN ACTIVA (STANDING)" in input
    assert clause_text("RC-2023-E:2.1.9.5(b)") in input
    assert "1. Escritura pública." in input
    assert clause_text("RC-2023-E:2.1.9.5(a)") not in input
    assert clause_text("RC-2023-E:2.2.1.3(d)") not in input
    assert "María Isabel Rivera Ortiz" in input
    # The rest of 2.1.9.5 is its Section's own Requirement; the pinned clause is the Criterion's.
    standing_section, _ = call_for(decisions, "seccion_2_1_9_5")
    assert clause_text("RC-2023-E:2.1.9.5(a)") in standing_section
    assert clause_text("RC-2023-E:2.1.9.5(b)") not in standing_section


def test_pcoc_criterion_probabilities_are_carried_into_the_recommendation(pcoc_retriever):
    recommendation = pipeline(
        criterion_probabilities(pcoc_legitimacion=0.91, pcoc_parametros=None),
        FakeNotesLLM([]),
        pcoc_retriever,
        criteria=CRITERIA,
    ).review(GOOD_PCOC)

    assert [(a.criterion, a.probability) for a in recommendation.criterion_requirements] == [
        (PCOC_STANDING, 0.91),
        (PCOC_PARAMETERS, None),
    ]


def test_new_permiso_unico_is_assessed_on_its_new_request_criteria():
    decisions = criterion_probabilities(pu_uso=0.95)

    recommendation = pipeline(
        decisions,
        FakeNotesLLM([]),
        FakeRetriever({"RC-2023-E:3.7.1.2"}),
        criteria=CRITERIA,
    ).review(GOOD_PERMISO_UNICO)

    assert [(a.id, a.probability) for a in recommendation.criterion_requirements] == [
        ("pu_uso", 0.95)
    ]
    assert recommendation.retrieved_section_ids == ["RC-2023-E:3.7.1.2", "RC-2023-E:3.7.1.1"]
    assert {name for _, questions in decisions.calls for name in questions} == {
        "pu_uso",
        "seccion_3_7_1_1",
        "seccion_3_7_1_2",
    }
    input, _ = call_for(decisions, "pu_uso")
    assert clause_text("RC-2023-E:3.7.1.1(f)") in input
    assert "Cafetería con servicio de mesas" in input


def test_permiso_unico_renewal_is_assessed_on_its_renewal_criteria():
    decisions = criterion_probabilities(pu_renovacion_plazo=0.02)

    recommendation = pipeline(
        decisions,
        FakeNotesLLM([]),
        FakeRetriever({"RC-2023-E:3.7.1.2"}),
        criteria=CRITERIA,
    ).review(GOOD_PERMISO_UNICO | {"renewal": RENEWAL})

    assert [(a.id, a.probability) for a in recommendation.criterion_requirements] == [
        ("pu_renovacion_plazo", 0.02)
    ]
    input, _ = call_for(decisions, "pu_renovacion_plazo")
    assert clause_text("RC-2023-E:3.7.1.4(f)") in input
    assert clause_text("RC-2023-E:3.7.1.4(g)") in input
    assert "1020 días (2.8 años) después de la expiración" in input


def test_section_questions_name_the_request_type(retriever):
    decisions = FakeDecisions(default=0.7)

    pipeline(decisions, FakeNotesLLM([]), retriever).review(GOOD_PERMISO_UNICO)

    _, questions = call_for(decisions, "seccion_2_1_9_5")
    assert questions["seccion_2_1_9_5"] == (
        "¿Cumple esta Solicitud de Permiso Único con los requisitos citados de la Sección "
        "2.1.9.5 del Reglamento?"
    )


def test_notes_receive_every_requirement_with_its_probability(pcoc_retriever):
    notes_llm = FakeNotesLLM([])

    pipeline(
        criterion_probabilities(pcoc_legitimacion=0.91, pcoc_parametros=None),
        notes_llm,
        pcoc_retriever,
        criteria=CRITERIA,
    ).review(GOOD_PCOC)

    [prompt] = notes_llm.prompts
    assert (
        "- [pcoc_legitimacion] El proponente acredita legitimación activa. "
        "(RC-2023-E:2.1.9.5(b)): 0.91"
    ) in prompt
    assert (
        "- [pcoc_parametros] La obra cumple los parámetros del distrito. "
        "(RC-2023-E:2.2.1.3(d)): no evaluado (pregunta rechazada)"
    ) in prompt
    assert "- [seccion_2_1_9_3] Requisitos aplicables de la Sección 2.1.9.3" in prompt


def test_a_pin_that_matches_no_clause_is_a_configuration_error(retriever):
    typo = PCOC_STANDING.model_copy(update={"pinned": ["RC-2023-E:2.1.9.5(z)"]})

    with pytest.raises(MissingPinnedClause, match=r"pcoc_legitimacion.*2\.1\.9\.5\(z\)"):
        pipeline(
            criterion_probabilities(pcoc_legitimacion=0.9),
            FakeNotesLLM([]),
            retriever,
            criteria=[typo],
        ).review(GOOD_PCOC)


def test_a_permiso_unico_pinned_clause_reaches_decide_even_when_the_selector_would_set_it_aside():
    decisions = criterion_probabilities(pu_uso=0.95)

    recommendation = pipeline(
        decisions,
        FakeNotesLLM([]),
        FakeRetriever({"RC-2023-E:3.7.1.1"}),
        FakeSelector({"RC-2023-E:3.7.1.1(f)": "no_aplica"}),
        criteria=CRITERIA,
    ).review(GOOD_PERMISO_UNICO)

    scored = all_inputs(decisions)
    assert clause_text("RC-2023-E:3.7.1.1(f)") in scored
    [use] = [s for s in recommendation.selection if s.ref == "RC-2023-E:3.7.1.1(f)"]
    assert use.category == "aplica" and "pu_uso" in use.reason


def test_each_retrieved_section_is_validated_once_with_all_its_parts_and_the_request(
    pcoc_retriever,
):
    validator = FakeSectionValidator()

    pipeline(
        criterion_probabilities(),
        FakeNotesLLM([]),
        pcoc_retriever,
        criteria=[PCOC_STANDING],
        validator=validator,
    ).review(GOOD_PCOC)

    asked = [[(c.section_id, c.part) for c in chunks] for chunks, _ in validator.calls]
    # 2.1.9.5 is pinned by a Criterion, so it applies without asking.
    assert sorted(asked) == [
        [("RC-2023-E:2.1.9.2", "1-of-2"), ("RC-2023-E:2.1.9.2", "2-of-2")],
        [("RC-2023-E:2.1.9.3", None)],
    ]
    assert {request.applicant_name for _, request in validator.calls} == {
        "María Isabel Rivera Ortiz"
    }


def test_a_section_that_does_not_apply_is_neither_selected_from_scored_nor_shown_to_notes():
    retriever = FakeRetriever({"RC-2023-E:2.1.9.3", "RC-2023-E:2.1.9.5", "RC-2023-E:2.2.6"})
    validator = FakeSectionValidator({"RC-2023-E:2.2.6": "No se solicita una prórroga."})
    selector = FakeSelector()
    decisions = FakeDecisions(default=0.9)
    notes_llm = FakeNotesLLM([])

    recommendation = pipeline(
        decisions, notes_llm, retriever, selector, validator=validator
    ).review(GOOD_PCOC)

    [(shown, _)] = selector.calls
    assert not any(clause.section_id == "RC-2023-E:2.2.6" for clause in shown)
    assert clause_text("RC-2023-E:2.2.6(c)") not in all_inputs(decisions)
    assert "seccion_2_2_6" not in {name for _, questions in decisions.calls for name in questions}
    assert clause_text("RC-2023-E:2.2.6(c)") not in notes_llm.prompts[0]
    assert "RC-2023-E:2.2.6" in recommendation.retrieved_section_ids
    relevance = {r.section_id: r for r in recommendation.section_relevance}
    assert relevance["RC-2023-E:2.2.6"].applies is False
    assert relevance["RC-2023-E:2.2.6"].reason == "No se solicita una prórroga."
    assert relevance["RC-2023-E:2.1.9.3"].applies is True


def test_a_section_pinned_by_a_criterion_always_applies(pcoc_retriever):
    validator = FakeSectionValidator({"RC-2023-E:2.1.9.5": "No aplica."})

    recommendation = pipeline(
        criterion_probabilities(),
        FakeNotesLLM([]),
        pcoc_retriever,
        criteria=[PCOC_STANDING],
        validator=validator,
    ).review(GOOD_PCOC)

    relevance = {r.section_id: r for r in recommendation.section_relevance}
    assert relevance["RC-2023-E:2.1.9.5"].applies is True
    assert "pcoc_legitimacion" in relevance["RC-2023-E:2.1.9.5"].reason


def test_a_note_citing_a_section_that_does_not_apply_is_flagged():
    retriever = FakeRetriever({"RC-2023-E:2.1.9.3", "RC-2023-E:2.2.6"})
    validator = FakeSectionValidator({"RC-2023-E:2.2.6": "No se solicita una prórroga."})
    extension = DraftNote(text="...", section_ids=["RC-2023-E:2.2.6(c)"], curable=False)

    recommendation = pipeline(
        FakeDecisions(), FakeNotesLLM([extension]), retriever, validator=validator
    ).review(GOOD_PCOC)

    assert recommendation.notes[0].unretrieved_section_ids == ["RC-2023-E:2.2.6(c)"]


def test_a_section_whose_clauses_are_all_set_aside_is_no_requirement(pcoc_retriever):
    every_value_clause: dict[str, ClauseCategory] = {
        f"RC-2023-E:2.1.9.3({letter})": "no_aplica" for letter in "abc"
    }
    decisions = FakeDecisions()

    recommendation = pipeline(
        decisions, FakeNotesLLM([]), pcoc_retriever, FakeSelector(every_value_clause)
    ).review(GOOD_PCOC)

    assert [c.id for c in recommendation.requirements] == ["seccion_2_1_9_2", "seccion_2_1_9_5"]


def test_a_request_with_no_applicable_requirement_gets_no_score(retriever):
    validator = FakeSectionValidator(
        {"RC-2023-E:2.1.9.3": "No aplica.", "RC-2023-E:2.1.9.5": "No aplica."}
    )

    recommendation = pipeline(
        FakeDecisions(), FakeNotesLLM([]), retriever, validator=validator
    ).review(GOOD_PCOC)

    assert recommendation.requirements == []
    assert recommendation.approval_score is None
    assert recommendation.verdict == "manual_review"


def test_a_fixed_category_for_one_request_type_does_not_apply_to_the_other(pcoc_retriever):
    plans_for_permiso_unico = FixedCategory(
        ref="RC-2023-E:2.1.9.2(k)",
        category="no_aplica",
        reason="Fijado: un Permiso Único no radica planos de construcción.",
        request_types=["PERMISO_UNICO"],
    )
    reviews = {}
    for raw in (GOOD_PCOC, GOOD_PERMISO_UNICO):
        selector = FakeSelector()
        recommendation = pipeline(
            FakeDecisions(),
            FakeNotesLLM([]),
            pcoc_retriever,
            selector,
            fixed=[plans_for_permiso_unico],
        ).review(raw)
        [(shown, _)] = selector.calls
        selection = {s.ref: s for s in recommendation.selection}
        reviews[raw["request_type"]] = (
            "RC-2023-E:2.1.9.2(k)" in {clause.ref for clause in shown},
            selection["RC-2023-E:2.1.9.2(k)"].category,
        )

    assert reviews == {"PCOC": (True, "aplica"), "PERMISO_UNICO": (False, "no_aplica")}
