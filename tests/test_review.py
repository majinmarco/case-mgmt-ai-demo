import copy
from collections.abc import Mapping

import pytest

from case_mgmt_ai_demo.chunks import Chunk, Clause, load_fixture_chunks, split_clauses
from case_mgmt_ai_demo.config import Settings
from case_mgmt_ai_demo.permit_request import PermitRequest
from case_mgmt_ai_demo.review import (
    ClauseCategory,
    ClauseSelection,
    DraftNote,
    InvalidPermitRequest,
    ReviewPipeline,
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


class FakeDecisions:
    def __init__(self, probabilities: Mapping[str, float | None]):
        self.probabilities = probabilities
        self.calls: list[tuple[str, dict[str, str]]] = []

    def predicates(self, input: str, questions: Mapping[str, str]) -> dict[str, float | None]:
        self.calls.append((input, dict(questions)))
        return {name: self.probabilities[name] for name in questions}


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
    def __init__(self, section_ids: set[str]):
        self.chunks = [c for c in load_fixture_chunks() if c.section_id in section_ids]
        self.requests: list[PermitRequest] = []

    def retrieve(self, request: PermitRequest) -> list[Chunk]:
        self.requests.append(request)
        return self.chunks


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


def pipeline(decisions, notes_llm, retriever, selector=None, pinned=()) -> ReviewPipeline:
    settings = Settings(approval_cutoff=0.5, borderline_low=0.4, borderline_high=0.6)
    return ReviewPipeline(
        decisions, notes_llm, retriever, selector or FakeSelector(), settings, pinned=pinned
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
    decisions = FakeDecisions({"aprobacion": 0.86})
    notes_llm = FakeNotesLLM([STANDING_NOTE, VALUE_NOTE])

    recommendation = pipeline(decisions, notes_llm, retriever).review(GOOD_PCOC)

    assert recommendation.approval_score == 0.86
    assert recommendation.verdict == "approve"
    assert not recommendation.borderline
    assert [note.text for note in recommendation.notes] == [STANDING_NOTE.text, VALUE_NOTE.text]
    assert recommendation.notes[0].section_ids == ["RC-2023-E:2.1.9.5(b)"]
    assert not any(note.flagged for note in recommendation.notes)
    assert recommendation.retrieved_section_ids == ["RC-2023-E:2.1.9.3", "RC-2023-E:2.1.9.5"]


def test_decisions_api_is_asked_only_the_overall_predicate_over_sections_and_request(retriever):
    decisions = FakeDecisions({"aprobacion": 0.86})

    pipeline(decisions, FakeNotesLLM([VALUE_NOTE]), retriever).review(GOOD_PCOC)

    [(input, questions)] = decisions.calls
    assert list(questions) == ["aprobacion"]
    assert "[RC-2023-E:2.1.9.5]" in input
    assert "Tendrá que incluirse evidencia fehaciente" in input
    assert "María Isabel Rivera Ortiz" in input
    assert "254-087-123-45" in input


def test_notes_prompt_carries_sections_request_and_score(retriever):
    notes_llm = FakeNotesLLM([VALUE_NOTE])

    pipeline(FakeDecisions({"aprobacion": 0.86}), notes_llm, retriever).review(GOOD_PCOC)

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
    decisions = FakeDecisions({"aprobacion": 0.86})
    notes_llm = FakeNotesLLM([VALUE_NOTE])

    with pytest.raises(InvalidPermitRequest) as error:
        pipeline(decisions, notes_llm, retriever).review(raw)

    assert any(message.startswith(f"{field}:") for message in error.value.errors)
    assert decisions.calls == [] and notes_llm.prompts == [] and retriever.requests == []


def test_missing_required_field_is_invalid_intake(retriever):
    raw = {k: v for k, v in GOOD_PCOC.items() if k != "work_description"}

    with pytest.raises(InvalidPermitRequest, match="work_description"):
        pipeline(FakeDecisions({}), FakeNotesLLM(), retriever).review(raw)


def test_note_citing_a_section_that_was_not_retrieved_is_flagged(retriever):
    stray = DraftNote(
        text="El permiso puede prorrogarse hasta dos veces.",
        section_ids=["RC-2023-E:2.1.9.3", "RC-2023-E:2.2.6(c)"],
        curable=False,
    )
    notes_llm = FakeNotesLLM([STANDING_NOTE, stray])

    recommendation = pipeline(FakeDecisions({"aprobacion": 0.86}), notes_llm, retriever).review(
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
        FakeDecisions({"aprobacion": 0.86}), FakeNotesLLM([sibling]), retriever
    ).review(GOOD_PCOC)

    assert recommendation.notes[0].unretrieved_section_ids == ["RC-2023-E:2.1.9.30"]


def test_refused_approval_predicate_means_manual_review_but_notes_are_still_written(retriever):
    notes_llm = FakeNotesLLM([VALUE_NOTE])

    recommendation = pipeline(FakeDecisions({"aprobacion": None}), notes_llm, retriever).review(
        GOOD_PCOC
    )

    assert recommendation.approval_score is None
    assert recommendation.verdict == "manual_review"
    assert not recommendation.borderline
    assert len(recommendation.notes) == 1
    assert "sin puntuación" in notes_llm.prompts[0]


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
        FakeDecisions({"aprobacion": score}), FakeNotesLLM([VALUE_NOTE]), retriever
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
    ],
)
def test_invalid_intake_errors_are_in_spanish(retriever, raw, message):
    with pytest.raises(InvalidPermitRequest) as error:
        pipeline(FakeDecisions({}), FakeNotesLLM(), retriever).review(raw)

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

    pipeline(FakeDecisions({"aprobacion": 0.7}), FakeNotesLLM([]), pcoc_retriever, selector).review(
        GOOD_PCOC
    )

    [(clauses, request)] = selector.calls
    assert len(clauses) == 23 + 3 + 6
    assert clauses[0].ref == "RC-2023-E:2.1.9.2(a)"
    assert clauses[-1].ref == "RC-2023-E:2.1.9.5(f)"
    assert request.applicant_name == "María Isabel Rivera Ortiz"


def test_only_aplica_clauses_reach_decide_word_for_word_under_their_section(pcoc_retriever):
    decisions = FakeDecisions({"aprobacion": 0.7})

    pipeline(decisions, FakeNotesLLM([]), pcoc_retriever, FakeSelector(PROCEDURAL)).review(
        GOOD_PCOC
    )

    [(input, _)] = decisions.calls
    for ref in PROCEDURAL:
        assert clause_text(ref) not in input
    for ref in ["RC-2023-E:2.1.9.2(k)", "RC-2023-E:2.1.9.2(m)", "RC-2023-E:2.1.9.5(b)"]:
        assert clause_text(ref) in input
    assert "1. Escritura pública.\n2. Contrato de arrendamiento." in input
    # Both parts of 2.1.9.2 sit under one heading, ahead of the request fields.
    assert input.count("[RC-2023-E:2.1.9.2] REQUISITOS DE PRESENTACIÓN") == 1
    assert input.index(clause_text("RC-2023-E:2.1.9.2(w)")) < input.index("[RC-2023-E:2.1.9.3]")
    assert input.index("[RC-2023-E:2.1.9.5]") < input.index("SOLICITUD DE PERMISO")


def test_an_applicable_requirement_the_request_omits_stays_in_the_decide_input(pcoc_retriever):
    # No value certification attached and no value declared: 2.1.9.3(a) is a deficiency.
    raw = GOOD_PCOC | {"attached_documents": ["planos_sellados"], "estimated_work_value": None}
    decisions = FakeDecisions({"aprobacion": 0.2})

    pipeline(decisions, FakeNotesLLM([]), pcoc_retriever, FakeSelector(PROCEDURAL)).review(raw)

    [(input, _)] = decisions.calls
    assert clause_text("RC-2023-E:2.1.9.3(a)") in input


def test_write_notes_still_receives_the_full_retrieved_sections(pcoc_retriever):
    notes_llm = FakeNotesLLM([])

    pipeline(
        FakeDecisions({"aprobacion": 0.7}), notes_llm, pcoc_retriever, FakeSelector(PROCEDURAL)
    ).review(GOOD_PCOC)

    [prompt] = notes_llm.prompts
    for ref in PROCEDURAL:
        assert clause_text(ref) in prompt


def test_recommendation_carries_the_selection_with_reasons_in_reglamento_order(pcoc_retriever):
    recommendation = pipeline(
        FakeDecisions({"aprobacion": 0.7}),
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
        FakeDecisions({"aprobacion": 0.7}), FakeNotesLLM([]), pcoc_retriever, selector
    ).review(GOOD_PCOC)

    assert "RC-2023-E:2.1.9.2(z)" not in {s.ref for s in recommendation.selection}
    assert recommendation.unknown_clause_refs == ["RC-2023-E:2.1.9.2(z)"]


def test_a_clause_the_selector_left_out_is_reported_and_kept_for_decide(pcoc_retriever):
    decisions = FakeDecisions({"aprobacion": 0.7})
    selector = FakeSelector(PROCEDURAL, omit=frozenset({"RC-2023-E:2.1.9.3(b)"}))

    recommendation = pipeline(decisions, FakeNotesLLM([]), pcoc_retriever, selector).review(
        GOOD_PCOC
    )

    assert recommendation.unclassified_clause_refs == ["RC-2023-E:2.1.9.3(b)"]
    [(input, _)] = decisions.calls
    assert clause_text("RC-2023-E:2.1.9.3(b)") in input


def test_a_clause_classified_twice_keeps_its_first_classification(pcoc_retriever):
    again = ClauseSelection(ref="RC-2023-E:2.1.9.2(a)", category="aplica", reason="otra vez")
    decisions = FakeDecisions({"aprobacion": 0.7})

    recommendation = pipeline(
        decisions, FakeNotesLLM([]), pcoc_retriever, FakeSelector(PROCEDURAL, extra=(again,))
    ).review(GOOD_PCOC)

    assert [s.category for s in recommendation.selection if s.ref == "RC-2023-E:2.1.9.2(a)"] == [
        "tramite_del_sistema"
    ]
    assert clause_text("RC-2023-E:2.1.9.2(a)") not in decisions.calls[0][0]


PINNED_COMPLETENESS = ClauseSelection(
    ref="RC-2023-E:2.1.9.2(m)",
    category="tramite_del_sistema",
    reason="Fijado: la completitud se juzga por los requisitos específicos.",
)


def test_a_pinned_clause_is_not_shown_to_the_selector_and_keeps_its_configured_category(
    pcoc_retriever,
):
    selector = FakeSelector()
    decisions = FakeDecisions({"aprobacion": 0.7})

    recommendation = pipeline(
        decisions, FakeNotesLLM([]), pcoc_retriever, selector, pinned=[PINNED_COMPLETENESS]
    ).review(GOOD_PCOC)

    [(clauses, _)] = selector.calls
    assert "RC-2023-E:2.1.9.2(m)" not in {clause.ref for clause in clauses}
    selection = {s.ref: s for s in recommendation.selection}
    assert selection["RC-2023-E:2.1.9.2(m)"] == PINNED_COMPLETENESS
    assert [s.ref for s in recommendation.selection][11:14] == [
        "RC-2023-E:2.1.9.2(l)",
        "RC-2023-E:2.1.9.2(m)",
        "RC-2023-E:2.1.9.2(n)",
    ]
    assert recommendation.unclassified_clause_refs == []
    assert clause_text("RC-2023-E:2.1.9.2(m)") not in decisions.calls[0][0]


def test_a_pinned_category_wins_over_the_selector(pcoc_retriever):
    stray = ClauseSelection(ref="RC-2023-E:2.1.9.2(m)", category="aplica", reason="del modelo")

    recommendation = pipeline(
        FakeDecisions({"aprobacion": 0.7}),
        FakeNotesLLM([]),
        pcoc_retriever,
        FakeSelector(extra=(stray,)),
        pinned=[PINNED_COMPLETENESS],
    ).review(GOOD_PCOC)

    assert [s for s in recommendation.selection if s.ref == "RC-2023-E:2.1.9.2(m)"] == [
        PINNED_COMPLETENESS
    ]
    assert recommendation.unknown_clause_refs == []


def test_a_pin_for_a_clause_that_was_not_retrieved_is_ignored(retriever):
    # `retriever` holds 2.1.9.3 and 2.1.9.5 only, so the 2.1.9.2(m) pin has nothing to apply to.
    recommendation = pipeline(
        FakeDecisions({"aprobacion": 0.7}),
        FakeNotesLLM([]),
        retriever,
        pinned=[PINNED_COMPLETENESS],
    ).review(GOOD_PCOC)

    assert "RC-2023-E:2.1.9.2(m)" not in {s.ref for s in recommendation.selection}
