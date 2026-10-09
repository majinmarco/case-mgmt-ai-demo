import copy
from collections.abc import Mapping

import pytest

from case_mgmt_ai_demo.chunks import Chunk, load_fixture_chunks
from case_mgmt_ai_demo.config import Settings
from case_mgmt_ai_demo.permit_request import PermitRequest
from case_mgmt_ai_demo.review import DraftNote, InvalidPermitRequest, ReviewPipeline

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


def pipeline(decisions, notes_llm, retriever) -> ReviewPipeline:
    settings = Settings(approval_cutoff=0.5, borderline_low=0.4, borderline_high=0.6)
    return ReviewPipeline(decisions, notes_llm, retriever, settings)


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
