import json
from pathlib import Path

import pytest
from openai import OpenAI

from case_mgmt_ai_demo.adapters import build_review_pipeline
from case_mgmt_ai_demo.config import load_settings

EXAMPLES = Path(__file__).parent.parent / "examples"


def fixture_pipeline():
    """Over the fixture Sections, which these scores were established on (CMAWD-19); the real
    index brings Sections the example requests have no fields to evidence."""
    settings = load_settings().model_copy(update={"retriever": "fixture"})  # exports .env
    return settings, build_review_pipeline(settings, OpenAI())


@pytest.mark.real_model
def test_good_pcoc_request_gets_scored_recommendation_with_cited_notes_from_real_models():
    _, pipeline = fixture_pipeline()
    raw = json.loads((EXAMPLES / "pcoc-buena.json").read_text(encoding="utf-8"))

    recommendation = pipeline.review(raw)

    assert recommendation.approval_score is not None, "overall predicate was refused"
    assert recommendation.notes, "no Notes were written"
    assert not recommendation.dropped_notes, [
        note.unretrieved_section_ids for note in recommendation.dropped_notes
    ]


REPEATS = 3


@pytest.fixture(scope="module")
def real_pipeline():
    return fixture_pipeline()


def example(name: str) -> dict:
    return json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.real_model
@pytest.mark.parametrize("run", range(REPEATS))
@pytest.mark.parametrize("name", ["pcoc-buena", "permiso-unico-buena"])
def test_good_request_scores_at_or_above_the_cutoff(real_pipeline, name, run):
    settings, pipeline = real_pipeline

    recommendation = pipeline.review(example(name))

    score = recommendation.approval_score
    assert score is not None and score >= settings.approval_cutoff, [
        (r.id, r.probability, r.clause_refs) for r in recommendation.requirements
    ]


@pytest.mark.real_model
@pytest.mark.parametrize("run", range(REPEATS))
@pytest.mark.parametrize("name", ["pcoc-mala", "permiso-unico-mala"])
def test_bad_request_stays_below_0_1(real_pipeline, name, run):
    _, pipeline = real_pipeline

    recommendation = pipeline.review(example(name))

    assert recommendation.approval_score is not None and recommendation.approval_score < 0.1


@pytest.mark.real_model
def test_a_review_against_the_real_index_traces_what_was_retrieved_and_how_near():
    """Needs the index `ingest` builds (data/chroma)."""
    settings = load_settings().model_copy(update={"retriever": "index"})  # exports .env
    pipeline = build_review_pipeline(settings, OpenAI())

    recommendation = pipeline.review(example("pcoc-buena"))

    retrieved = recommendation.trace.retrieved
    scored = [s.similarity for s in retrieved if s.similarity is not None]
    assert 0 < len(scored) <= settings.retrieval_top_k
    assert scored == sorted(scored, reverse=True)
    # Every Section a PCOC Criterion pins is there, whether or not retrieval found it.
    pinned = {s.section_id for s in retrieved if s.pinned_by}
    assert pinned == {
        "RC-2023-E:2.1.9.2",
        "RC-2023-E:2.1.9.3",
        "RC-2023-E:2.1.9.5",
        "RC-2023-E:2.2.1.3",
    }
    assert {r.section_id for r in recommendation.section_relevance} == {
        s.section_id for s in retrieved
    }
    assert not [e for e in recommendation.errors if e.step == "retrieve"]
