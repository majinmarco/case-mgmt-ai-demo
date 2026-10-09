import json
from pathlib import Path

import pytest
from openai import OpenAI

from case_mgmt_ai_demo.adapters import build_review_pipeline
from case_mgmt_ai_demo.config import load_settings

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.mark.real_model
def test_good_pcoc_request_gets_scored_recommendation_with_cited_notes_from_real_models():
    settings = load_settings()  # exports .env, so before OpenAI()
    pipeline = build_review_pipeline(settings, OpenAI())
    raw = json.loads((EXAMPLES / "pcoc-buena.json").read_text(encoding="utf-8"))

    recommendation = pipeline.review(raw)

    assert recommendation.approval_score is not None, "overall predicate was refused"
    assert recommendation.notes, "no Notes were written"
    assert all(note.section_ids for note in recommendation.notes), "a Note cites no Section"
    assert not any(note.flagged for note in recommendation.notes), [
        note.unretrieved_section_ids for note in recommendation.notes
    ]


REPEATS = 3


@pytest.fixture(scope="module")
def real_pipeline():
    settings = load_settings()  # exports .env, so before OpenAI()
    return settings, build_review_pipeline(settings, OpenAI())


def example(name: str) -> dict:
    return json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.real_model
@pytest.mark.parametrize("run", range(REPEATS))
def test_good_pcoc_request_scores_at_or_above_the_cutoff_with_requirement_selection(
    real_pipeline, run
):
    settings, pipeline = real_pipeline

    recommendation = pipeline.review(example("pcoc-buena"))

    score = recommendation.approval_score
    assert score is not None and score >= settings.approval_cutoff, (
        score,
        [s.ref for s in recommendation.selection if s.category == "aplica"],
    )


@pytest.mark.real_model
@pytest.mark.parametrize("run", range(REPEATS))
def test_bad_pcoc_request_stays_below_0_1_with_requirement_selection(real_pipeline, run):
    _, pipeline = real_pipeline

    recommendation = pipeline.review(example("pcoc-mala"))

    assert recommendation.approval_score is not None and recommendation.approval_score < 0.1
