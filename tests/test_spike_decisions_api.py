import pytest
from openai import OpenAI

from case_mgmt_ai_demo.config import load_settings
from case_mgmt_ai_demo.spikes.decisions_api import decide


@pytest.mark.real_model
def test_good_pcoc_request_scores_clearly_above_bad_one_in_spanish():
    model = load_settings().decisions_api_model  # exports .env, so before OpenAI()
    client = OpenAI()

    good = decide(client, model, "buena", "es").probabilities()["aprobacion"]
    bad = decide(client, model, "mala", "es").probabilities()["aprobacion"]

    assert good is not None and bad is not None, "overall predicate was refused"
    assert good - bad >= 0.5
