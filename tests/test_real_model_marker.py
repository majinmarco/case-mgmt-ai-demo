from pathlib import Path

CONFTEST = (Path(__file__).parent / "conftest.py").read_text()

REAL_MODEL_TEST = """
import pytest

@pytest.mark.real_model
def test_calls_openai():
    pass

def test_offline():
    pass
"""


def test_real_model_tests_are_skipped_by_default(pytester, monkeypatch):
    monkeypatch.delenv("RUN_REAL_MODEL_TESTS", raising=False)
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(REAL_MODEL_TEST)

    result = pytester.runpytest()

    result.assert_outcomes(passed=1, skipped=1)


def test_real_model_tests_run_when_enabled(pytester, monkeypatch):
    monkeypatch.setenv("RUN_REAL_MODEL_TESTS", "1")
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(REAL_MODEL_TEST)

    result = pytester.runpytest()

    result.assert_outcomes(passed=2)
