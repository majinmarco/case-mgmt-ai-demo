"""The demo cases through the CLI: seeding the queue from the cache, refreshing the cache and the
regression suite, with the review pipeline faked."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from case_mgmt_ai_demo import main
from case_mgmt_ai_demo.demo_cases import DemoCases
from case_mgmt_ai_demo.determination import Determination
from case_mgmt_ai_demo.permit_request import parse_permit_request
from case_mgmt_ai_demo.review import (
    ClauseSelection,
    Recommendation,
    RequirementAssessment,
    SectionRelevance,
)
from case_mgmt_ai_demo.store import PermitRequestStore
from test_cli import fee_check, recommendation
from test_review import GOOD_PCOC as PCOC
from test_review import GOOD_PERMISO_UNICO as PERMISO_UNICO

# Each declaring the fee it owes, as a demo case expected to pass its Fee Check does.
GOOD_PCOC = PCOC | {"declared_fee": 1210}
GOOD_PERMISO_UNICO = PERMISO_UNICO | {"licenses_requested": [], "declared_fee": 100}
APPROVE = {"verdict": "approve", "borderline": False, "fee_matches": True}


def reviewed(raw: Mapping[str, Any], **overrides) -> Recommendation:
    fields = {"request": parse_permit_request(raw), "fee_check": fee_check(raw)}
    return recommendation(**fields | overrides)


class Demo:
    """A demo cases directory and a data directory, both temporary."""

    def __init__(self, tmp_path):
        self.directory = tmp_path / "demo_cases"
        self.directory.mkdir()
        self.db_path = tmp_path / "data" / "app.sqlite"

    def case(self, slug: str, raw: Mapping[str, Any], expected: Mapping[str, Any] = APPROVE):
        case = {"expected": expected, "rationale": "Cumple.", "request": raw}
        (self.directory / f"{slug}.json").write_text(json.dumps(case), encoding="utf-8")

    def cache(self, slug: str, rec: Recommendation):
        (self.directory / "cache").mkdir(exist_ok=True)
        self.cache_path(slug).write_text(rec.model_dump_json(), encoding="utf-8")

    def cache_path(self, slug: str):
        return self.directory / "cache" / f"{slug}.json"

    def cached(self, slug: str) -> Recommendation:
        return Recommendation.model_validate_json(self.cache_path(slug).read_text("utf-8"))

    @property
    def store(self) -> PermitRequestStore:
        return PermitRequestStore(self.db_path)


@pytest.fixture
def demo(tmp_path, monkeypatch):
    monkeypatch.setenv("PRA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PRA_DEMO_CASES_DIR", str(tmp_path / "demo_cases"))
    # Seeding must need neither a key nor a network: the real pipeline cannot be built.
    monkeypatch.setenv("OPENAI_API_KEY", "")
    return Demo(tmp_path)


class CasePipeline:
    """Reviews each Permit Request as the Recommendation given for its Applicant."""

    def __init__(self, *outcomes: Recommendation | Exception):
        self.outcomes = list(outcomes)
        self.reviewed: list[str] = []

    def review(self, raw: Mapping[str, Any]) -> Recommendation:
        self.reviewed.append(raw["applicant_name"])
        for outcome in self.outcomes:
            if isinstance(outcome, Exception):
                raise outcome
            if outcome.request.applicant_name == raw["applicant_name"]:
                return outcome
        raise AssertionError(f"unexpected review of {raw['applicant_name']}")


def test_seed_stores_each_demo_case_with_its_cached_recommendation_in_file_order(demo, capsys):
    demo.case("2-cafeteria", GOOD_PERMISO_UNICO)
    demo.case("1-vivienda", GOOD_PCOC)
    demo.cache("1-vivienda", reviewed(GOOD_PCOC, approval_score=0.91))
    demo.cache("2-cafeteria", reviewed(GOOD_PERMISO_UNICO, approval_score=0.77))

    assert main(["seed"]) == 0

    first, second = demo.store.all()
    assert (first.request.request_type, second.request.request_type) == ("PCOC", "PERMISO_UNICO")
    assert demo.store.recommendation(first.id) == reviewed(GOOD_PCOC, approval_score=0.91)
    assert demo.store.recommendation(second.id) == reviewed(GOOD_PERMISO_UNICO, approval_score=0.77)
    out = capsys.readouterr().out
    assert "1-vivienda" in out and "Aprobar" in out and "0.91" in out


def test_seed_again_adds_nothing(demo):
    demo.case("1-vivienda", GOOD_PCOC)
    demo.cache("1-vivienda", reviewed(GOOD_PCOC))
    main(["seed"])

    assert main(["seed"]) == 0

    assert len(demo.store.all()) == 1


def test_seed_gives_a_case_stored_unreviewed_the_recommendation_cached_since(demo):
    demo.case("1-vivienda", GOOD_PCOC)
    main(["seed"])
    demo.cache("1-vivienda", reviewed(GOOD_PCOC, approval_score=0.91))

    assert main(["seed"]) == 0

    [stored] = demo.store.all()
    assert demo.store.recommendation(stored.id) == reviewed(GOOD_PCOC, approval_score=0.91)


def test_a_case_that_is_not_borderline_must_name_its_expected_verdict(demo, capsys):
    demo.case("1-vivienda", GOOD_PCOC, APPROVE | {"verdict": None})

    assert main(["seed"]) == 1

    assert "1-vivienda.json" in capsys.readouterr().err


def test_seed_again_after_a_refresh_stores_the_new_recommendation(demo):
    demo.case("1-vivienda", GOOD_PCOC)
    demo.cache("1-vivienda", reviewed(GOOD_PCOC, approval_score=0.91))
    main(["seed"])
    demo.cache("1-vivienda", reviewed(GOOD_PCOC, approval_score=0.74))

    assert main(["seed"]) == 0

    [stored] = demo.store.all()
    assert demo.store.recommendation(stored.id) == reviewed(GOOD_PCOC, approval_score=0.74)


def test_seed_leaves_a_determined_case_with_the_recommendation_it_was_determined_on(demo):
    demo.case("1-vivienda", GOOD_PCOC)
    demo.cache("1-vivienda", reviewed(GOOD_PCOC, approval_score=0.91))
    main(["seed"])
    [stored] = demo.store.all()
    demo.store.save_determination(
        stored.id, Determination(kind="accept", recommended="approve", outcome="approve")
    )
    demo.cache("1-vivienda", reviewed(GOOD_PCOC, approval_score=0.74))

    assert main(["seed"]) == 0

    assert demo.store.recommendation(stored.id) == reviewed(GOOD_PCOC, approval_score=0.91)
    assert demo.store.determination(stored.id) is not None


def test_seed_with_reset_empties_the_queue_first(demo):
    demo.case("1-vivienda", GOOD_PCOC)
    demo.cache("1-vivienda", reviewed(GOOD_PCOC))
    demo.store.add(parse_permit_request(GOOD_PERMISO_UNICO))

    assert main(["seed", "--reset"]) == 0

    [stored] = demo.store.all()
    assert stored.request.request_type == "PCOC"
    assert demo.store.recommendation(stored.id) is not None


def test_seed_stores_a_case_without_a_cached_recommendation_unreviewed_and_says_so(demo, capsys):
    demo.case("1-vivienda", GOOD_PCOC)

    assert main(["seed"]) == 1

    [stored] = demo.store.all()
    assert demo.store.recommendation(stored.id) is None
    err = capsys.readouterr().err
    assert "1-vivienda" in err and "refresh" in err


def test_seed_ignores_a_cached_recommendation_made_for_other_fields(demo, capsys):
    demo.case("1-vivienda", GOOD_PCOC | {"floors": 1})
    demo.cache("1-vivienda", reviewed(GOOD_PCOC))

    assert main(["seed"]) == 1

    [stored] = demo.store.all()
    assert demo.store.recommendation(stored.id) is None
    assert "refresh" in capsys.readouterr().err


def test_seed_without_demo_cases_fails(demo, capsys):
    assert main(["seed"]) == 1

    assert str(demo.directory) in capsys.readouterr().err


def test_refresh_reviews_every_case_and_caches_its_recommendation(demo, capsys):
    demo.case("1-vivienda", GOOD_PCOC)
    demo.case("2-cafeteria", GOOD_PERMISO_UNICO)
    pipeline = CasePipeline(
        reviewed(GOOD_PCOC, approval_score=0.91),
        reviewed(GOOD_PERMISO_UNICO, approval_score=0.77),
    )

    assert main(["refresh"], pipeline=pipeline) == 0

    assert demo.cached("1-vivienda") == reviewed(GOOD_PCOC, approval_score=0.91)
    assert demo.cached("2-cafeteria") == reviewed(GOOD_PERMISO_UNICO, approval_score=0.77)
    out = capsys.readouterr().out
    assert "1-vivienda" in out and "0.91" in out


def test_a_refreshed_cache_is_what_seed_stores(demo):
    demo.case("1-vivienda", GOOD_PCOC)
    main(["refresh"], pipeline=CasePipeline(reviewed(GOOD_PCOC, approval_score=0.91)))

    assert main(["seed"]) == 0

    [stored] = demo.store.all()
    assert demo.store.recommendation(stored.id) == reviewed(GOOD_PCOC, approval_score=0.91)


def test_refresh_keeps_the_cached_recommendation_when_the_new_one_is_not_the_expected(
    demo, capsys
):
    demo.case("1-vivienda", GOOD_PCOC)
    demo.cache("1-vivienda", reviewed(GOOD_PCOC, approval_score=0.91))
    flipped = reviewed(GOOD_PCOC, approval_score=0.12, verdict="not_approve")

    assert main(["refresh"], pipeline=CasePipeline(flipped)) == 1

    assert demo.cached("1-vivienda").approval_score == 0.91
    err = capsys.readouterr().err
    assert "1-vivienda" in err and "se esperaba «Aprobar», se obtuvo «No aprobar»" in err


def test_refresh_of_one_case_leaves_the_others_alone(demo):
    demo.case("1-vivienda", GOOD_PCOC)
    demo.case("2-cafeteria", GOOD_PERMISO_UNICO)
    pipeline = CasePipeline(reviewed(GOOD_PERMISO_UNICO))

    assert main(["refresh", "--case", "2-cafeteria"], pipeline=pipeline) == 0

    assert pipeline.reviewed == [GOOD_PERMISO_UNICO["applicant_name"]]
    assert not demo.cache_path("1-vivienda").exists()


def test_refresh_of_an_unknown_case_names_the_cases_there_are(demo, capsys):
    demo.case("1-vivienda", GOOD_PCOC)

    assert main(["refresh", "--case", "9-otro"], pipeline=CasePipeline()) == 1

    assert "1-vivienda" in capsys.readouterr().err


def test_refresh_reports_a_review_that_fails_and_carries_on(demo, capsys):
    demo.case("1-vivienda", GOOD_PCOC)
    demo.cache("1-vivienda", reviewed(GOOD_PCOC, approval_score=0.91))

    assert main(["refresh"], pipeline=CasePipeline(RuntimeError("sin índice"))) == 1

    assert demo.cached("1-vivienda").approval_score == 0.91
    assert "sin índice" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("expected", "outcome", "problem"),
    [
        (
            APPROVE | {"borderline": True},
            {},
            "caso límite: se esperaba un caso límite",
        ),
        (APPROVE, {"borderline": True}, "caso límite: no se esperaba"),
        (APPROVE | {"fee_matches": False}, {}, "derechos: se esperaba que los declarados no"),
        (
            APPROVE | {"verdict": None, "borderline": True},
            {"verdict": "manual_review", "approval_score": None, "borderline": True},
            "veredicto: se esperaba una puntuación",
        ),
    ],
)
def test_refresh_checks_the_borderline_flag_and_the_fee_check(
    demo, capsys, expected, outcome, problem
):
    demo.case("1-vivienda", GOOD_PCOC, expected)
    rec = reviewed(GOOD_PCOC, **outcome)

    assert main(["refresh"], pipeline=CasePipeline(rec)) == 1

    assert problem in capsys.readouterr().err


def test_a_borderline_case_may_fall_either_side_of_the_cutoff(demo):
    demo.case("1-vivienda", GOOD_PCOC, APPROVE | {"verdict": None, "borderline": True})
    rec = reviewed(GOOD_PCOC, approval_score=0.45, verdict="not_approve", borderline=True)

    assert main(["refresh"], pipeline=CasePipeline(rec)) == 0


def requirement(requirement_id: str, probability: float | None) -> RequirementAssessment:
    return RequirementAssessment(
        id=requirement_id,
        text_es="Requisito.",
        clause_refs=["RC-2023-E:2.1.9.2(k)"],
        probability=probability,
    )


def selection(ref: str, category: str) -> ClauseSelection:
    return ClauseSelection(ref=f"RC-2023-E:{ref}", category=category, reason="Razón.")


def relevance(number: str, applies: bool) -> SectionRelevance:
    return SectionRelevance(section_id=f"RC-2023-E:{number}", applies=applies, reason="Razón.")


def regress(demo, capsys, baseline: Recommendation, current: Recommendation, *args: str):
    """The exit code and report of `regress` on one case with that baseline and that review."""
    demo.case("1-vivienda", GOOD_PCOC)
    demo.cache("1-vivienda", baseline)
    exit_code = main(["regress", *args], pipeline=CasePipeline(current))
    return exit_code, capsys.readouterr().out


def test_regress_passes_a_case_reviewed_as_expected_and_as_its_baseline(demo, capsys):
    exit_code, out = regress(demo, capsys, reviewed(GOOD_PCOC), reviewed(GOOD_PCOC))

    assert exit_code == 0
    assert "1-vivienda: Aprobar (0.86)" in out
    assert "⚠" not in out


def test_regress_reports_an_expected_verdict_that_flipped_and_fails(demo, capsys):
    flipped = reviewed(GOOD_PCOC, approval_score=0.44, verdict="not_approve")

    exit_code, out = regress(demo, capsys, reviewed(GOOD_PCOC), flipped)

    assert exit_code == 1
    assert "veredicto: se esperaba «Aprobar», se obtuvo «No aprobar»" in out


def test_regress_does_not_touch_the_cache(demo, capsys):
    flipped = reviewed(GOOD_PCOC, approval_score=0.44, verdict="not_approve")

    regress(demo, capsys, reviewed(GOOD_PCOC), flipped)
    regress(demo, capsys, reviewed(GOOD_PCOC), reviewed(GOOD_PCOC, approval_score=0.99))

    assert demo.cached("1-vivienda") == reviewed(GOOD_PCOC)


def test_regress_reports_an_approval_score_that_moved_though_the_verdict_did_not_flip(
    demo, capsys
):
    baseline = reviewed(GOOD_PCOC, approval_score=0.86)

    exit_code, out = regress(demo, capsys, baseline, reviewed(GOOD_PCOC, approval_score=0.61))

    assert exit_code == 0
    assert "⚠ Puntuación de aprobación: 0.86 → 0.61" in out


def test_regress_leaves_out_a_score_that_moved_within_the_threshold(demo, capsys):
    baseline = reviewed(GOOD_PCOC, approval_score=0.86)

    _, out = regress(demo, capsys, baseline, reviewed(GOOD_PCOC, approval_score=0.72))

    assert "⚠" not in out


def test_the_threshold_is_configured(demo, capsys, monkeypatch):
    monkeypatch.setenv("PRA_REGRESSION_DRIFT_THRESHOLD", "0.1")
    baseline = reviewed(GOOD_PCOC, approval_score=0.86)

    _, out = regress(demo, capsys, baseline, reviewed(GOOD_PCOC, approval_score=0.72))

    assert "⚠ Puntuación de aprobación: 0.86 → 0.72" in out
    assert "0.10" in out


def test_regress_reports_each_requirement_whose_probability_moved(demo, capsys):
    baseline = reviewed(
        GOOD_PCOC,
        requirements=[
            requirement("pcoc_planos", 0.95),
            requirement("pcoc_predio", 0.90),
            requirement("seccion_3_2_1_2", 0.88),
            requirement("seccion_9_5_2_2", 0.90),
        ],
    )
    current = reviewed(
        GOOD_PCOC,
        requirements=[
            requirement("pcoc_planos", 0.70),
            requirement("pcoc_predio", 0.93),
            requirement("seccion_3_2_1_2", None),
            requirement("seccion_3_2_1_1", 0.91),
        ],
    )

    _, out = regress(demo, capsys, baseline, current)

    assert "⚠ pcoc_planos: 0.95 → 0.70" in out
    assert "pcoc_predio" not in out
    assert "⚠ seccion_3_2_1_2: 0.88 → no evaluado" in out
    assert "⚠ seccion_9_5_2_2: 0.90 → ya no es un requisito" in out
    assert "⚠ seccion_3_2_1_1: requisito nuevo → 0.91" in out


def test_regress_reports_each_clause_whose_category_changed(demo, capsys):
    baseline = reviewed(
        GOOD_PCOC,
        selection=[selection("2.1.9.2(m)", "tramite_del_sistema"), selection("2.1.9.2(l)", "no_aplica")],
        unclassified_clause_refs=["RC-2023-E:2.1.9.2(q)"],
    )
    current = reviewed(
        GOOD_PCOC,
        selection=[
            selection("2.1.9.2(m)", "aplica"),
            selection("2.1.9.2(l)", "no_aplica"),
            selection("2.1.9.2(q)", "no_verificable"),
        ],
    )

    exit_code, out = regress(demo, capsys, baseline, current)

    assert exit_code == 0
    assert "⚠ RC-2023-E:2.1.9.2(m): trámite del sistema → aplica (cambia lo que se puntúa)" in out
    assert "⚠ RC-2023-E:2.1.9.2(q): sin clasificar → no verificable (cambia lo que se puntúa)" in out
    assert "2.1.9.2(l)" not in out


def test_a_clause_set_aside_under_another_category_is_reported_as_not_changing_the_score(
    demo, capsys
):
    baseline = reviewed(GOOD_PCOC, selection=[selection("2.1.9.2(l)", "no_aplica")])
    current = reviewed(GOOD_PCOC, selection=[selection("2.1.9.2(l)", "no_verificable")])

    _, out = regress(demo, capsys, baseline, current)

    assert "⚠ RC-2023-E:2.1.9.2(l): no aplica → no verificable\n" in out


def test_regress_reports_a_clause_its_section_gained_or_lost(demo, capsys):
    applies = [relevance("2.1.9.2", True)]
    baseline = reviewed(
        GOOD_PCOC,
        section_relevance=applies,
        selection=[selection("2.1.9.2(k)", "aplica"), selection("2.1.9.2(l)", "no_aplica")],
    )
    current = reviewed(
        GOOD_PCOC,
        section_relevance=applies,
        selection=[selection("2.1.9.2(k)", "aplica"), selection("2.1.9.2(x)", "aplica")],
    )

    _, out = regress(demo, capsys, baseline, current)

    assert "⚠ RC-2023-E:2.1.9.2(l): no aplica → no existía\n" in out
    assert "⚠ RC-2023-E:2.1.9.2(x): no existía → aplica (cambia lo que se puntúa)" in out
    assert "2.1.9.2(k)" not in out


def test_regress_reports_each_section_whose_relevance_changed(demo, capsys):
    baseline = reviewed(
        GOOD_PCOC,
        section_relevance=[relevance("2.2.6", False), relevance("3.2.1.2", True), relevance("9.5.2.2", True)],
        selection=[selection("9.5.2.2(a)", "aplica")],
    )
    current = reviewed(
        GOOD_PCOC,
        section_relevance=[relevance("2.2.6", True), relevance("3.2.1.2", True), relevance("8.1.2.1", False)],
        selection=[selection("2.2.6(a)", "aplica")],
    )

    _, out = regress(demo, capsys, baseline, current)

    assert "⚠ RC-2023-E:2.2.6: no aplica → aplica" in out
    assert "⚠ RC-2023-E:9.5.2.2: aplica → no recuperada" in out
    assert "⚠ RC-2023-E:8.1.2.1: no recuperada → no aplica" in out
    assert "3.2.1.2" not in out
    # The Clauses of a Section retrieved on one side only have nothing to be compared with.
    assert "9.5.2.2(a)" not in out and "2.2.6(a)" not in out


def test_regress_says_when_a_case_has_no_baseline(demo, capsys):
    demo.case("1-vivienda", GOOD_PCOC)

    assert main(["regress"], pipeline=CasePipeline(reviewed(GOOD_PCOC))) == 0

    out = capsys.readouterr().out
    assert "sin línea base" in out and "refresh" in out


def test_regress_reports_a_review_that_fails_and_carries_on(demo, capsys):
    demo.case("1-vivienda", GOOD_PCOC)

    assert main(["regress"], pipeline=CasePipeline(RuntimeError("sin índice"))) == 1

    assert "sin índice" in capsys.readouterr().out


# The demo script itself: the cases in the repository and their cached Recommendations.
REPO_CASES = DemoCases(Path(__file__).parent.parent / "demo_cases")


def test_the_demo_script_is_eight_cases_across_both_request_types():
    expected = [case.expected for case in REPO_CASES.cases]

    assert len(expected) == 8
    assert {case.request.request_type for case in REPO_CASES.cases} == {"PCOC", "PERMISO_UNICO"}
    clear = [e for e in expected if not e.borderline and e.fee_matches]
    assert sum(1 for e in clear if e.verdict == "approve") == 3
    assert sum(1 for e in clear if e.verdict == "not_approve") == 2
    assert sum(1 for e in expected if e.borderline) == 2
    assert sum(1 for e in expected if not e.fee_matches) == 1


@pytest.mark.parametrize("case", REPO_CASES.cases, ids=lambda case: case.slug)
def test_each_demo_case_has_the_expected_recommendation_cached(case):
    cached = REPO_CASES.cached(case)

    assert cached is not None, "no Recommendation cached for the case as it is: run refresh"
    assert case.expected.unmet_es(cached) == []


def test_the_wrong_fee_case_shows_the_fee_check_note():
    [case] = [case for case in REPO_CASES.cases if not case.expected.fee_matches]

    cached = REPO_CASES.cached(case)

    assert cached is not None
    [note] = [note for note in cached.notes if "COBROS:AnejoA:6" in note.section_ids]
    assert note.curable and "$1,524.00" in note.text and "$1,540.00" in note.text
