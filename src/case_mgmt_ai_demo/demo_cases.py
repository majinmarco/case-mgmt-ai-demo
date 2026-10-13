"""The demo cases: curated Permit Requests, each with the outcome it is expected to get and why.

They are the demo script and a regression suite. A case is a JSON file in the demo cases directory
(`PRA_DEMO_CASES_DIR`); its Recommendation, as the real models last gave it, is cached beside it
in `cache/`. Seeding the queue reads only those files, so the demo needs no network. The cached
Recommendation is also the baseline the regression suite compares a new review with.
"""

import json
from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from case_mgmt_ai_demo.cli import CATEGORIES_ES, NOT_ASSESSED, VERDICTS_ES, RecommendationSource
from case_mgmt_ai_demo.permit_request import PermitRequest
from case_mgmt_ai_demo.review import Recommendation
from case_mgmt_ai_demo.store import AlreadyDetermined, PermitRequestStore

CACHE_DIR = "cache"
_UNCLASSIFIED = "sin_clasificar"
_ABSENT = "ausente"  # not a Clause of its Section in that review
_CATEGORIES_ES = CATEGORIES_ES | {_UNCLASSIFIED: "sin clasificar", _ABSENT: "no existía"}
# The categories under which a Clause reaches decide.
_SCORED = ("aplica", _UNCLASSIFIED)
# Reviews of different cases share nothing, and each waits on the model APIs.
_CONCURRENT_REVIEWS = 4


class Expected(BaseModel):
    """What the Recommendation of a demo case should come to."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["approve", "not_approve"] | None = Field(
        description="None for a Borderline case, whose score may fall either side of the cutoff."
    )
    borderline: bool
    fee_matches: bool = Field(description="False when the Fee Check should add its Note.")

    @model_validator(mode="after")
    def _only_a_borderline_case_names_no_verdict(self) -> Self:
        if self.verdict is None and not self.borderline:
            raise ValueError("A case that is not Borderline must name its expected verdict")
        return self

    def unmet_es(self, recommendation: Recommendation) -> list[str]:
        """Each way the Recommendation is not the expected one; empty when it is."""
        unmet = [
            f"falló una llamada a la API ({error.step}): {error.message}"
            for error in recommendation.errors
        ]
        got = VERDICTS_ES[recommendation.verdict]
        if self.verdict is None:
            if recommendation.verdict == "manual_review":
                unmet.append(f"veredicto: se esperaba una puntuación, se obtuvo «{got}»")
        elif recommendation.verdict != self.verdict:
            unmet.append(f"veredicto: se esperaba «{VERDICTS_ES[self.verdict]}», se obtuvo «{got}»")
        if recommendation.borderline != self.borderline:
            expected = "se esperaba un caso límite" if self.borderline else "no se esperaba"
            unmet.append(
                f"caso límite: {expected}, y la puntuación fue {_shown(recommendation.approval_score)}"
            )
        fee_check = recommendation.fee_check
        if fee_check is None or fee_check.matches is not self.fee_matches:
            expected = "coincidieran" if self.fee_matches else "no coincidieran"
            unmet.append(f"derechos: se esperaba que los declarados {expected} con los calculados")
        return unmet


class DemoCase(BaseModel):
    """A curated Permit Request with its expected outcome."""

    model_config = ConfigDict(extra="forbid")

    slug: str = Field(description="The name of its file, without the extension.")
    expected: Expected
    rationale: str = Field(min_length=1, description="Why that outcome is expected.")
    request: PermitRequest


class InvalidDemoCases(ValueError):
    """The demo cases directory holds no case, or a file in it is not one."""


@dataclass(frozen=True)
class Seeded:
    """A demo case in the queue."""

    case: DemoCase
    request_id: int
    recommendation: Recommendation | None  # the one it has in the queue; None: unreviewed
    added: bool  # False when its Permit Request was already stored


@dataclass(frozen=True)
class Refreshed:
    """A demo case reviewed again for the cache."""

    case: DemoCase
    recommendation: Recommendation | None  # None when the review itself failed
    problems: list[str] = field(default_factory=list)

    @property
    def cached(self) -> bool:
        """Only the expected Recommendation replaces the cached one."""
        return not self.problems


@dataclass(frozen=True)
class Regressed:
    """A demo case reviewed again and compared with what is expected and with its baseline."""

    case: DemoCase
    recommendation: Recommendation | None  # None when the review itself failed
    problems: list[str]  # ways it is not the expected Recommendation: the suite fails
    has_baseline: bool
    moved: list[str] = field(default_factory=list)  # scores that moved past the threshold
    reclassified: list[str] = field(default_factory=list)  # Sections and Clauses judged anew


class DemoCases:
    """The demo cases of a directory, in file order, and their cached Recommendations."""

    def __init__(self, directory: Path):
        self._directory = directory
        paths = sorted(directory.glob("*.json")) if directory.is_dir() else []
        if not paths:
            raise InvalidDemoCases(f"No hay casos de demostración en {directory}")
        self.cases = [_load(path) for path in paths]

    def cached(self, case: DemoCase) -> Recommendation | None:
        """The cached Recommendation of the case; None when there is none, or when it was made
        for other fields or by a version whose Recommendations this one cannot read."""
        path = self._cache_path(case)
        if not path.is_file():
            return None
        try:
            recommendation = Recommendation.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:
            return None
        return recommendation if recommendation.request == case.request else None

    def seed(self, store: PermitRequestStore) -> list[Seeded]:
        """Store every case with its cached Recommendation. A case already stored is given
        the cached Recommendation too, unless it has its Determination, which was made on
        the Recommendation it has."""
        stored = {s.request.model_dump_json(): s.id for s in store.all()}
        seeded: list[Seeded] = []
        for case in self.cases:
            recommendation = self.cached(case)
            request_id = stored.get(case.request.model_dump_json())
            added = request_id is None
            if request_id is None:
                request_id = store.add(case.request)
            if recommendation:
                try:
                    store.save_recommendation(request_id, recommendation)
                except AlreadyDetermined:
                    pass
            seeded.append(Seeded(case, request_id, store.recommendation(request_id), added))
        return seeded

    def refresh(
        self, pipeline: RecommendationSource, slugs: Collection[str] | None = None
    ) -> list[Refreshed]:
        """Review the cases (all, or those named) and cache each Recommendation that is the
        expected one; any other leaves the case's cache as it was, so the demo still runs."""

        def refresh(case: DemoCase) -> Refreshed:
            recommendation, problems = _review(case, pipeline)
            if recommendation and not problems:
                path = self._cache_path(case)
                path.parent.mkdir(exist_ok=True)
                path.write_text(recommendation.model_dump_json(indent=1) + "\n", encoding="utf-8")
            return Refreshed(case, recommendation, problems)

        cases = [case for case in self.cases if slugs is None or case.slug in slugs]
        with ThreadPoolExecutor(max_workers=_CONCURRENT_REVIEWS) as pool:
            return list(pool.map(refresh, cases))

    def regress(self, pipeline: RecommendationSource, threshold: float) -> list[Regressed]:
        """Review every case and compare each Recommendation with the expected one and with
        the cached one, its baseline: the scores that moved by more than `threshold`, and the
        Sections and Clauses classified differently. Nothing is cached."""

        def regress(case: DemoCase) -> Regressed:
            recommendation, problems = _review(case, pipeline)
            baseline = self.cached(case)
            if recommendation is None or baseline is None:
                return Regressed(case, recommendation, problems, baseline is not None)
            return Regressed(
                case,
                recommendation,
                problems,
                has_baseline=True,
                moved=_moved_es(baseline, recommendation, threshold),
                reclassified=_reclassified_es(baseline, recommendation),
            )

        with ThreadPoolExecutor(max_workers=_CONCURRENT_REVIEWS) as pool:
            return list(pool.map(regress, self.cases))

    def _cache_path(self, case: DemoCase) -> Path:
        return self._directory / CACHE_DIR / f"{case.slug}.json"


def _load(path: Path) -> DemoCase:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return DemoCase.model_validate({**raw, "slug": path.stem})
    except (json.JSONDecodeError, TypeError, ValidationError) as error:
        reason = str(error).splitlines()[0]
        raise InvalidDemoCases(f"{path} no es un caso de demostración: {reason}") from error


def _review(
    case: DemoCase, pipeline: RecommendationSource
) -> tuple[Recommendation | None, list[str]]:
    """The case's Recommendation from the pipeline, and each way it is not the expected one."""
    try:
        recommendation = pipeline.review(case.request.model_dump(mode="json"))
    except Exception as error:  # reported with the case: a missing index, a Criteria mismatch
        return None, [f"la revisión falló: {type(error).__name__}: {error}"]
    return recommendation, case.expected.unmet_es(recommendation)


def _moved_es(baseline: Recommendation, current: Recommendation, threshold: float) -> list[str]:
    """The Approval Score and each Requirement probability that moved by more than the
    threshold, or that is assessed on one side only."""
    moved: list[str] = []
    if _far(baseline.approval_score, current.approval_score, threshold):
        moved.append(
            "Puntuación de aprobación: "
            f"{_shown(baseline.approval_score)} → {_shown(current.approval_score)}"
        )
    before = {r.id: r.probability for r in baseline.requirements}
    after = {r.id: r.probability for r in current.requirements}
    for requirement_id in before | after:
        if requirement_id not in after:
            moved.append(f"{requirement_id}: {_shown(before[requirement_id])} → ya no es un requisito")
        elif requirement_id not in before:
            moved.append(f"{requirement_id}: requisito nuevo → {_shown(after[requirement_id])}")
        elif _far(before[requirement_id], after[requirement_id], threshold):
            moved.append(
                f"{requirement_id}: {_shown(before[requirement_id])} → "
                f"{_shown(after[requirement_id])}"
            )
    return moved


def _far(before: float | None, after: float | None, threshold: float) -> bool:
    if before is None or after is None:
        return before is not after
    return abs(after - before) > threshold


def _reclassified_es(baseline: Recommendation, current: Recommendation) -> list[str]:
    """Each retrieved Section whose relevance changed, and each Clause whose category did.
    Clauses are compared only where both reviews classified the Section's Clauses."""
    before = {r.section_id: _applies_es(r.applies) for r in baseline.section_relevance}
    after = {r.section_id: _applies_es(r.applies) for r in current.section_relevance}
    not_retrieved = "no recuperada"
    changed = [
        f"{section_id}: {before.get(section_id, not_retrieved)} → "
        f"{after.get(section_id, not_retrieved)}"
        for section_id in before | after
        if before.get(section_id) != after.get(section_id)
    ]
    # Where the Section applied both times, a Clause on one side only is one the Section's
    # text gained or lost: a change of extraction or chunking.
    classified = {s for s in before if before[s] == after.get(s) == _applies_es(True)}
    was, now = _categories(baseline), _categories(current)
    for ref in was | now:
        category, new_category = was.get(ref, _ABSENT), now.get(ref, _ABSENT)
        if category == new_category or ref.split("(", 1)[0] not in classified:
            continue
        # Set aside under another category, the Clause is as absent from decide as before.
        scored = (category in _SCORED) != (new_category in _SCORED)
        changed.append(
            f"{ref}: {_CATEGORIES_ES[category]} → {_CATEGORIES_ES[new_category]}"
            + (" (cambia lo que se puntúa)" if scored else "")
        )
    return changed


def _applies_es(applies: bool) -> str:
    return "aplica" if applies else "no aplica"


def _categories(recommendation: Recommendation) -> dict[str, str]:
    """The category of every Clause the review considered; `_UNCLASSIFIED` for those the
    selector left out."""
    categories: dict[str, str] = {s.ref: s.category for s in recommendation.selection}
    return categories | dict.fromkeys(recommendation.unclassified_clause_refs, _UNCLASSIFIED)


def _shown(probability: float | None) -> str:
    return NOT_ASSESSED if probability is None else f"{probability:.2f}"
