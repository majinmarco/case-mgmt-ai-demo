"""The Determination: the Reviewer's final action on a Permit Request.

Outside the review graph: the Reviewer accepts the Recommendation, overrides it with a reason, or
issues a Cure Request listing the deficiencies the Applicant must fix.
"""

from typing import Literal

from pydantic import BaseModel, Field

from case_mgmt_ai_demo.review import Recommendation, Verdict

DeterminationKind = Literal["accept", "override", "cure_request"]
Outcome = Literal["approve", "not_approve"]


class Determination(BaseModel):
    kind: DeterminationKind
    recommended: Verdict = Field(description="The verdict of the Recommendation it was made on.")
    outcome: Outcome | None = Field(
        description="What the Reviewer determined; None for a Cure Request, which does neither."
    )
    reason: str = Field(default="", description="Why the Recommendation was overridden.")
    deficiencies: list[str] = Field(
        default_factory=list, description="What a Cure Request asks the Applicant to fix."
    )

    @property
    def status_es(self) -> str:
        """The Determination in a line, as the Reviewer reads it on the case page and the queue."""
        if self.outcome is None:
            return "Requerimiento de subsanación"
        outcome = OUTCOMES_ES[self.outcome]
        if self.kind == "accept":
            return f"Recomendación aceptada: {outcome}"
        if self.recommended == "manual_review":
            return f"Determinación sin veredicto recomendado: {outcome}"
        return f"Recomendación anulada: {outcome}"


OUTCOMES_ES: dict[Outcome, str] = {"approve": "Aprobar", "not_approve": "No aprobar"}


class InvalidDetermination(ValueError):
    """What the Reviewer asked for cannot be recorded; the message, in Spanish, says why."""


def cure_request_draft(recommendation: Recommendation) -> str:
    """What a Cure Request starts from: each curable Note with its citations, one per line
    (a Note that spans lines is put on one: each line is a deficiency)."""
    return "\n".join(
        f"{' '.join(note.text.split())} ({', '.join(note.section_ids)})"
        for note in recommendation.notes
        if note.curable
    )


def determine(
    recommendation: Recommendation,
    kind: str,
    reason: str = "",
    outcome: str = "",
    deficiencies: str = "",
) -> Determination:
    """The Determination the Reviewer asks for on a Recommendation.

    Overriding an Approve / Not approve verdict determines the other one. A Recommendation
    without a verdict (manual review) cannot be accepted: the Reviewer names the outcome.
    A Cure Request lists one deficiency per line of `deficiencies`.
    """
    recommended = recommendation.verdict
    if kind == "accept":
        if recommended == "manual_review":
            raise InvalidDetermination(
                "La Recomendación no tiene veredicto que aceptar: indique el resultado y la razón."
            )
        return Determination(kind="accept", recommended=recommended, outcome=recommended)
    if kind == "override":
        if not reason.strip():
            raise InvalidDetermination("Anular la Recomendación requiere una razón.")
        if recommended == "approve":
            determined: Outcome = "not_approve"
        elif recommended == "not_approve":
            determined = "approve"
        elif outcome == "approve" or outcome == "not_approve":
            determined = outcome
        else:
            raise InvalidDetermination("Indique el resultado: Aprobar o No aprobar.")
        return Determination(
            kind="override", recommended=recommended, outcome=determined, reason=reason.strip()
        )
    if kind == "cure_request":
        listed = [line.strip() for line in deficiencies.splitlines() if line.strip()]
        if not listed:
            raise InvalidDetermination(
                "Un Requerimiento de subsanación debe enumerar al menos una deficiencia."
            )
        return Determination(
            kind="cure_request", recommended=recommended, outcome=None, deficiencies=listed
        )
    raise InvalidDetermination("Determinación desconocida.")
