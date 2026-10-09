"""Criteria: the curated requirements of each Request Type, each assessed on its own.

Curated in YAML, one file per Request Type (`fixtures/criteria/`), so non-engineers can review
what is checked. Each Criterion pins the Clauses that establish it: the pipeline always
retrieves their Sections, always scores the request against them, and asks the Decisions API
one predicate per Criterion over those Clauses alone.
"""

from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field

from case_mgmt_ai_demo.chunks import Clause, RequestType
from case_mgmt_ai_demo.permit_request import ApplicationKind, PermitRequest


class Criterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$", description="Also the Decisions API question name.")
    request_type: RequestType
    text_es: str = Field(min_length=1)
    text_en: str = Field(min_length=1)
    pinned: list[str] = Field(
        min_length=1,
        description="Clause refs ('RC-2023-E:2.1.9.2(k)'), or Section IDs to pin every Clause.",
    )
    instructions: str = Field(min_length=1, description="The predicate question, in Spanish.")
    application_kinds: list[ApplicationKind] = Field(
        default_factory=lambda: list(get_args(ApplicationKind)),
        description="New requests, renewals, or both.",
    )

    @property
    def pinned_section_ids(self) -> list[str]:
        return list(dict.fromkeys(pin.split("(", 1)[0] for pin in self.pinned))

    def pins(self, clause: Clause) -> bool:
        return clause.ref in self.pinned or clause.section_id in self.pinned

    def unmatched_pins(self, clauses: list[Clause]) -> list[str]:
        """Pins that name none of the given Clauses."""
        matched = {c.ref for c in clauses} | {c.section_id for c in clauses}
        return [pin for pin in self.pinned if pin not in matched]

    def applies_to(self, request: PermitRequest) -> bool:
        return (
            request.request_type == self.request_type
            and request.application_kind in self.application_kinds
        )


class _CriteriaFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_type: RequestType
    criteria: list[dict]


def load_criteria(directory: Traversable | Path | None = None) -> list[Criterion]:
    """Every Criterion in the directory's YAML files, in file order; IDs must be unique."""
    directory = directory or files("case_mgmt_ai_demo.fixtures").joinpath("criteria")
    criteria: list[Criterion] = []
    for path in sorted(directory.iterdir(), key=lambda p: p.name):
        if not path.name.endswith(".yaml"):
            continue
        content = _CriteriaFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        criteria += [
            Criterion.model_validate(item | {"request_type": content.request_type})
            for item in content.criteria
        ]
    ids = [criterion.id for criterion in criteria]
    if duplicates := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"Duplicate Criterion IDs: {', '.join(duplicates)}")
    return criteria
