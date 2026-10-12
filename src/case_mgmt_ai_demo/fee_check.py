"""The Fee Check: does the fee a Permit Request declares match the COBROS fee schedule?

Deterministic: the fee owed is calculated from curated fee rules (`fixtures/fee_rules.yaml`) for
the request's Request Type and Applicant type, and compared with the declared fee. It never
touches the Approval Score; a mismatch reaches the Reviewer as a curable Note.
"""

from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Literal, NamedTuple, Self, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from case_mgmt_ai_demo.permit_request import (
    ApplicantType,
    EstablishmentType,
    PcocPermitRequest,
    PermisoUnicoPermitRequest,
    PermitRequest,
)

# The columns of the COBROS TERCERO.3 exemption table.
ChargeKind = Literal["licencias", "servicios", "aranceles", "estampillas"]

SQFT_PER_M2 = Decimal("10.7639")
_CENT = Decimal("0.01")


class _Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Charge(_Rule):
    ref: str = Field(min_length=1, description="The Anejo A row, e.g. '6(a)'.")
    concept_es: str = Field(min_length=1)
    kind: ChargeKind


class ExemptionRule(_Rule):
    """The percentage of each kind of charge one Applicant type pays."""

    description_es: str = Field(min_length=1)
    percent: dict[ChargeKind, int]
    percent_works_by_government: dict[ChargeKind, int] = Field(
        default_factory=dict,
        description="Overrides for works carried out wholly by the Government.",
    )

    @model_validator(mode="after")
    def _every_kind_has_a_percentage(self) -> Self:
        if missing := set(get_args(ChargeKind)) - self.percent.keys():
            raise ValueError(f"No percentage for {', '.join(sorted(missing))}")
        return self

    def percentages(self, works_by_government: bool) -> dict[ChargeKind, int]:
        return self.percent | (self.percent_works_by_government if works_by_government else {})


class Exemptions(_Rule):
    section_id: str
    applicant_types: dict[ApplicantType, ExemptionRule]

    @model_validator(mode="after")
    def _every_applicant_type_has_a_rule(self) -> Self:
        if missing := set(get_args(ApplicantType)) - self.applicant_types.keys():
            raise ValueError(f"No exemption rule for {', '.join(sorted(missing))}")
        return self


class ValueCharge(_Charge):
    """A charge per unit of estimated work value, or fraction of it."""

    rate: Decimal = Field(gt=0)
    minimum: Decimal = Field(default=Decimal(0), ge=0)


class PcocRules(_Rule):
    section_id: str
    value_unit: Decimal = Field(gt=0)
    charges: list[ValueCharge] = Field(min_length=1)


class AreaTier(_Rule):
    up_to_sqft: Decimal = Field(gt=0)
    per_year: Decimal = Field(ge=0)


class AreaTieredCharge(_Charge):
    """A yearly charge set by the floor area of the premises."""

    section_id: str
    tiers: list[AreaTier] = Field(min_length=1, description="In ascending floor area.")


class YearlyRate(_Charge):
    rate: Decimal = Field(ge=0)


class SanitaryLicenseRules(_Rule):
    section_id: str
    per_year: dict[EstablishmentType, YearlyRate]

    @model_validator(mode="after")
    def _every_establishment_type_has_a_rate(self) -> Self:
        if missing := set(get_args(EstablishmentType)) - self.per_year.keys():
            raise ValueError(f"No Licencia Sanitaria rate for {', '.join(sorted(missing))}")
        return self


class FlatCharge(_Charge):
    amount: Decimal = Field(ge=0)


class UseRules(_Rule):
    section_id: str
    base: FlatCharge
    bebidas_alcoholicas: FlatCharge


class PermisoUnicoRules(_Rule):
    prevencion_incendios: AreaTieredCharge
    licencia_sanitaria: SanitaryLicenseRules
    uso: UseRules


class FeeRules(_Rule):
    exemptions: Exemptions
    pcoc: PcocRules
    permiso_unico: PermisoUnicoRules


class FeeCharge(BaseModel):
    """One charge of the fee owed, before and after the Applicant's exemption."""

    ref: str = Field(description="The Anejo A row, e.g. '6(a)'.")
    section_id: str
    concept_es: str
    base: Decimal = Field(description="The charge before any exemption.")
    percent: int = Field(description="The percentage of it this Applicant type pays.")
    amount: Decimal


class FeeCheck(BaseModel):
    """The Fee Check result for one Permit Request."""

    declared_fee: Decimal
    expected_fee: Decimal | None = Field(
        description="The fee owed; None when the request lacks the data to calculate it."
    )
    charges: list[FeeCharge] = Field(description="They add up to the expected fee.")
    exemption_es: str | None = Field(
        description="The exemption applied; None when every charge is paid in full."
    )
    section_ids: list[str] = Field(description="The COBROS Sections the result rests on.")
    unverifiable_es: str | None = Field(
        default=None, description="Why the fee could not be calculated."
    )

    @computed_field
    @property
    def matches(self) -> bool | None:
        """None when the fee could not be calculated."""
        return None if self.expected_fee is None else self.declared_fee == self.expected_fee

    def problem_es(self) -> str | None:
        """What is wrong with the declared fee, as the text of a Note; None when it matches."""
        declared = money(self.declared_fee)
        if self.expected_fee is None:
            return (
                f"No se pudieron verificar los derechos declarados ({declared}): "
                f"{self.unverifiable_es}."
            )
        if self.matches:
            return None
        exemption = f", con la exención «{self.exemption_es}»" if self.exemption_es else ""
        return (
            f"Los derechos declarados ({declared}) no coinciden con los que corresponden según "
            f"la Orden Administrativa OGPe 2022-07{exemption}: {money(self.expected_fee)}."
        )


class _Owed(NamedTuple):
    """A charge the request owes, before the Applicant's exemption."""

    charge: _Charge
    section_id: str
    base: Decimal


class _Missing(NamedTuple):
    """Data the request lacks, without which one fee item cannot be calculated."""

    reason_es: str
    section_id: str


def money(amount: Decimal) -> str:
    return f"${amount:,.2f}"


def check_fee(request: PermitRequest, rules: FeeRules) -> FeeCheck:
    """Calculate the fee the request owes and compare it with the fee it declares."""
    exemption = rules.exemptions.applicant_types[request.applicant_type]
    if isinstance(request, PcocPermitRequest):
        percentages = exemption.percentages(request.works_by_government)
        owed, missing = _pcoc_charges(request, rules.pcoc)
    else:
        percentages = exemption.percentages(works_by_government=False)
        owed, missing = _permiso_unico_charges(request, rules.permiso_unico)
    if missing:
        return FeeCheck(
            declared_fee=request.declared_fee,
            expected_fee=None,
            charges=[],
            exemption_es=None,
            section_ids=list(dict.fromkeys(m.section_id for m in missing)),
            unverifiable_es="; ".join(m.reason_es for m in missing),
        )
    charges = [
        FeeCharge(
            ref=charge.ref,
            section_id=section_id,
            concept_es=charge.concept_es,
            base=_cents(base),
            percent=percentages[charge.kind],
            amount=_cents(base * percentages[charge.kind] / 100),
        )
        for charge, section_id, base in owed
    ]
    exempted = any(charge.percent != 100 for charge in charges)
    section_ids = list(dict.fromkeys(charge.section_id for charge in charges))
    return FeeCheck(
        declared_fee=request.declared_fee,
        expected_fee=sum((charge.amount for charge in charges), Decimal("0.00")),
        charges=charges,
        exemption_es=exemption.description_es if exempted else None,
        section_ids=section_ids + ([rules.exemptions.section_id] if exempted else []),
    )


def _pcoc_charges(
    request: PcocPermitRequest, rules: PcocRules
) -> tuple[list[_Owed], list[_Missing]]:
    if request.estimated_work_value is None:
        return [], [_Missing("no se declaró el valor estimado de la obra", rules.section_id)]
    units = (request.estimated_work_value / rules.value_unit).to_integral_value(ROUND_CEILING)
    return [
        _Owed(charge, rules.section_id, max(units * charge.rate, charge.minimum))
        for charge in rules.charges
    ], []


def _permiso_unico_charges(
    request: PermisoUnicoPermitRequest, rules: PermisoUnicoRules
) -> tuple[list[_Owed], list[_Missing]]:
    owed: list[_Owed] = []
    missing: list[_Missing] = []
    years = request.validity_years
    if "prevencion_incendios" in request.licenses_requested:
        fire = rules.prevencion_incendios
        if request.site_sketch is None:
            missing.append(_Missing("no se declaró el área del local (croquis)", fire.section_id))
        else:
            area_sqft = request.site_sketch.floor_area_m2 * SQFT_PER_M2
            if tier := next((t for t in fire.tiers if area_sqft <= t.up_to_sqft), None):
                owed.append(_Owed(fire, fire.section_id, tier.per_year * years))
            else:
                reason = (
                    f"el área del local ({area_sqft:,.0f} p.c.) excede los tramos de la tarifa "
                    "con que cuenta esta verificación"
                )
                missing.append(_Missing(reason, fire.section_id))
    if "sanitaria" in request.licenses_requested:
        sanitary = rules.licencia_sanitaria
        if request.establishment_type is None:
            reason = "no se declaró el tipo de establecimiento"
            missing.append(_Missing(reason, sanitary.section_id))
        else:
            yearly = sanitary.per_year[request.establishment_type]
            owed.append(_Owed(yearly, sanitary.section_id, yearly.rate * years))
    use = rules.uso
    owed.append(_Owed(use.base, use.section_id, use.base.amount))
    if "bebidas_alcoholicas" in request.licenses_requested:
        alcohol = use.bebidas_alcoholicas
        owed.append(_Owed(alcohol, use.section_id, alcohol.amount))
    return owed, missing


def _cents(amount: Decimal) -> Decimal:
    return amount.quantize(_CENT, ROUND_HALF_UP)


def load_fee_rules(path: Traversable | Path | None = None) -> FeeRules:
    """The curated fee rules; by default those of COBROS OA-2022-07."""
    path = path or files("case_mgmt_ai_demo.fixtures").joinpath("fee_rules.yaml")
    return FeeRules.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
