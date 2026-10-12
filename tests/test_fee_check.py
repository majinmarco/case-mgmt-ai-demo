from decimal import Decimal

import pytest

from case_mgmt_ai_demo.fee_check import check_fee, load_fee_rules
from case_mgmt_ai_demo.permit_request import parse_permit_request
from test_review import GOOD_PCOC, GOOD_PERMISO_UNICO

RULES = load_fee_rules()
SKETCH = {
    "floor_area_m2": 120,
    "shows_internal_layout": True,
    "shows_access_road": True,
    "shows_parking": True,
    "shows_dimensions": True,
}
# A cafeteria of 120 m² (1,292 p.c.) asking for both licences for 3 years.
CAFETERIA = GOOD_PERMISO_UNICO | {"establishment_type": "cafeteria", "site_sketch": SKETCH}


def fee_check(raw, **overrides):
    return check_fee(parse_permit_request(raw | overrides), RULES)


def amounts(check) -> dict[str, Decimal]:
    return {charge.ref: charge.amount for charge in check.charges}


def test_pcoc_fee_is_charged_per_thousand_of_work_value_with_the_filing_minimum():
    # $185,000: filing 185 x $0.35 = $64.75, below the $100 minimum.
    check = fee_check(GOOD_PCOC, estimated_work_value=185000, declared_fee=1210)

    assert amounts(check) == {
        "6(a)": Decimal("100.00"),
        "6(b)": Decimal("925.00"),
        "6(c)": Decimal("185.00"),
    }
    assert check.expected_fee == Decimal("1210.00")
    assert check.matches is True
    assert check.exemption_es is None
    assert check.problem_es() is None


def test_pcoc_filing_charge_above_the_minimum_is_charged_by_value():
    check = fee_check(GOOD_PCOC, estimated_work_value=2500000)

    assert amounts(check)["6(a)"] == Decimal("875.00")
    assert check.expected_fee == Decimal("15875.00")


def test_a_fraction_of_a_thousand_of_work_value_is_charged_as_a_whole_thousand():
    check = fee_check(GOOD_PCOC, estimated_work_value=185500)

    assert amounts(check) == {
        "6(a)": Decimal("100.00"),
        "6(b)": Decimal("930.00"),
        "6(c)": Decimal("186.00"),
    }


def test_a_wrong_declared_fee_is_a_mismatch_citing_the_anejo_a_item():
    check = fee_check(GOOD_PCOC, declared_fee=1850)

    assert check.matches is False
    assert check.declared_fee == Decimal("1850")
    assert check.expected_fee == Decimal("1210.00")
    assert check.section_ids == ["COBROS:AnejoA:6"]
    problem = check.problem_es()
    assert problem is not None
    assert "$1,850.00" in problem
    assert "$1,210.00" in problem


def test_permiso_unico_fee_adds_the_use_and_each_licence_per_year_requested():
    check = fee_check(CAFETERIA, declared_fee=535)

    assert amounts(check) == {
        "28(a)": Decimal("210.00"),  # 1,001-3,000 p.c.: $70 x 3 years
        "30(c)": Decimal("225.00"),  # cafetería: $75 x 3 years
        "41(a)": Decimal("100.00"),
    }
    assert check.expected_fee == Decimal("535.00")
    assert check.matches is True
    assert check.section_ids == ["COBROS:AnejoA:28", "COBROS:AnejoA:30", "COBROS:AnejoA:41"]


def test_only_the_licences_requested_are_charged():
    check = fee_check(CAFETERIA, licenses_requested=["sanitaria"], validity_years=1)

    assert amounts(check) == {"30(c)": Decimal("75.00"), "41(a)": Decimal("100.00")}


def test_selling_alcohol_adds_its_use_charge():
    check = fee_check(CAFETERIA, licenses_requested=["bebidas_alcoholicas"])

    assert amounts(check) == {"41(a)": Decimal("100.00"), "41(b)": Decimal("150.00")}


@pytest.mark.parametrize(
    ("floor_area_m2", "per_year"),
    [(92, 60), (93, 70), (278, 70), (279, 80), (1858, 90), (9290, 110)],
)
def test_fire_prevention_rate_follows_the_floor_area_in_square_feet(floor_area_m2, per_year):
    check = fee_check(
        CAFETERIA,
        site_sketch=SKETCH | {"floor_area_m2": floor_area_m2},
        licenses_requested=["prevencion_incendios"],
        validity_years=1,
    )

    assert amounts(check)["28(a)"] == Decimal(per_year)


def test_a_cooperative_is_exempt_from_every_charge():
    check = fee_check(GOOD_PCOC, applicant_type="cooperativa", declared_fee=0)

    assert check.expected_fee == Decimal("0.00")
    assert check.matches is True
    assert check.exemption_es is not None
    assert "ooperativa" in check.exemption_es
    assert {charge.percent for charge in check.charges} == {0}


def test_government_pays_a_quarter_of_services_and_all_of_tariffs_and_stamps():
    check = fee_check(GOOD_PCOC, applicant_type="gobierno")

    assert amounts(check) == {
        "6(a)": Decimal("25.00"),
        "6(b)": Decimal("925.00"),
        "6(c)": Decimal("185.00"),
    }
    assert check.exemption_es is not None


def test_works_carried_out_wholly_by_government_pay_a_quarter_of_tariffs_and_no_stamps():
    check = fee_check(GOOD_PCOC, applicant_type="gobierno", works_by_government=True)

    assert amounts(check) == {
        "6(a)": Decimal("25.00"),
        "6(b)": Decimal("231.25"),
        "6(c)": Decimal("0.00"),
    }
    assert check.expected_fee == Decimal("256.25")


def test_government_pays_a_quarter_of_permiso_unico_use_and_licences():
    check = fee_check(CAFETERIA, applicant_type="gobierno")

    assert amounts(check) == {
        "28(a)": Decimal("52.50"),
        "30(c)": Decimal("56.25"),
        "41(a)": Decimal("25.00"),
    }


def test_a_mismatch_under_an_exemption_also_cites_the_exemption_table():
    check = fee_check(GOOD_PCOC, applicant_type="cooperativa", declared_fee=1210)

    assert check.matches is False
    assert check.section_ids == ["COBROS:AnejoA:6", "COBROS:TERCERO"]


def test_non_profit_and_private_applicants_pay_in_full():
    private = fee_check(GOOD_PCOC, applicant_type="privado")
    non_profit = fee_check(GOOD_PCOC, applicant_type="sin_fines_de_lucro")

    assert non_profit.expected_fee == private.expected_fee == Decimal("1210.00")
    assert non_profit.exemption_es is None


@pytest.mark.parametrize(
    ("raw", "overrides", "reason", "section_id"),
    [
        (GOOD_PCOC, {"estimated_work_value": None}, "valor estimado", "COBROS:AnejoA:6"),
        (CAFETERIA, {"establishment_type": None}, "establecimiento", "COBROS:AnejoA:30"),
        (CAFETERIA, {"site_sketch": None}, "área del local", "COBROS:AnejoA:28"),
        (
            CAFETERIA,
            {"site_sketch": SKETCH | {"floor_area_m2": 9291}},
            "área del local",
            "COBROS:AnejoA:28",
        ),
    ],
)
def test_a_fee_the_request_cannot_evidence_is_unverifiable(raw, overrides, reason, section_id):
    check = fee_check(raw, **overrides)

    assert check.expected_fee is None
    assert check.matches is None
    assert check.charges == []
    assert check.section_ids == [section_id]
    problem = check.problem_es()
    assert problem is not None
    assert reason in problem


def test_every_missing_datum_is_reported_with_its_item():
    check = fee_check(CAFETERIA, establishment_type=None, site_sketch=None)

    assert check.section_ids == ["COBROS:AnejoA:28", "COBROS:AnejoA:30"]
    problem = check.problem_es()
    assert problem is not None
    assert "área del local" in problem
    assert "establecimiento" in problem
