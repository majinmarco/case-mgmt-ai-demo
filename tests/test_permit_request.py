import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from case_mgmt_ai_demo.permit_request import PermitRequest
from test_review import GOOD_PCOC

EXAMPLES = Path(__file__).parent.parent / "examples"

DISTRICT = {
    "lot_area_m2": 450,
    "min_lot_area_m2": 300,
    "dwelling_units": 1,
    "max_dwelling_units": 1,
    "occupation_pct": 35,
    "max_occupation_pct": 50,
    "yards_m": {"front": 5, "side": 2, "rear": 3},
    "min_yards_m": {"front": 4, "side": 1.5, "rear": 3},
    "parking_spaces": 2,
    "required_parking_spaces": 2,
    "loading_spaces": 0,
    "required_loading_spaces": 0,
}
PLANS = {
    "professional": "ingeniero",
    "professional_name": "Ing. Carlos Torres Vega",
    "license_number": "12345",
    "drafted_by": None,
    "sheets_certified": True,
    "compliance_certification": True,
    "file_format": "DXF",
    "north_oriented": True,
}
EVIDENCED_PCOC = GOOD_PCOC | {
    "work_value_method": "guias_ogpe",
    "district_parameters": DISTRICT,
    "plans_certification": PLANS,
    "catastro_matches_ogpe": True,
}


def test_value_certification_method_is_shown_with_the_value():
    fields = PermitRequest.model_validate(EVIDENCED_PCOC).fields_es()

    assert fields["Valor estimado de la obra"] == "$185,000, calculado según las guías de la OGPe"


def test_district_parameters_show_each_value_against_the_district_limit():
    fields = PermitRequest.model_validate(EVIDENCED_PCOC).fields_es()

    assert fields["Parámetros del distrito"] == (
        "tamaño del solar 450 m² (mínimo 300 m²); densidad 1 unidad(es) de vivienda (máximo 1); "
        "área de ocupación 35% (máximo 50%); patios delantero 5 m / laterales 2 m / posterior 3 m "
        "(mínimos 4 m / 1.5 m / 3 m); espacios de estacionamiento 2 (requeridos 2); "
        "espacios de carga y descarga 0 (requeridos 0)"
    )


def test_plans_certification_names_the_professional_and_the_file_details():
    fields = PermitRequest.model_validate(EVIDENCED_PCOC).fields_es()

    assert fields["Certificación de los planos"] == (
        "sellados y firmados por ingeniero licenciado Ing. Carlos Torres Vega (licencia 12345), "
        "quien también los delineó; cada hoja incluye las certificaciones específicas requeridas "
        "por el Reglamento; se acompaña el documento de certificación del proyectista de "
        "cumplimiento con el Reglamento Conjunto y los códigos de construcción vigentes; "
        "archivo digital DXF orientado al norte"
    )


def test_plans_drafted_by_another_professional_name_them():
    plans = PLANS | {"drafted_by": "Arq. Ana Díaz (licencia 678)"}
    fields = PermitRequest.model_validate(
        EVIDENCED_PCOC | {"plans_certification": plans}
    ).fields_es()

    assert "delineados por Arq. Ana Díaz (licencia 678);" in fields["Certificación de los planos"]


@pytest.mark.parametrize(
    ("matches", "shown"),
    [
        (
            True,
            "Caguas, 254-087-123-45 (la información catastral concuerda con el sistema de la OGPe)",
        ),
        (
            False,
            "Caguas, 254-087-123-45 (la información catastral no concuerda con el sistema de la OGPe)",
        ),
        (None, "Caguas, 254-087-123-45"),
    ],
)
def test_catastral_concordance_is_shown_with_the_catastro(matches, shown):
    fields = PermitRequest.model_validate(
        EVIDENCED_PCOC | {"catastro_matches_ogpe": matches}
    ).fields_es()

    assert fields["Municipio y número de catastro"] == shown


def test_plans_without_the_compliance_certification_or_north_orientation_say_so():
    plans = PLANS | {
        "sheets_certified": False,
        "compliance_certification": False,
        "north_oriented": False,
    }
    fields = PermitRequest.model_validate(
        EVIDENCED_PCOC | {"plans_certification": plans}
    ).fields_es()

    shown = fields["Certificación de los planos"]
    assert "no todas las hojas incluyen las certificaciones específicas" in shown
    assert "no se acompaña el documento de certificación del proyectista" in shown
    assert "no orientado al norte" in shown


def test_undeclared_evidence_fields_are_shown_as_not_declared():
    fields = PermitRequest.model_validate(GOOD_PCOC).fields_es()

    assert fields["Valor estimado de la obra"] == "$185,000, método de cálculo no declarado"
    assert fields["Parámetros del distrito"] == "no declarados"
    assert fields["Certificación de los planos"] == "no declarada"


@pytest.mark.parametrize(
    "change",
    [
        {"work_value_method": "a_ojo"},
        {"district_parameters": DISTRICT | {"lot_area_m2": 0}},
        {"district_parameters": DISTRICT | {"occupation_pct": 120}},
        {"plans_certification": PLANS | {"professional": "contable"}},
    ],
)
def test_evidence_fields_are_validated(change):
    with pytest.raises(ValidationError):
        PermitRequest.model_validate(EVIDENCED_PCOC | change)


@pytest.mark.parametrize("name", ["pcoc-buena", "pcoc-mala"])
def test_examples_are_valid_permit_requests(name):
    raw = json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))

    PermitRequest.model_validate(raw)


def test_good_example_evidences_value_method_district_parameters_and_plans():
    raw = json.loads((EXAMPLES / "pcoc-buena.json").read_text(encoding="utf-8"))

    fields = PermitRequest.model_validate(raw).fields_es()

    assert "guías de la OGPe" in fields["Valor estimado de la obra"]
    assert fields["Parámetros del distrito"] != "no declarados"
    assert "se acompaña el documento de certificación" in fields["Certificación de los planos"]
    assert "concuerda con el sistema de la OGPe" in fields["Municipio y número de catastro"]
