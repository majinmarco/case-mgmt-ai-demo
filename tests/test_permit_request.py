import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from case_mgmt_ai_demo.permit_request import parse_permit_request
from test_review import GOOD_PCOC, GOOD_PERMISO_UNICO

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
    fields = parse_permit_request(EVIDENCED_PCOC).fields_es()

    assert fields["Valor estimado de la obra"] == "$185,000, calculado según las guías de la OGPe"


def test_district_parameters_show_each_value_against_the_district_limit():
    fields = parse_permit_request(EVIDENCED_PCOC).fields_es()

    assert fields["Parámetros del distrito"] == (
        "tamaño del solar 450 m² (mínimo 300 m²); densidad 1 unidad(es) de vivienda (máximo 1); "
        "área de ocupación 35% (máximo 50%); patios delantero 5 m / laterales 2 m / posterior 3 m "
        "(mínimos 4 m / 1.5 m / 3 m); espacios de estacionamiento 2 (requeridos 2); "
        "espacios de carga y descarga 0 (requeridos 0)"
    )


def test_plans_certification_names_the_professional_and_the_file_details():
    fields = parse_permit_request(EVIDENCED_PCOC).fields_es()

    assert fields["Certificación de los planos"] == (
        "sellados y firmados por ingeniero licenciado Ing. Carlos Torres Vega (licencia 12345), "
        "quien también los delineó; cada hoja incluye las certificaciones específicas requeridas "
        "por el Reglamento; se acompaña el documento de certificación del proyectista de "
        "cumplimiento con el Reglamento Conjunto y los códigos de construcción vigentes; "
        "archivo digital DXF orientado al norte"
    )


def test_plans_drafted_by_another_professional_name_them():
    plans = PLANS | {"drafted_by": "Arq. Ana Díaz (licencia 678)"}
    fields = parse_permit_request(
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
    fields = parse_permit_request(
        EVIDENCED_PCOC | {"catastro_matches_ogpe": matches}
    ).fields_es()

    assert fields["Municipio y número de catastro"] == shown


def test_plans_without_the_compliance_certification_or_north_orientation_say_so():
    plans = PLANS | {
        "sheets_certified": False,
        "compliance_certification": False,
        "north_oriented": False,
    }
    fields = parse_permit_request(
        EVIDENCED_PCOC | {"plans_certification": plans}
    ).fields_es()

    shown = fields["Certificación de los planos"]
    assert "no todas las hojas incluyen las certificaciones específicas" in shown
    assert "no se acompaña el documento de certificación del proyectista" in shown
    assert "no orientado al norte" in shown


def test_undeclared_evidence_fields_are_shown_as_not_declared():
    fields = parse_permit_request(GOOD_PCOC).fields_es()

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
        parse_permit_request(EVIDENCED_PCOC | change)


@pytest.mark.parametrize(
    "name", ["pcoc-buena", "pcoc-mala", "permiso-unico-buena", "permiso-unico-mala"]
)
def test_examples_are_valid_permit_requests(name):
    raw = json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))

    parse_permit_request(raw)


SKETCH = {
    "floor_area_m2": 120,
    "shows_internal_layout": True,
    "shows_access_road": True,
    "shows_parking": True,
    "shows_dimensions": True,
}
RENEWAL = {
    "permit_number": "2023-512345-PU-001234",
    "expiry_date": "2026-03-31",
    "filing_date": "2026-01-15",
    "use_conforms_certification": True,
}


def test_request_type_picks_the_permiso_unico_fields():
    fields = parse_permit_request(GOOD_PERMISO_UNICO).fields_es()

    assert fields["Tipo de solicitud"] == "Permiso Único (solicitud nueva)"
    assert fields["Solicitante"] == "Ana Lucía Torres Rivera (persona privada)"
    assert fields["Uso propuesto"] == (
        "Cafetería con servicio de mesas, sin venta de bebidas alcohólicas "
        "(uso permitido ministerialmente en el distrito)"
    )
    assert fields["Empleados / estacionamiento / horario"] == (
        "6 empleados; 8 espacios de estacionamiento; lunes a sábado, de 6:00 a. m. a 3:00 p. m."
    )
    assert fields["Licencias solicitadas"] == (
        "licencia sanitaria; certificado de prevención de incendios; vigencia solicitada 3 años"
    )
    assert fields["Croquis de la instalación"] == "no declarado"
    assert "memorial explicativo" in fields["Documentos adjuntos"]
    assert fields["Derechos declarados"] == "$150"


def test_site_sketch_lists_what_it_shows():
    fields = parse_permit_request(GOOD_PERMISO_UNICO | {"site_sketch": SKETCH}).fields_es()

    assert fields["Croquis de la instalación"] == (
        "área del local 120 m²; incluye la distribución interna; incluye la ubicación con "
        "respecto a la vía de acceso; incluye el estacionamiento a proveer; incluye las dimensiones"
    )


def test_site_sketch_says_what_it_leaves_out():
    sketch = SKETCH | {"shows_internal_layout": False, "shows_dimensions": False}

    shown = parse_permit_request(GOOD_PERMISO_UNICO | {"site_sketch": sketch}).fields_es()[
        "Croquis de la instalación"
    ]

    assert "no incluye la distribución interna" in shown
    assert "no incluye las dimensiones" in shown


@pytest.mark.parametrize(
    ("renewal", "shown"),
    [
        (
            RENEWAL,
            "renovación del Permiso Único 2023-512345-PU-001234, que expira el 2026-03-31; "
            "presentada el 2026-01-15, 75 días antes de la expiración; se acompaña la "
            "certificación del dueño u operador de que el uso está conforme al Permiso Único",
        ),
        (
            RENEWAL
            | {"filing_date": "2028-08-31", "use_conforms_certification": False},
            "renovación del Permiso Único 2023-512345-PU-001234, que expira el 2026-03-31; "
            "presentada el 2028-08-31, 884 días (2.4 años) después de la expiración; no se "
            "acompaña la certificación del dueño u operador de que el uso está conforme al "
            "Permiso Único",
        ),
    ],
)
def test_a_renewal_shows_when_it_was_filed_against_the_expiry(renewal, shown):
    request = parse_permit_request(GOOD_PERMISO_UNICO | {"renewal": renewal})

    assert request.application_kind == "renovacion"
    assert request.fields_es()["Tipo de solicitud"] == "Permiso Único (renovación)"
    assert request.fields_es()["Renovación"] == shown


def test_a_request_without_a_renewal_is_new():
    assert parse_permit_request(GOOD_PERMISO_UNICO).application_kind == "nueva"
    assert parse_permit_request(GOOD_PCOC).application_kind == "nueva"
    assert "Renovación" not in parse_permit_request(GOOD_PERMISO_UNICO).fields_es()


@pytest.mark.parametrize(
    "change",
    [
        {"validity_years": 4},
        {"validity_years": 0},
        {"licenses_requested": ["armas"]},
        {"use_authorization": "por_favor"},
        {"attached_documents": ["planos_sellados"]},
        {"site_sketch": SKETCH | {"floor_area_m2": 0}},
        {"renewal": RENEWAL | {"expiry_date": "mañana"}},
        {"work_description": "una obra"},
    ],
)
def test_permiso_unico_fields_are_validated(change):
    with pytest.raises(ValidationError):
        parse_permit_request(GOOD_PERMISO_UNICO | change)


def test_good_example_evidences_value_method_district_parameters_and_plans():
    raw = json.loads((EXAMPLES / "pcoc-buena.json").read_text(encoding="utf-8"))

    fields = parse_permit_request(raw).fields_es()

    assert "guías de la OGPe" in fields["Valor estimado de la obra"]
    assert fields["Parámetros del distrito"] != "no declarados"
    assert "se acompaña el documento de certificación" in fields["Certificación de los planos"]
    assert "concuerda con el sistema de la OGPe" in fields["Municipio y número de catastro"]


@pytest.mark.parametrize(
    ("authorized", "shown"),
    [
        (True, "arrendatario (autorizado por el dueño para la acción propuesta)"),
        (False, "arrendatario (sin autorización del dueño para la acción propuesta)"),
        (None, "arrendatario"),
    ],
)
def test_owner_authorization_is_shown_with_the_applicant_role(authorized, shown):
    raw = GOOD_PERMISO_UNICO | {"owner_authorization": authorized}

    assert parse_permit_request(raw).fields_es()["Rol del solicitante"] == shown


def test_parking_is_shown_against_the_district_requirement():
    fields = parse_permit_request(
        GOOD_PERMISO_UNICO | {"required_parking_spaces": 6}
    ).fields_es()

    assert fields["Empleados / estacionamiento / horario"] == (
        "6 empleados; 8 espacios de estacionamiento (requeridos por el distrito 6); "
        "lunes a sábado, de 6:00 a. m. a 3:00 p. m."
    )


@pytest.mark.parametrize(
    ("flood_zone", "shown"),
    [
        (False, "el predio no ubica en zona inundable"),
        (True, "el predio ubica en zona inundable"),
        (None, "no declarada"),
    ],
)
def test_flood_zone_is_shown(flood_zone, shown):
    fields = parse_permit_request(GOOD_PERMISO_UNICO | {"in_flood_zone": flood_zone}).fields_es()

    assert fields["Zona inundable"] == shown


def test_a_permiso_unico_can_attach_the_site_documents_of_2_1_9_2():
    raw = GOOD_PERMISO_UNICO | {"attached_documents": ["foto_aerea", "ubicacion_geografica"]}

    assert parse_permit_request(raw).fields_es()["Documentos adjuntos"] == (
        "foto aérea del predio; ubicación geográfica del predio"
    )


def test_good_permiso_unico_example_evidences_standing_parking_flood_zone_and_site():
    raw = json.loads((EXAMPLES / "permiso-unico-buena.json").read_text(encoding="utf-8"))

    fields = parse_permit_request(raw).fields_es()

    assert "autorizado por el dueño" in fields["Rol del solicitante"]
    assert "requeridos por el distrito" in fields["Empleados / estacionamiento / horario"]
    assert fields["Zona inundable"] == "el predio no ubica en zona inundable"
    assert "foto aérea del predio" in fields["Documentos adjuntos"]
    assert "ubicación geográfica del predio" in fields["Documentos adjuntos"]
    assert "área del local" in fields["Croquis de la instalación"]


@pytest.mark.parametrize("raw", [GOOD_PCOC, GOOD_PERMISO_UNICO | {"renewal": RENEWAL}])
def test_the_form_describes_every_field_a_request_shows(raw):
    request = parse_permit_request(raw)

    assert request.fields_es().keys() <= request.form_es().keys()
    assert all(request.form_es().values())
