"""The new Permit Request form over HTTP, with the review pipeline faked."""

import re
from typing import get_args

import pytest
from pydantic import BaseModel

from case_mgmt_ai_demo.permit_request import (
    PcocPermitRequest,
    PermisoUnicoPermitRequest,
    parse_permit_request,
)
from case_mgmt_ai_demo.review import MissingPinnedClause
from test_cli import FakePipeline
from test_review import GOOD_PCOC, GOOD_PERMISO_UNICO, RENEWAL
from test_web import Site, site, text_of  # noqa: F401 (site is a fixture)

NEW = "/solicitudes/nueva"


def field_names(html: str) -> set[str]:
    return set(re.findall(r'<(?:input|select|textarea)[^>]* name="([^"]+)"', html))


def field_paths(model: type[BaseModel], prefix: str = "") -> set[str]:
    """Every field of a model as its dotted path, down into the models it nests."""
    paths: set[str] = set()
    for name, field in model.model_fields.items():
        nested = [
            option
            for option in (field.annotation, *get_args(field.annotation))
            if isinstance(option, type) and issubclass(option, BaseModel)
        ]
        if nested:
            paths |= field_paths(nested[0], f"{prefix}{name}.")
        else:
            paths.add(prefix + name)
    return paths


def test_the_home_page_links_to_the_new_permit_request_form(site):
    assert f'href="{NEW}"' in site.client.get("/").text


def test_the_form_opens_on_the_fields_and_documents_of_a_pcoc(site):
    response = site.client.get(NEW)

    assert response.status_code == 200
    page = text_of(response.text)
    assert "Nueva Solicitud de Permiso" in page
    assert "Número de catastro" in page
    assert "Descripción de la obra" in page
    assert "planos sellados y firmados por profesional licenciado" in page
    assert "Uso propuesto" not in page
    assert {"applicant_name", "catastro", "work_description"} <= field_names(response.text)


def test_choosing_a_request_type_swaps_in_its_fields_without_reloading_the_page(site):
    form = site.client.get(NEW).text

    [select] = re.findall(r'<select[^>]* name="request_type"[^>]*>', form)
    assert 'hx-get="/solicitudes/nueva/campos"' in select
    assert 'hx-target="#campos-tipo"' in select
    assert '<div id="campos-tipo">' in form

    response = site.client.get("/solicitudes/nueva/campos", params={"request_type": "PERMISO_UNICO"})

    assert response.status_code == 200
    assert "<html" not in response.text  # the fields HTMX swaps in, not a page
    fields = text_of(response.text)
    assert "Uso propuesto" in fields
    assert "certificado de antecedentes penales" in fields
    assert "Descripción de la obra" not in fields
    # The fields every Request Type shares stay in the page, as typed.
    assert "applicant_name" not in field_names(response.text)


def test_switching_request_type_keeps_what_both_types_declare(site):
    assert 'hx-include="#campos-tipo"' in site.client.get(NEW).text

    typed = {"request_type": "PERMISO_UNICO", "declared_fee": "150", "work_description": "Obra"}
    html = site.client.get("/solicitudes/nueva/campos", params=typed).text

    assert re.search(r'<input[^>]* name="declared_fee" value="150"', html)
    assert "Obra" not in html


def test_an_unknown_request_type_has_no_fields(site):
    response = site.client.get("/solicitudes/nueva/campos", params={"request_type": "OTRO"})

    assert response.status_code == 422


@pytest.mark.parametrize(
    "model", [PcocPermitRequest, PermisoUnicoPermitRequest], ids=["PCOC", "PERMISO_UNICO"]
)
def test_the_form_has_a_field_for_every_field_of_the_request_type(site, model):
    [request_type] = get_args(model.model_fields["request_type"].annotation)

    html = site.client.get(NEW, params={"request_type": request_type}).text

    assert field_names(html) == field_paths(model)
    # Every option is shown under its Spanish label.
    assert not re.search(r"<option[^>]*>\s*</option>", html)
    assert not re.search(r'<input type="checkbox"[^>]*>\s*</label>', html)


def form_of(raw, prefix="") -> dict[str, str | list[str]]:
    """A Permit Request as a browser submits the form: every value a string, a nested field
    under its dotted path, a field without a value left out."""
    form: dict[str, str | list[str]] = {}
    for name, value in raw.items():
        if isinstance(value, dict):
            form |= form_of(value, f"{prefix}{name}.")
        elif isinstance(value, list):
            form[prefix + name] = value
        elif isinstance(value, bool):
            form[prefix + name] = "true" if value else "false"
        elif value is not None:
            form[prefix + name] = str(value)
    return form


def test_submitting_a_pcoc_stores_it_reviews_it_and_opens_its_case_page(site):
    response = site.client.post("/solicitudes", data=form_of(GOOD_PCOC), follow_redirects=False)

    assert response.status_code == 303
    [stored] = site.store.all()
    assert response.headers["location"] == f"/solicitudes/{stored.id}"
    assert stored.request == parse_permit_request(GOOD_PCOC)
    assert site.pipeline.reviewed == [stored.request.model_dump(mode="json")]
    page = text_of(site.client.get(response.headers["location"]).text)
    assert "María Isabel Rivera Ortiz (persona privada)" in page
    assert "Puntuación de aprobación 0.86" in page


def test_submitting_a_permiso_unico_renewal_stores_its_nested_fields(site):
    sketch = {
        "floor_area_m2": 140,
        "shows_internal_layout": True,
        "shows_access_road": True,
        "shows_parking": False,
        "shows_dimensions": True,
    }
    raw = GOOD_PERMISO_UNICO | {"renewal": RENEWAL, "site_sketch": sketch, "in_flood_zone": False}

    response = site.client.post("/solicitudes", data=form_of(raw), follow_redirects=False)

    assert response.status_code == 303
    [stored] = site.store.all()
    assert stored.request == parse_permit_request(raw)
    page = text_of(site.client.get(response.headers["location"]).text)
    assert "Permiso Único (renovación)" in page
    assert "no incluye el estacionamiento a proveer" in page
    assert "Puntuación de aprobación 0.86" in page


def test_a_field_left_blank_is_not_declared(site):
    blank = {"standing_document": "", "estimated_work_value": " ", "plans_certification.drafted_by": ""}

    site.client.post("/solicitudes", data=form_of(GOOD_PCOC) | blank)

    [stored] = site.store.all()
    assert stored.request.standing_document is None
    assert stored.request.estimated_work_value is None
    assert stored.request.plans_certification is None


def test_a_submission_from_htmx_is_sent_to_the_case_page(site):
    response = site.client.post(
        "/solicitudes", data=form_of(GOOD_PCOC), headers={"HX-Request": "true"}
    )

    [stored] = site.store.all()
    assert response.status_code == 204
    assert response.headers["hx-redirect"] == f"/solicitudes/{stored.id}"


def field_block(html: str, name: str) -> str:
    """What a reader sees around one field: its label, and any error under it."""
    [block] = [
        block
        for block in re.findall(r'<div class="campo[^"]*">.*?</div>', html, re.DOTALL)
        if f'name="{name}"' in block
    ]
    return text_of(block)


def test_an_invalid_submission_shows_each_error_under_its_field_and_stores_nothing(site):
    invalid = form_of(GOOD_PCOC) | {"catastro": "254-087", "applicant_name": "", "floors": "dos"}

    response = site.client.post("/solicitudes", data=invalid)

    assert response.status_code == 422
    assert "La solicitud no es válida" in text_of(response.text)
    # The messages of pipeline intake, each with the field it is about.
    assert "Número de catastro" in field_block(response.text, "catastro")
    assert "formato no válido" in field_block(response.text, "catastro")
    assert "campo obligatorio" in field_block(response.text, "applicant_name")
    assert "debe ser un número entero" in field_block(response.text, "floors")
    assert "campo obligatorio" not in field_block(response.text, "municipio")
    assert site.store.all() == []
    assert site.pipeline.reviewed == []


def test_an_invalid_submission_keeps_what_was_entered(site):
    invalid = form_of(GOOD_PERMISO_UNICO) | {"validity_years": "5"}

    html = site.client.post("/solicitudes", data=invalid).text

    assert "debe ser menor o igual que 3" in field_block(html, "validity_years")
    assert re.search(r'<option value="PERMISO_UNICO" selected>', html)
    assert re.search(r'<input[^>]* name="municipio" value="Ponce"', html)
    assert re.search(r'<input[^>]* name="validity_years" value="5"', html)
    assert re.search(r'<option value="arrendatario" selected>', html)
    assert re.search(r'name="attached_documents" value="croquis" checked', html)
    assert not re.search(r'name="attached_documents" value="patente_municipal" checked', html)


def test_an_error_inside_a_group_is_shown_under_its_field(site):
    invalid = form_of(GOOD_PERMISO_UNICO | {"renewal": RENEWAL}) | {"renewal.expiry_date": ""}

    html = site.client.post("/solicitudes", data=invalid).text

    assert "campo obligatorio" in field_block(html, "renewal.expiry_date")
    assert "campo obligatorio" not in field_block(html, "renewal.filing_date")


def test_a_group_missing_a_nested_field_shows_the_error_under_each_of_its_inputs(site):
    parameters = {"district_parameters.lot_area_m2": "450"}

    html = site.client.post("/solicitudes", data=form_of(GOOD_PCOC) | parameters).text

    assert "campo obligatorio" in field_block(html, "district_parameters.yards_m.front")
    assert "campo obligatorio" in field_block(html, "district_parameters.min_yards_m.rear")
    assert "campo obligatorio" not in field_block(html, "district_parameters.lot_area_m2")
    assert "yards_m" not in text_of(html)  # no error is left to the list above the form


def test_an_invalid_submission_from_htmx_gets_only_the_form_back(site):
    response = site.client.post(
        "/solicitudes", data=form_of(GOOD_PCOC) | {"catastro": "x"}, headers={"HX-Request": "true"}
    )

    assert response.status_code == 422
    assert "<html" not in response.text
    assert response.text.lstrip().startswith("<form")
    assert "formato no válido" in field_block(response.text, "catastro")


def test_a_submission_of_an_unknown_request_type_is_invalid(site):
    response = site.client.post("/solicitudes", data=form_of(GOOD_PCOC) | {"request_type": "OTRO"})

    assert response.status_code == 422
    assert "valor no permitido" in field_block(response.text, "request_type")
    assert site.store.all() == []


def test_the_form_shows_progress_while_the_submission_is_reviewed(site):
    html = site.client.get(NEW).text

    [form] = re.findall(r"<form[^>]*>", html)
    assert 'hx-post="/solicitudes"' in form
    assert 'hx-indicator="find .revisando"' in form
    assert 'hx-disabled-elt="find button"' in form  # no second submission while it runs
    [indicator] = re.findall(r'<span class="revisando htmx-indicator".*?</span>', html, re.DOTALL)
    assert "<progress" in indicator
    assert "Revisando la solicitud contra el Reglamento…" in text_of(indicator)


def test_a_review_that_fails_keeps_the_permit_request_and_says_why_on_its_case_page(tmp_path):
    site = Site(tmp_path, FakePipeline(MissingPinnedClause("pcoc_planos pins 2.1.9.2(k)")))

    response = site.client.post("/solicitudes", data=form_of(GOOD_PCOC), follow_redirects=False)

    [stored] = site.store.all()
    assert response.headers["location"] == f"/solicitudes/{stored.id}"
    page = text_of(site.client.get(response.headers["location"]).text)
    assert "No se pudo revisar la solicitud" in page
    assert "MissingPinnedClause: pcoc_planos pins 2.1.9.2(k)" in page
    assert "Aún no hay una Recomendación" in page
    # Said once: the case page opened again offers the review, without the old failure.
    assert "No se pudo revisar" not in text_of(site.client.get(f"/solicitudes/{stored.id}").text)
