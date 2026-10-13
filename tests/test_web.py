"""The Reviewer web UI over HTTP, with the review pipeline and the Reglamento index faked."""

import re
import sqlite3

import pytest
from fastapi.testclient import TestClient

from case_mgmt_ai_demo.chunks import Chunk
from case_mgmt_ai_demo.config import Settings
from case_mgmt_ai_demo.permit_request import parse_permit_request
from case_mgmt_ai_demo.review import (
    ClauseSelection,
    InvalidPermitRequest,
    MissingPinnedClause,
    ModelFailure,
    Note,
    PipelineTrace,
    RequirementAssessment,
    RetrievedSection,
    TraceStep,
)
from case_mgmt_ai_demo.store import PermitRequestStore
from case_mgmt_ai_demo.web import create_app
from test_cli import FakePipeline, criterion_requirement, fee_check, recommendation
from test_review import GOOD_PCOC, GOOD_PERMISO_UNICO, PCOC_PARAMETERS, PCOC_STANDING, PU_USE


def chunk(section_id: str, page: int, text: str, part: str | None = None) -> Chunk:
    cobros = section_id.startswith("COBROS:")
    return Chunk(
        section_id=section_id,
        doc="COBROS-OA-2022-07" if cobros else "RC-2023-E",
        level="anejo_item" if cobros else "seccion",
        heading="REQUISITOS",
        parents=[],
        page_start=page,
        page_end=page + 1,
        text=text,
        part=part,
        request_types=["PCOC", "PERMISO_UNICO"],
    )


# Pages as in the real index: Sección 2.1.9.2 is split, and its second part starts at (k).
CHUNKS = [
    chunk("RC-2023-E:2.1.9.2", 77, "Toda solicitud:\na. Uno.\ng. Siete.", part="1-of-2"),
    chunk("RC-2023-E:2.1.9.2", 78, "k. Planos.\nm. Subsanación.", part="2-of-2"),
    chunk("RC-2023-E:2.1.9.5", 80, "a. Legitimación.\nb. Documentos."),
    chunk("RC-2023-E:3.7.1.2", 249, "a. Solicitud.\nb. Documentos:\n  3. Croquis."),
    chunk("COBROS:AnejoA:6", 13, "a. Radicación."),
]


class FakeSections:
    """Stands in for the Reglamento index: Sections by ID, with their pages."""

    def section(self, section_id: str) -> list[Chunk]:
        return [c for c in CHUNKS if c.section_id == section_id]


class Site:
    """The app over a store in a temporary directory, as one run of `serve` sees it."""

    def __init__(self, tmp_path, pipeline=None):
        self.path = tmp_path / "app.sqlite"
        rc, cobros = tmp_path / "RC-2023-E.pdf", tmp_path / "COBROS OA-2022-07.pdf"
        rc.write_bytes(b"%PDF-1.7 reglamento")
        cobros.write_bytes(b"%PDF-1.7 cobros")
        self.settings = Settings(data_dir=tmp_path, reglamento_pdf=rc, cobros_pdf=cobros)
        self.pipeline = pipeline or FakePipeline(recommendation())
        self.store = PermitRequestStore(self.path)
        self.client = self.restart()

    def restart(self) -> TestClient:
        self.store = PermitRequestStore(self.path)
        self.client = TestClient(
            create_app(self.store, self.pipeline, FakeSections(), self.settings)
        )
        return self.client

    def add(self, raw, recommended=None) -> int:
        request_id = self.store.add(parse_permit_request(raw))
        if recommended is not None:
            self.store.save_recommendation(request_id, recommended)
        return request_id

    def page(self, raw=GOOD_PCOC, **overrides) -> str:
        """The case page of a Permit Request stored with a Recommendation."""
        request = parse_permit_request(raw)
        request_id = self.add(raw, recommendation(request=request, **overrides))
        response = self.client.get(f"/solicitudes/{request_id}")
        assert response.status_code == 200
        return response.text


@pytest.fixture
def site(tmp_path) -> Site:
    return Site(tmp_path)


def text_of(html: str) -> str:
    """The page as a reader sees it: tags dropped, whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def citation_href(html: str, citation: str) -> str | None:
    found = re.search(rf'<a class="cita" href="([^"]+)"[^>]*>\s*{re.escape(citation)}\s*</a>', html)
    return found.group(1) if found else None


def test_case_page_shows_the_fields_of_a_pcoc_permit_request(site):
    page = text_of(site.page(GOOD_PCOC))

    assert "Permiso de Construcción (PCOC)" in page
    assert "María Isabel Rivera Ortiz (persona privada)" in page
    assert "Descripción de la obra" in page
    assert "Certificación de los planos" in page


def test_case_page_shows_the_fields_of_a_permiso_unico_permit_request(site):
    requirements = [criterion_requirement(PU_USE, 0.93)]

    page = text_of(site.page(GOOD_PERMISO_UNICO, requirements=requirements, approval_score=0.93))

    assert "Permiso Único (solicitud nueva)" in page
    assert "Uso propuesto" in page
    assert "Croquis de la instalación" in page
    assert PU_USE.text_es in page
    assert "Puntuación de aprobación 0.93" in page


def test_case_page_shows_the_approval_score_and_verdict(site):
    html = site.page()

    page = text_of(html)
    assert "Puntuación de aprobación 0.86" in page
    assert "Aprobar" in page
    assert "Caso límite" not in page
    assert 'class="medidor"' in html


def test_a_borderline_recommendation_is_flagged(site):
    page = text_of(site.page(approval_score=0.55, borderline=True))

    assert "Puntuación de aprobación 0.55" in page
    assert "Caso límite: revisar con detenimiento" in page


def test_a_recommendation_without_a_score_asks_for_manual_review(site):
    requirements = [
        criterion_requirement(PCOC_STANDING, 0.91),
        criterion_requirement(PCOC_PARAMETERS, None),
    ]

    page = text_of(
        site.page(approval_score=None, verdict="manual_review", requirements=requirements)
    )

    assert "Sin puntuación — revisión manual" in page
    assert "Puntuación de aprobación 0." not in page
    assert f"{PCOC_PARAMETERS.text_es} no evaluado" in page


def test_every_requirement_is_listed_with_its_probability_and_the_weakest_marked(site):
    html = site.page()

    page = text_of(html)
    assert f"{PCOC_STANDING.text_es} 0.91" in page
    assert f"{PCOC_PARAMETERS.text_es} 0.97" in page
    assert "Requisitos aplicables de la Sección 2.1.9.2 (REQUISITOS DE PRESENTACIÓN) 0.86" in page
    [weakest] = re.findall(r'<li class="requisito mas-debil">(.*?)</li>', html, re.DOTALL)
    assert "Sección 2.1.9.2" in weakest
    assert "Requisito más débil" in text_of(weakest)


def test_sections_that_do_not_apply_are_listed_with_their_reasons(site):
    page = text_of(site.page())

    assert "Secciones recuperadas que no aplican" in page
    assert "RC-2023-E:2.2.6 No se solicita una prórroga del permiso." in page


def test_notes_are_shown_and_a_curable_note_is_distinguished(site):
    html = site.page()

    notes = re.findall(r'<li class="nota[^"]*">.*?</li>', html, re.DOTALL)
    assert len(notes) == 2
    assert "La solicitante es dueña" in notes[0]
    assert "Subsanable" not in notes[0]
    assert "Falta la determinación de cumplimiento ambiental." in notes[1]
    assert 'class="nota subsanable"' in notes[1]
    assert "Subsanable" in text_of(notes[1])


def test_the_fee_check_shows_each_charge_when_the_fee_matches(site):
    check = fee_check(GOOD_PCOC | {"declared_fee": 1210})

    page = text_of(site.page(fee_check=check))

    assert "Verificación de derechos" in page
    assert "Los derechos declarados coinciden" in page
    assert "Declarados $1,210.00" in page
    assert "6(b) Aranceles $925.00" in page


def test_a_fee_check_mismatch_is_flagged_with_both_amounts(site):
    html = site.page(fee_check=fee_check(GOOD_PCOC))

    page = text_of(html)
    assert "Los derechos declarados no coinciden" in page
    assert "Declarados $1,850.00" in page
    assert "Calculados $1,210.00" in page
    assert citation_href(html, "COBROS:AnejoA:6") == "/reglamento/COBROS%20OA-2022-07.pdf#page=13"


def test_a_fee_that_could_not_be_verified_says_why(site):
    check = fee_check(GOOD_PCOC | {"estimated_work_value": None})

    page = text_of(site.page(fee_check=check))

    assert "No se pudo verificar: no se declaró el valor estimado de la obra" in page
    assert "Calculados" not in page


def test_the_exemption_applied_is_shown(site):
    check = fee_check(GOOD_PCOC | {"applicant_type": "gobierno"})

    page = text_of(site.page(fee_check=check))

    assert "Exención aplicada: Gobierno estatal, federal o municipal" in page
    assert "25% de $100.00" in page


def test_a_citation_links_to_the_pdf_on_the_page_of_its_section(site):
    requirement = RequirementAssessment(
        id="seccion_2_1_9_2",
        text_es="Requisitos aplicables de la Sección 2.1.9.2",
        clause_refs=["RC-2023-E:2.1.9.2"],
        probability=0.86,
    )

    html = site.page(requirements=[requirement])

    assert citation_href(html, "RC-2023-E:2.1.9.2") == "/reglamento/RC-2023-E.pdf#page=77"
    assert citation_href(html, "RC-2023-E:2.1.9.5(b)") == "/reglamento/RC-2023-E.pdf#page=80"


def test_a_clause_citation_of_a_split_section_links_to_the_part_holding_the_clause(site):
    html = site.page()

    # (g) is in the first part of Sección 2.1.9.2, (k) in the second.
    assert citation_href(html, "RC-2023-E:2.1.9.2(g)") == "/reglamento/RC-2023-E.pdf#page=77"
    assert citation_href(html, "RC-2023-E:2.1.9.2(k)") == "/reglamento/RC-2023-E.pdf#page=78"


def test_a_citation_of_a_sub_item_links_to_the_page_of_its_clause(site):
    note = Note(text="Falta el croquis.", section_ids=["RC-2023-E:3.7.1.2(b)(3)"], curable=True)

    html = site.page(notes=[note])

    assert citation_href(html, "RC-2023-E:3.7.1.2(b)(3)") == "/reglamento/RC-2023-E.pdf#page=249"


def test_a_citation_is_plain_text_when_its_pdf_is_not_there(site):
    site.settings.reglamento_pdf.unlink()

    html = site.page()

    assert "RC-2023-E:2.1.9.5(b)" in text_of(html)
    assert citation_href(html, "RC-2023-E:2.1.9.5(b)") is None


def test_a_citation_of_a_section_missing_from_the_index_is_plain_text(site):
    html = site.page()

    assert "RC-2023-E:2.2.6" in text_of(html)
    assert citation_href(html, "RC-2023-E:2.2.6") is None
    assert not re.search(r'href="[^"]*"[^>]*>\s*RC-2023-E:2\.2\.6', html)


def test_the_pdf_opens_in_a_side_panel_with_a_link_to_a_new_tab(site):
    html = site.page()

    assert '<aside id="visor"' in html
    assert re.search(r'<a id="visor-pestana"[^>]*target="_blank"', html)
    assert '<script src="/static/app.js"' in html


def test_the_reglamento_pdfs_are_served(site):
    reglamento = site.client.get("/reglamento/RC-2023-E.pdf")
    cobros = site.client.get("/reglamento/COBROS%20OA-2022-07.pdf")

    assert reglamento.status_code == cobros.status_code == 200
    assert reglamento.headers["content-type"] == "application/pdf"
    assert "content-disposition" not in reglamento.headers  # shown by the browser, not saved
    assert reglamento.content == b"%PDF-1.7 reglamento"
    assert cobros.content == b"%PDF-1.7 cobros"


def test_only_the_reglamento_pdfs_are_served(site):
    assert site.client.get("/reglamento/app.sqlite").status_code == 404
    assert site.client.get("/reglamento/..%2Fapp.sqlite").status_code == 404


def test_the_pipeline_trace_is_collapsible_and_holds_what_each_node_returned(site):
    trace = PipelineTrace(
        retrieved=[
            RetrievedSection(
                section_id="RC-2023-E:2.1.9.2",
                heading="REQUISITOS DE PRESENTACIÓN",
                similarity=0.612,
                pinned_by=["pcoc_planos"],
                clause_refs=["RC-2023-E:2.1.9.2(k)"],
            ),
            RetrievedSection(
                section_id="RC-2023-E:2.2.6", heading="PRÓRROGAS", similarity=None, pinned_by=[]
            ),
        ],
        steps=[TraceStep(node="decide", output={"approval_score": 0.86})],
    )

    html = site.page(trace=trace)

    [details] = re.findall(r'<details class="traza">(.*?)</details>\s*</section>', html, re.DOTALL)
    assert "<summary>" in details
    shown = text_of(details)
    assert "0.612 RC-2023-E:2.1.9.2 REQUISITOS DE PRESENTACIÓN" in shown
    assert "fijada por pcoc_planos" in shown
    assert "no aplica: No se solicita una prórroga del permiso." in shown
    assert "trámite del sistema RC-2023-E:2.1.9.2(a)" in shown
    assert "decide" in shown and "&#34;approval_score&#34;: 0.86" in details


def test_failed_model_calls_are_shown(site):
    errors = [ModelFailure(step="decide", subject="pcoc_legitimacion", message="gpt: timeout")]

    page = text_of(site.page(verdict="manual_review", errors=errors))

    assert "Revisión manual" in page
    assert "decide [pcoc_legitimacion]: gpt: timeout" in page


def test_a_stored_recommendation_is_still_there_after_a_restart(site):
    request_id = site.add(GOOD_PCOC, recommendation())

    client = site.restart()

    page = text_of(client.get(f"/solicitudes/{request_id}").text)
    assert "María Isabel Rivera Ortiz" in page
    assert "Puntuación de aprobación 0.86" in page
    assert site.pipeline.reviewed == []


def test_a_permit_request_without_a_recommendation_offers_to_review_it(site):
    request_id = site.add(GOOD_PCOC)

    html = site.client.get(f"/solicitudes/{request_id}").text

    assert "Aún no hay una Recomendación" in text_of(html)
    assert f'hx-post="/solicitudes/{request_id}/recomendacion"' in html
    assert '<script src="/static/htmx.min.js"' in html
    assert site.client.get("/static/htmx.min.js").status_code == 200


def test_reviewing_stores_the_recommendation_and_returns_it(site):
    request_id = site.add(GOOD_PCOC)

    response = site.client.post(f"/solicitudes/{request_id}/recomendacion")

    assert response.status_code == 200
    assert "<html" not in response.text  # the fragment HTMX swaps in, not a page
    assert "Puntuación de aprobación 0.86" in text_of(response.text)
    assert [raw["applicant_name"] for raw in site.pipeline.reviewed] == [
        "María Isabel Rivera Ortiz"
    ]
    assert "Puntuación de aprobación 0.86" in text_of(
        site.restart().get(f"/solicitudes/{request_id}").text
    )


def test_a_review_that_fails_says_so_and_stores_nothing(tmp_path):
    site = Site(tmp_path, FakePipeline(InvalidPermitRequest(["catastro: formato no válido"])))
    request_id = site.add(GOOD_PCOC)

    response = site.client.post(f"/solicitudes/{request_id}/recomendacion")

    assert response.status_code == 200
    assert "No se pudo revisar la solicitud" in text_of(response.text)
    assert "catastro: formato no válido" in response.text
    assert site.store.recommendation(request_id) is None


def test_a_review_that_cannot_run_names_what_went_wrong(tmp_path):
    site = Site(tmp_path, FakePipeline(MissingPinnedClause("pcoc_planos pins 2.1.9.2(k)")))
    request_id = site.add(GOOD_PCOC)

    response = site.client.post(f"/solicitudes/{request_id}/recomendacion")

    assert "MissingPinnedClause: pcoc_planos pins 2.1.9.2(k)" in text_of(response.text)
    assert "Aún no hay una Recomendación" in text_of(response.text)


def test_reviewing_again_asks_before_replacing_the_stored_recommendation(site):
    html = site.page()

    assert re.search(r'<form class="revisar"[^>]*hx-confirm="¿Revisar de nuevo\?', html)


def test_an_unknown_permit_request_is_not_found(site):
    assert site.client.get("/solicitudes/41").status_code == 404
    assert site.client.post("/solicitudes/41/recomendacion").status_code == 404


def test_the_home_page_links_to_every_permit_request(site):
    first = site.add(GOOD_PCOC)
    second = site.add(GOOD_PERMISO_UNICO)

    html = site.client.get("/").text

    assert f'href="/solicitudes/{first}"' in html
    assert f'href="/solicitudes/{second}"' in html
    assert "María Isabel Rivera Ortiz" in html


UNVERIFIABLE = ClauseSelection(
    ref="RC-2023-E:2.1.9.2(g)",
    category="no_verificable",
    reason="El formulario no admite un estudio de suelos.",
)


def test_requirements_the_form_cannot_evidence_are_listed_after_the_notes(site):
    html = site.page(selection=[UNVERIFIABLE])

    page = text_of(html)
    listed = page.index("Requisitos que el formulario no permite acreditar")
    assert page.index("Falta la determinación de cumplimiento ambiental.") < listed
    assert "RC-2023-E:2.1.9.2(g) El formulario no admite un estudio de suelos." in page[listed:]
    assert citation_href(html, "RC-2023-E:2.1.9.2(g)") == "/reglamento/RC-2023-E.pdf#page=77"


def test_the_requirements_say_how_many_more_the_form_cannot_evidence(site):
    page = text_of(site.page(selection=[UNVERIFIABLE]))

    assert "1 requisito más no se puntúa: el formulario no permite acreditarlo" in page


def test_a_page_with_nothing_the_form_cannot_evidence_does_not_mention_it(site):
    assert "el formulario no permite acreditar" not in text_of(site.page())


def test_a_refused_requirement_shows_the_clauses_it_was_asked_about_and_why(site):
    refused = RequirementAssessment(
        id="seccion_2_1_9_2",
        text_es="Requisitos aplicables de la Sección 2.1.9.2",
        clause_refs=["RC-2023-E:2.1.9.2(k)"],
        probability=None,
    )

    page = text_of(
        site.page(requirements=[refused], approval_score=None, verdict="manual_review")
    )

    assert "La pregunta fue rechazada, sin razón dada. Se preguntó por:" in page
    assert "RC-2023-E:2.1.9.2(k) Los planos deben estar sellados y certificados." in page


def determine(site, request_id, **form):
    return site.client.post(
        f"/solicitudes/{request_id}/determinacion", data=form, follow_redirects=False
    )


def test_accepting_the_recommendation_records_the_determination_beside_it(site):
    request_id = site.add(GOOD_PCOC, recommendation())

    response = determine(site, request_id, kind="accept")

    assert response.status_code == 303
    assert response.headers["location"] == f"/solicitudes/{request_id}"
    page = text_of(site.client.get(f"/solicitudes/{request_id}").text)
    assert "Recomendación aceptada: Aprobar" in page
    assert page.index("Puntuación de aprobación") < page.index("Recomendación aceptada")
    assert page.index("Recomendación aceptada") < page.index("Requisitos")


def test_overriding_the_recommendation_records_the_opposite_outcome_and_the_reason(site):
    request_id = site.add(GOOD_PCOC, recommendation())

    response = determine(
        site, request_id, kind="override", reason="El plano no corresponde al solar."
    )

    assert response.status_code == 303
    page = text_of(site.client.get(f"/solicitudes/{request_id}").text)
    assert "Recomendación anulada: No aprobar" in page
    assert "Razón: El plano no corresponde al solar." in page


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_an_override_without_a_reason_is_rejected(site, reason):
    request_id = site.add(GOOD_PCOC, recommendation())
    form = {"kind": "override"} | ({} if reason is None else {"reason": reason})

    response = determine(site, request_id, **form)

    assert response.status_code == 422
    assert "Anular la Recomendación requiere una razón." in text_of(response.text)
    assert site.store.determination(request_id) is None
    assert "Recomendación anulada" not in text_of(
        site.client.get(f"/solicitudes/{request_id}").text
    )


def test_a_recommendation_without_a_verdict_is_determined_with_an_outcome_and_a_reason(site):
    manual = recommendation(approval_score=None, verdict="manual_review")
    request_id = site.add(GOOD_PCOC, manual)

    assert determine(site, request_id, kind="accept").status_code == 422
    assert determine(site, request_id, kind="override", reason="Cumple.").status_code == 422
    response = determine(
        site, request_id, kind="override", reason="Cumple.", outcome="approve"
    )

    assert response.status_code == 303
    page = text_of(site.client.get(f"/solicitudes/{request_id}").text)
    assert "Determinación sin veredicto recomendado: Aprobar" in page
    assert "Razón: Cumple." in page


def cure_request_draft(html: str) -> str:
    [draft] = re.findall(r'<textarea[^>]*name="deficiencies"[^>]*>(.*?)</textarea>', html, re.DOTALL)
    return draft


def test_the_cure_request_is_pre_filled_with_the_curable_notes(site):
    html = site.page()

    assert cure_request_draft(html).strip() == (
        "Falta la determinación de cumplimiento ambiental. (RC-2023-E:2.1.9.2(g))"
    )


def test_issuing_a_cure_request_records_the_deficiencies_listed(site):
    request_id = site.add(GOOD_PCOC, recommendation())
    listed = "Falta la determinación ambiental. (RC-2023-E:2.1.9.2(g))\r\n\r\n  Falta el plano.  "

    response = determine(site, request_id, kind="cure_request", deficiencies=listed)

    assert response.status_code == 303
    html = site.client.get(f"/solicitudes/{request_id}").text
    assert "Requerimiento de subsanación" in text_of(html)
    assert re.findall(r'<li class="deficiencia">\s*(.*?)\s*</li>', html) == [
        "Falta la determinación ambiental. (RC-2023-E:2.1.9.2(g))",
        "Falta el plano.",
    ]


def test_a_cure_request_listing_nothing_is_rejected(site):
    request_id = site.add(GOOD_PCOC, recommendation())

    response = determine(site, request_id, kind="cure_request", deficiencies=" \r\n ")

    assert response.status_code == 422
    assert "debe enumerar al menos una deficiencia" in text_of(response.text)
    assert site.store.determination(request_id) is None


def test_a_rejected_determination_keeps_the_reason_the_reviewer_wrote(site):
    manual = recommendation(approval_score=None, verdict="manual_review")
    request_id = site.add(GOOD_PCOC, manual)

    response = determine(site, request_id, kind="override", reason="Cumple con todo.")

    assert response.status_code == 422
    assert re.search(r'<textarea[^>]*name="reason"[^>]*>Cumple con todo.</textarea>', response.text)


def test_a_determination_is_still_there_after_a_restart(site):
    request_id = site.add(GOOD_PCOC, recommendation())
    determine(site, request_id, kind="override", reason="El plano no corresponde al solar.")

    client = site.restart()

    page = text_of(client.get(f"/solicitudes/{request_id}").text)
    assert "Recomendación anulada: No aprobar" in page
    assert "Razón: El plano no corresponde al solar." in page


def test_a_determination_is_final(site):
    request_id = site.add(GOOD_PCOC, recommendation())
    determine(site, request_id, kind="accept")

    again = determine(site, request_id, kind="override", reason="Cambio de parecer.")

    assert again.status_code == 409
    html = site.client.get(f"/solicitudes/{request_id}").text
    assert "Recomendación aceptada: Aprobar" in text_of(html)
    assert 'name="kind"' not in html


def test_a_determined_permit_request_is_not_reviewed_again(site):
    request_id = site.add(GOOD_PCOC, recommendation())
    determine(site, request_id, kind="accept")

    response = site.client.post(f"/solicitudes/{request_id}/recomendacion")

    assert response.status_code == 409
    assert site.pipeline.reviewed == []
    assert "Revisar de nuevo" not in site.client.get(f"/solicitudes/{request_id}").text


def test_a_permit_request_without_a_recommendation_cannot_be_determined(site):
    request_id = site.add(GOOD_PCOC)

    assert 'name="kind"' not in site.client.get(f"/solicitudes/{request_id}").text
    assert determine(site, request_id, kind="accept").status_code == 409
    assert determine(site, 41, kind="accept").status_code == 404


def test_an_unknown_kind_of_determination_is_rejected(site):
    request_id = site.add(GOOD_PCOC, recommendation())

    assert determine(site, request_id, kind="deny").status_code == 422
    assert determine(site, request_id).status_code == 422


def queue_rows(site) -> list[str]:
    """Each Permit Request of the queue, as its row reads."""
    html = site.client.get("/").text
    return [text_of(row).strip() for row in re.findall(r"<tr>(.*?)</tr>", html, re.DOTALL)[1:]]


def test_the_queue_lists_each_permit_request_with_its_recommendation(site):
    site.add(GOOD_PCOC, recommendation())
    site.add(
        GOOD_PERMISO_UNICO,
        recommendation(approval_score=0.45, verdict="not_approve", borderline=True),
    )
    site.add(GOOD_PCOC, recommendation(approval_score=None, verdict="manual_review"))
    site.add(GOOD_PCOC)

    reviewed, borderline, manual, unreviewed = queue_rows(site)

    assert reviewed.split() == (
        "1 María Isabel Rivera Ortiz Permiso de Construcción Aprobar 0.86 Pendiente".split()
    )
    assert "Ana Lucía Torres Rivera Permiso Único" in borderline
    assert "No aprobar 0.45 Caso límite Pendiente" in borderline
    assert "Caso límite" not in reviewed
    assert "Sin puntuación — revisión manual — Pendiente" in manual
    assert "Sin revisar" in unreviewed and "Pendiente" not in unreviewed


def test_the_queue_shows_the_determination_of_each_permit_request(site):
    accepted = site.add(GOOD_PCOC, recommendation())
    overridden = site.add(GOOD_PCOC, recommendation())
    cure = site.add(GOOD_PCOC, recommendation())
    site.add(GOOD_PCOC, recommendation())
    determine(site, accepted, kind="accept")
    determine(site, overridden, kind="override", reason="El plano no corresponde al solar.")
    determine(site, cure, kind="cure_request", deficiencies="Falta el plano.")

    rows = queue_rows(site.restart() and site)

    assert rows[0].endswith("Recomendación aceptada: Aprobar")
    assert rows[1].endswith("Recomendación anulada: No aprobar")
    assert rows[2].endswith("Requerimiento de subsanación")
    assert rows[3].endswith("Pendiente")


def test_an_empty_queue_says_so(site):
    assert "No hay Solicitudes de Permiso almacenadas." in site.client.get("/").text


class DeterminedMeanwhile:
    """A review during which the Reviewer records a Determination on the stored Recommendation."""

    def __init__(self, site, request_id):
        self.site, self.request_id = site, request_id

    def review(self, raw):
        determine(self.site, self.request_id, kind="accept")
        return recommendation(approval_score=0.12, verdict="not_approve")


def test_a_review_that_ends_after_the_determination_does_not_replace_the_recommendation(site):
    request_id = site.add(GOOD_PCOC, recommendation())
    site.pipeline = DeterminedMeanwhile(site, request_id)
    client = site.restart()

    response = client.post(f"/solicitudes/{request_id}/recomendacion")

    assert response.status_code == 409
    page = text_of(client.get(f"/solicitudes/{request_id}").text)
    assert "Puntuación de aprobación 0.86" in page
    assert "Recomendación aceptada: Aprobar" in page


def test_a_determination_is_shown_even_when_its_recommendation_cannot_be_read(site):
    request_id = site.add(GOOD_PCOC, recommendation())
    determine(site, request_id, kind="override", reason="El plano no corresponde al solar.")
    with sqlite3.connect(site.path) as connection:
        connection.execute("UPDATE recommendations SET body = '{\"verdict\": \"old\"}'")

    html = site.client.get(f"/solicitudes/{request_id}").text

    assert "Recomendación anulada: No aprobar" in text_of(html)
    assert "Razón: El plano no corresponde al solar." in text_of(html)
    assert "hx-post" not in html
    assert queue_rows(site)[0].endswith("Recomendación anulada: No aprobar")


def test_a_curable_note_spanning_lines_is_one_deficiency_of_the_draft(site):
    note = Note(
        text="Falta el croquis.\nDebe mostrar los accesos.",
        section_ids=["RC-2023-E:3.7.1.2(b)(3)", "RC-2023-E:2.1.9.2(m)"],
        curable=True,
    )

    draft = cure_request_draft(site.page(notes=[note]))

    assert draft.strip() == (
        "Falta el croquis. Debe mostrar los accesos. "
        "(RC-2023-E:3.7.1.2(b)(3), RC-2023-E:2.1.9.2(m))"
    )
