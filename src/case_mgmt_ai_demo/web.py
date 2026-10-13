"""The Reviewer web UI: the queue of stored Permit Requests, and each one with its
Recommendation and the Reviewer's Determination, in Spanish.

FastAPI serves Jinja pages; HTMX swaps in the Recommendation when a review is asked for. Every
citation links to the Reglamento PDF the app serves, on the page the index gives for the cited
Section, and opens it in a side panel.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Protocol
from urllib.parse import parse_qs, quote

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from case_mgmt_ai_demo.chunks import Chunk, Doc, RequestType, split_clauses
from case_mgmt_ai_demo.cli import (
    CATEGORIES_ES,
    NOT_ASSESSED,
    VERDICTS_ES,
    RecommendationSource,
    verdict_es,
)
from case_mgmt_ai_demo.config import Settings
from case_mgmt_ai_demo.determination import (
    Determination,
    InvalidDetermination,
    cure_request_draft,
    determine,
)
from case_mgmt_ai_demo.fee_check import money
from case_mgmt_ai_demo.permit_request import LABELS, parse_permit_request
from case_mgmt_ai_demo.request_form import (
    DEFAULT_REQUEST_TYPE,
    REQUEST_TYPES_ES,
    core_groups,
    errors_by_field,
    raw_fields,
    type_groups,
)
from case_mgmt_ai_demo.review import InvalidPermitRequest, Recommendation, intake_errors_es
from case_mgmt_ai_demo.store import AlreadyDetermined, PermitRequestStore, StoredPermitRequest

_HERE = Path(__file__).parent
PDF_ROUTE = "/reglamento"


class SectionSource(Protocol):
    """Where a cited Section's pages come from: the Reglamento index (fakeable in tests)."""

    def section(self, section_id: str) -> list[Chunk]:
        """The chunk of a Section, or its parts in order; empty when the ID is unknown."""
        ...


@dataclass(frozen=True)
class Citation:
    """A cited Section or Clause, and where its link goes."""

    ref: str
    href: str | None  # None when there is nothing to open: shown as plain text


@dataclass(frozen=True)
class QueueEntry:
    """A Permit Request in the queue, with what has been done on it so far."""

    stored: StoredPermitRequest
    rec: Recommendation | None
    determination: Determination | None


def cited_page(citation: str, chunks: Sequence[Chunk]) -> int | None:
    """The PDF page a citation opens on: where its Section starts or, for a Clause of a split
    Section, where the part holding that Clause starts. None when the Section is unknown."""
    if not chunks:
        return None
    section_id, _, rest = citation.partition("(")
    if rest:
        clause_ref = f"{section_id}({rest.split(')', 1)[0]})"
        for chunk in chunks:
            if any(clause.ref == clause_ref for clause in split_clauses(chunk)):
                return chunk.page_start
    return chunks[0].page_start


def create_app(
    store: PermitRequestStore,
    pipeline: RecommendationSource,
    sections: SectionSource,
    settings: Settings,
) -> FastAPI:
    """The web app over the stored Permit Requests. `pipeline` reviews one when asked to;
    `sections` gives each citation its page."""
    app = FastAPI(title="Asistente de Revisión de Permisos", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")
    templates = Jinja2Templates(directory=_HERE / "templates")
    templates.env.filters["json_es"] = lambda value: json.dumps(
        value, ensure_ascii=False, indent=2
    )
    templates.env.filters["money"] = money
    pdfs: dict[Doc, Path] = {
        "RC-2023-E": settings.reglamento_pdf,
        "COBROS-OA-2022-07": settings.cobros_pdf,
    }
    # Why the review of a Permit Request just created failed, until its case page has said so.
    failed_reviews: dict[int, list[str]] = {}

    @app.exception_handler(AlreadyDetermined)
    def already_determined(request: Request, error: AlreadyDetermined) -> Response:
        """Also what a review that ends after the Determination was recorded comes to."""
        return PlainTextResponse("La solicitud ya tiene una Determinación", status_code=409)

    def citation(ref: str) -> Citation:
        chunks = sections.section(ref.split("(", 1)[0])
        page = cited_page(ref, chunks)
        # Without the Section's page, or the PDF to open on it, a link would lead nowhere.
        if page is None or not (pdf := pdfs[chunks[0].doc]).is_file():
            return Citation(ref, None)
        return Citation(ref, f"{PDF_ROUTE}/{quote(pdf.name)}#page={page}")

    def stored_or_404(request_id: int) -> StoredPermitRequest:
        stored = store.get(request_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="Solicitud de Permiso no encontrada")
        return stored

    def context(stored: StoredPermitRequest, rec: Recommendation | None) -> dict[str, Any]:
        return {
            "stored": stored,
            "rec": rec,
            "determination": store.determination(stored.id),
            "cure_request_draft": cure_request_draft(rec) if rec else "",
            "citation": _memoised(citation),
            "verdict": verdict_es(rec) if rec else None,
            "relevance": {r.section_id: r for r in rec.section_relevance} if rec else {},
            "reasons": {s.ref: s.reason for s in rec.selection} if rec else {},
            "categories": CATEGORIES_ES,
            "not_assessed": NOT_ASSESSED,
            "settings": settings,
        }

    def reviewed(stored: StoredPermitRequest) -> list[str]:
        """Review a Permit Request and store its Recommendation; when that fails, why."""
        try:
            rec = pipeline.review(stored.request.model_dump(mode="json"))
        except InvalidPermitRequest as error:
            return error.errors
        except Exception as error:  # shown to the Reviewer: a missing index, a Criteria mismatch
            return [f"{type(error).__name__}: {error}"]
        store.save_recommendation(stored.id, rec)
        return []

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        queue = [
            QueueEntry(stored, store.recommendation(stored.id), store.determination(stored.id))
            for stored in store.all()
        ]
        return templates.TemplateResponse(
            request, "home.html", {"queue": queue, "verdicts": VERDICTS_ES}
        )

    def form_context(
        request_type: RequestType,
        values: Mapping[str, Sequence[str]],
        errors: Sequence[tuple[str, str]] = (),
    ) -> dict[str, Any]:
        shown, other = errors_by_field(request_type, errors)
        return {
            "request_type": request_type,
            "request_types": REQUEST_TYPES_ES,
            "core_groups": core_groups(),
            "type_groups": type_groups(request_type),
            "labels": LABELS,
            "values": values,
            "errors": shown,
            "other_errors": other,  # listed above the form
        }

    # Before the case page, whose path would otherwise take "nueva" for a request ID.
    @app.get("/solicitudes/nueva", response_class=HTMLResponse)
    def new_request(
        request: Request, request_type: RequestType = DEFAULT_REQUEST_TYPE
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "new_request.html", form_context(request_type, {})
        )

    @app.get("/solicitudes/nueva/campos", response_class=HTMLResponse)
    def request_type_fields(request: Request, request_type: RequestType) -> HTMLResponse:
        """The fields of a Request Type, for HTMX to swap into the form when it is chosen,
        holding what the form already has for them (the declared fee, which every type has)."""
        context = form_context(request_type, parse_qs(request.url.query))
        return templates.TemplateResponse(
            request, "_request_fields.html", context | {"groups": context["type_groups"]}
        )

    @app.post("/solicitudes", response_class=HTMLResponse)
    def create_request(
        request: Request, submitted: Annotated[dict[str, list[str]], Depends(_submitted_form)]
    ) -> Response:
        """Store the Permit Request the form declares, review it and open its case page; an
        invalid one gets the form back, with each error under its field."""
        htmx = "HX-Request" in request.headers
        request_type = (submitted.get("request_type") or [""])[0]
        try:
            permit_request = parse_permit_request(raw_fields(request_type, submitted))
        except ValidationError as error:
            known: dict[str, RequestType] = {name: name for name in REQUEST_TYPES_ES}
            shown = known.get(request_type, DEFAULT_REQUEST_TYPE)
            return templates.TemplateResponse(
                request,
                "_request_form.html" if htmx else "new_request.html",
                form_context(shown, submitted, intake_errors_es(error)),
                status_code=422,
            )
        stored = StoredPermitRequest(store.add(permit_request), permit_request)
        if review_errors := reviewed(stored):
            failed_reviews[stored.id] = review_errors
        case_url = f"/solicitudes/{stored.id}"
        if htmx:  # HTMX would follow a redirect itself and swap the page in: tell it to navigate
            return Response(status_code=204, headers={"HX-Redirect": case_url})
        return RedirectResponse(case_url, status_code=303)

    @app.get("/solicitudes/{request_id}", response_class=HTMLResponse)
    def case_page(request: Request, request_id: int) -> HTMLResponse:
        stored = stored_or_404(request_id)
        return templates.TemplateResponse(
            request,
            "case.html",
            context(stored, store.recommendation(request_id))
            | {"review_errors": failed_reviews.pop(request_id, [])},
        )

    @app.post("/solicitudes/{request_id}/recomendacion", response_class=HTMLResponse)
    def review(request: Request, request_id: int) -> HTMLResponse:
        """Review the Permit Request, store its Recommendation and return it for HTMX to swap in."""
        stored = stored_or_404(request_id)
        if store.determination(request_id) is not None:
            # The Determination was made on the stored Recommendation, which therefore stays.
            raise AlreadyDetermined(request_id)
        review_errors = reviewed(stored)
        return templates.TemplateResponse(
            request,
            "_recommendation.html",
            context(stored, store.recommendation(request_id)) | {"review_errors": review_errors},
        )

    @app.post("/solicitudes/{request_id}/determinacion", response_class=HTMLResponse)
    async def record_determination(request: Request, request_id: int) -> Response:
        """Record the Reviewer's Determination and go back to the case page, which shows it."""
        stored = stored_or_404(request_id)
        rec = store.recommendation(request_id)
        if rec is None:
            raise HTTPException(status_code=409, detail="La solicitud no tiene una Recomendación")
        if store.determination(request_id) is not None:
            raise AlreadyDetermined(request_id)
        # A plain form post, read here: FastAPI's own form parsing needs another dependency.
        body = (await request.body()).decode(errors="replace")
        fields = parse_qs(body, keep_blank_values=True)
        form = {name: values[-1] for name, values in fields.items()}
        try:
            determination = determine(
                rec,
                form.get("kind", ""),
                reason=form.get("reason", ""),
                outcome=form.get("outcome", ""),
                deficiencies=form.get("deficiencies", ""),
            )
        except InvalidDetermination as error:
            return templates.TemplateResponse(
                request,
                "case.html",
                context(stored, rec) | {"determination_error": str(error), "form": form},
                status_code=422,
            )
        store.save_determination(request_id, determination)
        return RedirectResponse(f"/solicitudes/{request_id}", status_code=303)

    @app.get(PDF_ROUTE + "/{filename}")
    def reglamento_pdf(filename: str) -> FileResponse:
        path = next((pdf for pdf in pdfs.values() if pdf.name == filename), None)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="Documento no encontrado")
        # No download filename, so the browser's viewer shows it in the side panel.
        return FileResponse(path, media_type="application/pdf")

    return app


async def _submitted_form(request: Request) -> dict[str, list[str]]:
    """The values of each field of a submitted form, as the browser encodes them."""
    return parse_qs((await request.body()).decode(errors="replace"), keep_blank_values=True)


def _memoised(citation: Callable[[str], Citation]) -> Callable[[str], Citation]:
    """A page cites the same Sections many times; each is looked up once per page."""
    found: dict[str, Citation] = {}

    def lookup(ref: str) -> Citation:
        if ref not in found:
            found[ref] = citation(ref)
        return found[ref]

    return lookup
