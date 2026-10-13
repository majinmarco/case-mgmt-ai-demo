"""The Reviewer web UI: the queue of stored Permit Requests, and each one with its
Recommendation and the Reviewer's Determination, in Spanish.

FastAPI serves Jinja pages; HTMX swaps in the Recommendation when a review is asked for. Every
citation links to the Reglamento PDF the app serves, on the page the index gives for the cited
Section, and opens it in a side panel.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from case_mgmt_ai_demo.chunks import Chunk, Doc, split_clauses
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
from case_mgmt_ai_demo.review import InvalidPermitRequest, Recommendation
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

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        queue = [
            QueueEntry(stored, store.recommendation(stored.id), store.determination(stored.id))
            for stored in store.all()
        ]
        return templates.TemplateResponse(
            request, "home.html", {"queue": queue, "verdicts": VERDICTS_ES}
        )

    @app.get("/solicitudes/{request_id}", response_class=HTMLResponse)
    def case_page(request: Request, request_id: int) -> HTMLResponse:
        stored = stored_or_404(request_id)
        return templates.TemplateResponse(
            request, "case.html", context(stored, store.recommendation(request_id))
        )

    @app.post("/solicitudes/{request_id}/recomendacion", response_class=HTMLResponse)
    def review(request: Request, request_id: int) -> HTMLResponse:
        """Review the Permit Request, store its Recommendation and return it for HTMX to swap in."""
        stored = stored_or_404(request_id)
        if store.determination(request_id) is not None:
            # The Determination was made on the stored Recommendation, which therefore stays.
            raise AlreadyDetermined(request_id)
        try:
            rec = pipeline.review(stored.request.model_dump(mode="json"))
        except InvalidPermitRequest as error:
            failure = {"review_errors": error.errors}
        except Exception as error:  # shown to the Reviewer: a missing index, a Criteria mismatch
            failure = {"review_errors": [f"{type(error).__name__}: {error}"]}
        else:
            store.save_recommendation(request_id, rec)
            failure = {}
        return templates.TemplateResponse(
            request,
            "_recommendation.html",
            context(stored, store.recommendation(request_id)) | failure,
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


def _memoised(citation: Callable[[str], Citation]) -> Callable[[str], Citation]:
    """A page cites the same Sections many times; each is looked up once per page."""
    found: dict[str, Citation] = {}

    def lookup(ref: str) -> Citation:
        if ref not in found:
            found[ref] = citation(ref)
        return found[ref]

    return lookup
