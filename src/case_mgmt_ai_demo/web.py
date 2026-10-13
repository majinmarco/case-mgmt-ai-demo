"""The Reviewer web UI: a stored Permit Request and its Recommendation, in Spanish.

FastAPI serves Jinja pages; HTMX swaps in the Recommendation when a review is asked for. Every
citation links to the Reglamento PDF the app serves, on the page the index gives for the cited
Section, and opens it in a side panel.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from case_mgmt_ai_demo.chunks import Chunk, Doc, split_clauses
from case_mgmt_ai_demo.cli import CATEGORIES_ES, NOT_ASSESSED, RecommendationSource, verdict_es
from case_mgmt_ai_demo.config import Settings
from case_mgmt_ai_demo.fee_check import money
from case_mgmt_ai_demo.review import InvalidPermitRequest, Recommendation
from case_mgmt_ai_demo.store import PermitRequestStore, StoredPermitRequest

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
            "citation": _memoised(citation),
            "verdict": verdict_es(rec) if rec else None,
            "relevance": {r.section_id: r for r in rec.section_relevance} if rec else {},
            "categories": CATEGORIES_ES,
            "not_assessed": NOT_ASSESSED,
            "settings": settings,
        }

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "home.html", {"permit_requests": store.all()})

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
