"""The single `uv run case-mgmt-ai-demo` entry point."""

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TextIO

from case_mgmt_ai_demo.fee_check import FeeCheck, money
from case_mgmt_ai_demo.review import (
    InvalidPermitRequest,
    MissingPinnedClause,
    Recommendation,
    RetrievedSection,
)

if TYPE_CHECKING:  # imported when used: Chroma and pymupdf are slow to load
    from case_mgmt_ai_demo.config import Settings
    from case_mgmt_ai_demo.extraction import OcrEngine
    from case_mgmt_ai_demo.index import Embedder
    from case_mgmt_ai_demo.ingest import Ingestion
    from case_mgmt_ai_demo.store import PermitRequestStore

COMMANDS = {
    "review": "Produce a Recommendation for a Permit Request.",
    "ingest": "Chunk the Reglamento and load it into the index.",
    "serve": "Run the Reviewer web UI.",
}

VERDICTS_ES = {
    "approve": "Aprobar",
    "not_approve": "No aprobar",
    "manual_review": "Sin puntuación — revisión manual",
}

# Each document of the Reglamento with its ingestion configuration, in the order ingested.
INGESTION_CONFIGS = {"RC-2023-E": "rc_2023_e", "COBROS-OA-2022-07": "cobros_oa_2022_07"}

OCR_ENGINES = ("tesseract", "unlimited-ocr")

PROGRESS_STEPS_ES = {"extract": "Extrayendo el texto", "embed": "Incrustando fragmentos"}
PROGRESS_UNITS_ES = {"extract": "páginas", "embed": "fragmentos"}
_BAR_WIDTH = 30

NOT_ASSESSED = "no evaluado"

CATEGORIES_ES = {
    "aplica": "aplica",
    "no_verificable": "no verificable",
    "tramite_del_sistema": "trámite del sistema",
    "no_aplica": "no aplica",
}


UNVERIFIABLE_HEADING = (
    "Requisitos que el formulario no permite acreditar (sin puntuar; los verifica el Revisor):"
)


class RecommendationSource(Protocol):
    """The review pipeline, as the CLI needs it (fakeable in tests)."""

    def review(self, raw: Mapping[str, Any]) -> Recommendation: ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="case-mgmt-ai-demo", description="Permit Review Assistant demo."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in COMMANDS.items():
        subparser = subcommands.add_parser(name, help=help_text)
        if name == "review":
            subparser.add_argument("file", type=Path, help="Permit Request as a JSON file")
            subparser.add_argument(
                "--selection",
                action="store_true",
                help="List every clause with its category and the reason given.",
            )
            subparser.add_argument(
                "--trace",
                action="store_true",
                help="Print the pipeline trace: the retrieved Sections with their similarity, "
                "the clause selection, each Requirement's clauses and each node's output.",
            )
        if name == "ingest":
            subparser.add_argument(
                "--pdf", type=Path, help="Reglamento PDF (default: PRA_REGLAMENTO_PDF)"
            )
            subparser.add_argument(
                "--cobros-pdf", type=Path, help="Fee order PDF (default: PRA_COBROS_PDF)"
            )
            subparser.add_argument(
                "--doc",
                choices=list(INGESTION_CONFIGS),
                help="Ingest only this document (default: both).",
            )
            subparser.add_argument(
                "--no-index",
                action="store_true",
                help="Chunk and write the report only: no embeddings, index untouched.",
            )
            subparser.add_argument(
                "--ocr",
                choices=OCR_ENGINES,
                help="What reads text drawn as outlines (default: PRA_OCR_ENGINE, or tesseract).",
            )
            subparser.add_argument(
                "--clear-clauses",
                action="store_true",
                help="Forget the stored Clauses of each document ingested, so every Section's "
                "are extracted again when a review next needs them.",
            )
            subparser.add_argument(
                "--no-ocr",
                action="store_true",
                help="Do not read text drawn as outlines; it stays marked […].",
            )
        if name == "serve":
            subparser.add_argument("--host", default="127.0.0.1", help="Address to listen on.")
            subparser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    return parser


def main(
    argv: list[str] | None = None,
    pipeline: RecommendationSource | None = None,
    embedder: "Embedder | None" = None,
) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "review":
        return _review(
            args.file, pipeline, show_selection=args.selection, show_trace=args.trace
        )
    if args.command == "ingest":
        pdfs = {"RC-2023-E": args.pdf, "COBROS-OA-2022-07": args.cobros_pdf}
        docs = [args.doc] if args.doc else list(INGESTION_CONFIGS)
        return _ingest(
            {doc: pdfs[doc] for doc in docs},
            embedder,
            index=not args.no_index,
            ocr=None if args.no_ocr else args.ocr or "",
            clear_clauses=args.clear_clauses,
        )
    return _serve(args.host, args.port, pipeline)


def _review(
    path: Path, pipeline: RecommendationSource | None, show_selection: bool, show_trace: bool
) -> int:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        print(f"No se pudo leer {path}: {error.strerror}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as error:
        print(f"{path} no es JSON válido: {error}", file=sys.stderr)
        return 1
    if pipeline is None:
        from case_mgmt_ai_demo.adapters import IndexNotBuilt

        try:
            pipeline = _real_pipeline()
        except IndexNotBuilt as error:
            print(
                f"El índice del Reglamento ({error}) no contiene RC-2023-E: ejecute "
                "`case-mgmt-ai-demo ingest`, o use PRA_RETRIEVER=fixture.",
                file=sys.stderr,
            )
            return 1
    try:
        recommendation = pipeline.review(raw)
    except InvalidPermitRequest as error:
        print("Solicitud de permiso inválida:", file=sys.stderr)
        for message in error.errors:
            print(f"  - {message}", file=sys.stderr)
        return 1
    except MissingPinnedClause as error:
        print(
            f"Los Criterios no concuerdan con las Secciones disponibles: {error}. "
            "Revise los Criterios o vuelva a ejecutar `case-mgmt-ai-demo ingest`.",
            file=sys.stderr,
        )
        return 1
    print(format_recommendation(recommendation, show_selection, show_trace))
    return 0


def _real_pipeline() -> RecommendationSource:
    from openai import OpenAI

    from case_mgmt_ai_demo.adapters import build_review_pipeline
    from case_mgmt_ai_demo.config import load_settings

    settings = load_settings()  # exports .env, so before OpenAI()
    return build_review_pipeline(settings, OpenAI())


class _LazyPipeline:
    """The real pipeline, built at the first review: `serve` shows stored Recommendations
    without an API key or an index."""

    def __init__(self) -> None:
        self._pipeline: RecommendationSource | None = None

    def review(self, raw: Mapping[str, Any]) -> Recommendation:
        if self._pipeline is None:
            self._pipeline = _real_pipeline()
        return self._pipeline.review(raw)


def _serve(host: str, port: int, pipeline: RecommendationSource | None) -> int:
    import uvicorn

    from case_mgmt_ai_demo.config import load_settings
    from case_mgmt_ai_demo.index import ReglamentoIndex
    from case_mgmt_ai_demo.store import PermitRequestStore
    from case_mgmt_ai_demo.web import create_app

    settings = load_settings()
    store = PermitRequestStore(settings.db_path)
    if not store.all():
        _store_examples(settings.examples_dir, store)
    # Citations get their PDF page from the index; without one they are shown as plain text.
    index = ReglamentoIndex(settings.index_dir)
    app = create_app(store, pipeline or _LazyPipeline(), index, settings)
    print(f"Interfaz del Revisor: http://{host}:{port}", file=sys.stderr)
    uvicorn.run(app, host=host, port=port)
    return 0


def _store_examples(directory: Path, store: "PermitRequestStore") -> None:
    """Store every Permit Request in the directory (JSON files, by name); one that is not
    valid is reported and left out."""
    from pydantic import ValidationError

    from case_mgmt_ai_demo.permit_request import parse_permit_request

    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.json")):
        try:
            request = parse_permit_request(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            reason = str(error).splitlines()[0]
            print(f"Ejemplo omitido, no es una solicitud válida: {path} ({reason})", file=sys.stderr)
            continue
        store.add(request)


def _ingest(
    pdfs: Mapping[str, Path | None],
    embedder: "Embedder | None",
    index: bool,
    ocr: str | None,
    clear_clauses: bool = False,
) -> int:
    """Ingests each document in `pdfs`; None stands for the configured PDF. `ocr` names the
    OCR engine: empty for the configured one, None for no OCR. `clear_clauses` forgets every
    stored Clause of each document, not only those of the Sections that changed."""
    from case_mgmt_ai_demo.chunking import load_ingestion_config
    from case_mgmt_ai_demo.clause_store import build_clause_store
    from case_mgmt_ai_demo.config import load_settings
    from case_mgmt_ai_demo.ingest import ingest

    settings = load_settings()  # exports .env, so before OpenAI()
    configured = {"RC-2023-E": settings.reglamento_pdf, "COBROS-OA-2022-07": settings.cobros_pdf}
    sources = {doc: pdf or configured[doc] for doc, pdf in pdfs.items()}
    for doc, pdf in sources.items():
        if not pdf.is_file():
            which = "del Reglamento" if doc == "RC-2023-E" else "de la orden de cobros"
            print(f"No se encontró el PDF {which}: {pdf}", file=sys.stderr)
            return 1
    if index and embedder is None:
        from openai import OpenAI

        from case_mgmt_ai_demo.adapters import OpenAIEmbedder

        embedder = OpenAIEmbedder(OpenAI(), settings.embedding_model)
    engine = None
    if ocr is not None and "RC-2023-E" in sources:  # the fee order draws no text as outlines
        try:
            engine = _ocr_engine(ocr or settings.ocr_engine, settings)
        except OcrUnavailable as error:
            print(f"No se puede usar el OCR {error}", file=sys.stderr)
            return 1
    clause_store = build_clause_store(settings)
    for doc, pdf in sources.items():
        if engine and doc == "RC-2023-E":
            print(
                f"El OCR ({engine.name}) de las páginas con texto trazado tarda unos minutos.",
                file=sys.stderr,
            )
        config = load_ingestion_config(INGESTION_CONFIGS[doc])
        result = ingest(
            pdf,
            config,
            settings.data_dir,
            settings.index_dir,
            embedder if index else None,
            engine,
            ProgressBar(sys.stderr),
            clause_store,
        )
        print(format_ingestion(result, settings.index_dir))
        if clear_clauses:
            cleared = clause_store.clear(config.doc)
            print(f"Secciones de {doc} con incisos guardados descartados: todas ({cleared})")
    return 0


class OcrUnavailable(Exception):
    """The chosen OCR engine cannot run; the message completes "No se puede usar el OCR"."""


def _ocr_engine(name: str, settings: "Settings") -> "OcrEngine | None":
    """Tesseract is left out when its Spanish model is missing, as it always was; the engine
    that has to be asked for fails instead."""
    if name == "unlimited-ocr":
        from case_mgmt_ai_demo.unlimited_ocr import UnlimitedOcr, UnlimitedOcrUnavailable

        print("Cargando el modelo Unlimited-OCR…", file=sys.stderr)
        try:
            return UnlimitedOcr()
        except UnlimitedOcrUnavailable as error:
            raise OcrUnavailable(f"unlimited-ocr: {error}") from error
    from case_mgmt_ai_demo.extraction import Tesseract, find_tessdata

    tessdata = find_tessdata(settings.tessdata_dir)
    return Tesseract(tessdata) if tessdata else None


class ProgressBar:
    """One line on a terminal that fills as an ingestion advances; silent anywhere else."""

    def __init__(self, stream: TextIO):
        self._stream = stream

    def __call__(self, step: str, done: int, total: int) -> None:
        if not self._stream.isatty() or total <= 0:
            return
        filled = _BAR_WIDTH * done // total
        bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
        self._stream.write(
            f"\r{PROGRESS_STEPS_ES[step]} [{bar}] {100 * done // total:>3}% "
            f"({done}/{total} {PROGRESS_UNITS_ES[step]})"
        )
        if done >= total:
            self._stream.write("\n")
        self._stream.flush()


def format_ingestion(result: "Ingestion", index_dir: Path) -> str:
    report = result.report
    outlined = report.outlined_text
    lines = [
        f"Ingesta de {report.pdf} ({report.config.doc})",
        f"Fragmentos: {report.chunks} de {report.sections} Secciones "
        f"({report.split_sections} divididas en partes)",
    ]
    if outlined.read or outlined.unread:
        lines.append(
            f"Texto trazado como contornos: {outlined.read} tramos leídos por OCR, "
            f"{outlined.unread} sin leer (marcados […])"
        )
        if not outlined.ocr:
            lines.append(
                "  ⚠ Sin OCR (--no-ocr, o falta el modelo de español de Tesseract, "
                "spa.traineddata: instálelo o indique PRA_TESSDATA_DIR)."
            )
        else:
            lines.append(f"  Motor de OCR: {outlined.engine}")
    lines.append(f"Avisos de extracción: {len(report.flags)}")
    lines += [f"  {count:>4}  {check}" for check, count in sorted(report.flags_by_check.items())]
    if result.indexed:
        indexed = result.indexed
        lines.append(
            f"Índice ({index_dir}): {indexed.embedded} fragmentos incrustados, "
            f"{indexed.unchanged} sin cambios, {indexed.removed} eliminados"
        )
    else:
        lines.append("Índice: sin tocar (--no-index)")
    if result.clauses_forgotten is not None:
        lines.append(
            f"Secciones con incisos guardados descartados: {result.clauses_forgotten} "
            "(cambiaron o ya no existen; se extraen de nuevo al revisar)"
        )
    lines += [f"Informe: {result.report_path}", f"Texto de los fragmentos: {result.chunks_path}"]
    return "\n".join(lines)


def format_recommendation(
    recommendation: Recommendation, show_selection: bool = False, show_trace: bool = False
) -> str:
    request = recommendation.request
    score = recommendation.approval_score
    verdict = verdict_es(recommendation)
    if recommendation.borderline:
        verdict += " (caso límite: revisar con detenimiento)"
    set_aside = recommendation.set_aside_count
    considered = len(recommendation.selection) + len(recommendation.unclassified_clause_refs)
    shown_score = "sin puntuación" if score is None else f"{score:.2f}"
    if weakest := recommendation.weakest_requirement:
        shown_score += f" (requisito más débil: {weakest.id})"
    not_applicable = recommendation.sections_not_applicable
    lines = [
        f"Recomendación — Solicitud de {request.type_name_es} de {request.applicant_name}",
        f"Puntuación de aprobación: {shown_score}",
        f"Veredicto: {verdict}",
        f"Secciones recuperadas: {len(recommendation.section_relevance)} "
        f"({len(not_applicable)} no aplica; --selection para ver por qué)",
        f"Incisos evaluados: {considered - set_aside} de {considered} "
        f"({set_aside} descartados; --selection para ver la lista)",
    ]
    unverifiable = recommendation.unverifiable
    if unverifiable:
        lines.append(
            f"  ⚠ {len(unverifiable)} de los descartados son requisitos que el formulario no "
            "permite acreditar (lista al final)"
        )
    if recommendation.requirements:
        reasons = {s.ref: s.reason for s in recommendation.selection}
        indent = " " * (len(NOT_ASSESSED) + 4)
        lines.append("Requisitos:")
        for requirement in recommendation.requirements:
            probability = requirement.probability
            shown = NOT_ASSESSED if probability is None else f"{probability:.2f}"
            lines.append(
                f"  {shown:<{len(NOT_ASSESSED)}}  {requirement.text_es} [{requirement.id}]"
            )
            if probability is not None:
                lines.append(f"{indent}incisos: {_refs(requirement.clause_refs)}")
                continue
            # The Decisions API names no reason, so the Clauses asked about are spelled out.
            lines.append(f"{indent}⚠ pregunta rechazada, sin razón dada; se preguntó por:")
            for ref in requirement.clause_refs:
                reason = reasons.get(ref, "sin clasificar")
                lines.append(f"{indent}  {ref}: {reason}")
    for ref in recommendation.unclassified_clause_refs:
        lines.append(f"  ⚠ Inciso sin clasificar (se evalúa igualmente): {ref}")
    for ref in recommendation.unknown_clause_refs:
        lines.append(f"  ⚠ Inciso desconocido descartado: {ref}")
    if recommendation.errors:
        lines += ["", "Errores de la API (la revisión continuó sin esas respuestas):"]
        for error in recommendation.errors:
            subject = f" [{error.subject}]" if error.subject else ""
            lines.append(f"  ⚠ {error.step}{subject}: {error.message}")
    if recommendation.fee_check:
        lines += ["", *_fee_check_lines(recommendation.fee_check)]
    retried = " (reescritas una vez: citas no válidas en el primer intento)"
    lines += ["", f"Notas{retried if recommendation.notes_retried else ''}:"]
    for number, note in enumerate(recommendation.notes, start=1):
        curable = " (subsanable)" if note.curable else ""
        lines.append(f"  {number}. {note.text} [{', '.join(note.section_ids)}]{curable}")
    if recommendation.dropped_notes:
        lines += ["", "Notas descartadas (citas no válidas tras un reintento):"]
        for dropped in recommendation.dropped_notes:
            cited = f" [{', '.join(dropped.section_ids)}]" if dropped.section_ids else ""
            lines.append(f"  ⚠ {dropped.text}{cited}: {dropped.problem_es()}")
    if unverifiable:
        lines += ["", UNVERIFIABLE_HEADING]
        lines += [f"  {s.ref}: {s.reason}" for s in unverifiable]
    if show_selection and not show_trace:  # the trace gives every Section's verdict itself
        if not_applicable:
            lines += ["", "Secciones que no aplican:"]
            lines += [f"  {r.section_id}: {r.reason}" for r in not_applicable]
        lines += ["", *_selection_lines(recommendation)]
    if show_trace:
        lines += ["", *_trace_lines(recommendation)]
    return "\n".join(lines)


def verdict_es(recommendation: Recommendation) -> str:
    """The verdict as shown to a Reviewer; a failed model call is named as the reason."""
    if recommendation.verdict == "manual_review" and recommendation.errors:
        return "Revisión manual (falló una llamada a la API; ver abajo)"
    return VERDICTS_ES[recommendation.verdict]


def _refs(refs: list[str]) -> str:
    """Clause references with each Section named once: 'RC-2023-E:2.1.9.2(g, h); …'."""
    by_section: dict[str, list[str]] = {}
    for ref in refs:
        section_id, _, rest = ref.partition("(")
        by_section.setdefault(section_id, []).append(rest.removesuffix(")") or "párrafo inicial")
    return "; ".join(f"{section_id}({', '.join(parts)})" for section_id, parts in by_section.items())


def _selection_lines(recommendation: Recommendation) -> list[str]:
    width = max(map(len, CATEGORIES_ES.values()))
    lines = ["Selección de incisos:"]
    for selection in recommendation.selection:
        category = CATEGORIES_ES[selection.category]
        lines.append(f"  {category:<{width}}  {selection.ref}: {selection.reason}")
    return lines


def _trace_lines(recommendation: Recommendation) -> list[str]:
    """The pipeline trace: what was retrieved and how near, what was set aside and why, what
    each Requirement was assessed on, and what each node returned."""
    trace = recommendation.trace
    relevance = {r.section_id: r for r in recommendation.section_relevance}
    lines = ["Traza del pipeline", "", "Secciones recuperadas (similitud con la solicitud):"]
    for section in trace.retrieved:
        verdict = ""
        if found := relevance.get(section.section_id):
            verdict = f"\n      {'aplica' if found.applies else 'no aplica'}: {found.reason}"
            if not found.applies and section.clause_refs:
                # Its Clauses never reach the selection below, so they are named here.
                verdict += f"\n      incisos no evaluados: {', '.join(section.clause_refs)}"
        lines.append(
            f"  {_similarity(section)}  {section.section_id} {section.heading}"
            f"{_pinned(section)}{verdict}"
        )
    lines += ["", *_selection_lines(recommendation)]
    lines += [
        f"  ⚠ sin clasificar (se evalúa igualmente)  {ref}"
        for ref in recommendation.unclassified_clause_refs
    ]
    lines += ["", "Requisitos y sus incisos:"]
    for requirement in recommendation.requirements:
        probability = requirement.probability
        shown = NOT_ASSESSED if probability is None else f"{probability:.2f}"
        lines.append(
            f"  {shown:<{len(NOT_ASSESSED)}}  {requirement.id}: "
            f"{', '.join(requirement.clause_refs)}"
        )
    lines += ["", "Salida de cada nodo:"]
    for step in trace.steps:
        lines.append(f"  {step.node}")
        lines += [
            f"    {key}: {json.dumps(value, ensure_ascii=False)}"
            for key, value in step.output.items()
        ]
    return lines


def _similarity(section: RetrievedSection) -> str:
    return "  —  " if section.similarity is None else f"{section.similarity:.3f}"


def _pinned(section: RetrievedSection) -> str:
    if not section.pinned_by:
        return ""
    return f" (fijada por {', '.join(section.pinned_by)})"


def _fee_check_lines(fee_check: FeeCheck) -> list[str]:
    if fee_check.matches is None:
        outcome = f"⚠ no se pudo verificar: {fee_check.unverifiable_es}"
    else:
        outcome = "coinciden" if fee_check.matches else "⚠ no coinciden"
    lines = [
        f"Verificación de derechos: {outcome} [{', '.join(fee_check.section_ids)}]",
        f"  Declarados: {money(fee_check.declared_fee)}",
    ]
    if fee_check.expected_fee is not None:
        lines.append(f"  Calculados: {money(fee_check.expected_fee)}")
    for charge in fee_check.charges:
        exempted = "" if charge.percent == 100 else f" ({charge.percent}% de {money(charge.base)})"
        lines.append(f"    {money(charge.amount):>10}  {charge.ref} {charge.concept_es}{exempted}")
    if fee_check.exemption_es:
        lines.append(f"  Exención aplicada: {fee_check.exemption_es}")
    return lines
