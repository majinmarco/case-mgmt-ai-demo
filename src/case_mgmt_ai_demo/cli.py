"""The single `uv run case-mgmt-ai-demo` entry point."""

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from case_mgmt_ai_demo.fee_check import FeeCheck, money
from case_mgmt_ai_demo.review import (
    InvalidPermitRequest,
    MissingPinnedClause,
    Recommendation,
    RetrievedSection,
)

if TYPE_CHECKING:  # imported when used: Chroma and pymupdf are slow to load
    from case_mgmt_ai_demo.index import Embedder
    from case_mgmt_ai_demo.ingest import Ingestion

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

NOT_ASSESSED = "no evaluado"

CATEGORIES_ES = {
    "aplica": "aplica",
    "tramite_del_sistema": "trámite del sistema",
    "no_aplica": "no aplica",
}


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
                "--no-ocr",
                action="store_true",
                help="Do not read text drawn as outlines; it stays marked […].",
            )
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
            {doc: pdfs[doc] for doc in docs}, embedder, index=not args.no_index, ocr=not args.no_ocr
        )
    print(f"'{args.command}' is not implemented yet.", file=sys.stderr)
    return 1


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


def _ingest(
    pdfs: Mapping[str, Path | None], embedder: "Embedder | None", index: bool, ocr: bool
) -> int:
    """Ingests each document in `pdfs`; None stands for the configured PDF."""
    from case_mgmt_ai_demo.chunking import load_ingestion_config
    from case_mgmt_ai_demo.config import load_settings
    from case_mgmt_ai_demo.extraction import find_tessdata
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
    tessdata = find_tessdata(settings.tessdata_dir) if ocr else None
    for doc, pdf in sources.items():
        if tessdata and doc == "RC-2023-E":
            print(
                "Extrayendo el texto (el OCR de las páginas con texto trazado tarda unos minutos)…"
            )
        result = ingest(
            pdf,
            load_ingestion_config(INGESTION_CONFIGS[doc]),
            settings.data_dir,
            settings.index_dir,
            embedder if index else None,
            tessdata,
        )
        print(format_ingestion(result, settings.index_dir))
    return 0


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
    lines += [f"Informe: {result.report_path}", f"Texto de los fragmentos: {result.chunks_path}"]
    return "\n".join(lines)


def format_recommendation(
    recommendation: Recommendation, show_selection: bool = False, show_trace: bool = False
) -> str:
    request = recommendation.request
    score = recommendation.approval_score
    verdict = VERDICTS_ES[recommendation.verdict]
    if recommendation.verdict == "manual_review" and recommendation.errors:
        verdict = "Revisión manual (falló una llamada a la API; ver abajo)"
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
    if recommendation.requirements:
        lines.append("Requisitos:")
        for requirement in recommendation.requirements:
            probability = requirement.probability
            shown = NOT_ASSESSED if probability is None else f"{probability:.2f}"
            lines.append(
                f"  {shown:<{len(NOT_ASSESSED)}}  {requirement.text_es} [{requirement.id}]"
            )
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
    if show_selection and not show_trace:  # the trace gives every Section's verdict itself
        if not_applicable:
            lines += ["", "Secciones que no aplican:"]
            lines += [f"  {r.section_id}: {r.reason}" for r in not_applicable]
        lines += ["", *_selection_lines(recommendation)]
    if show_trace:
        lines += ["", *_trace_lines(recommendation)]
    return "\n".join(lines)


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
