"""The single `uv run case-mgmt-ai-demo` entry point."""

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from case_mgmt_ai_demo.review import InvalidPermitRequest, Recommendation

COMMANDS = {
    "review": "Produce a Recommendation for a Permit Request.",
    "ingest": "Chunk and index the Reglamento, then evaluate retrieval.",
    "serve": "Run the Reviewer web UI.",
}

VERDICTS_ES = {
    "approve": "Aprobar",
    "not_approve": "No aprobar",
    "manual_review": "Sin puntuación — revisión manual",
}

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
    return parser


def main(argv: list[str] | None = None, pipeline: RecommendationSource | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "review":
        return _review(args.file, pipeline, show_selection=args.selection)
    print(f"'{args.command}' is not implemented yet.", file=sys.stderr)
    return 1


def _review(path: Path, pipeline: RecommendationSource | None, show_selection: bool) -> int:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        print(f"No se pudo leer {path}: {error.strerror}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as error:
        print(f"{path} no es JSON válido: {error}", file=sys.stderr)
        return 1
    if pipeline is None:
        pipeline = _real_pipeline()
    try:
        recommendation = pipeline.review(raw)
    except InvalidPermitRequest as error:
        print("Solicitud de permiso inválida:", file=sys.stderr)
        for message in error.errors:
            print(f"  - {message}", file=sys.stderr)
        return 1
    print(format_recommendation(recommendation, show_selection))
    return 0


def _real_pipeline() -> RecommendationSource:
    from openai import OpenAI

    from case_mgmt_ai_demo.adapters import build_review_pipeline
    from case_mgmt_ai_demo.config import load_settings

    settings = load_settings()  # exports .env, so before OpenAI()
    return build_review_pipeline(settings, OpenAI())


def format_recommendation(recommendation: Recommendation, show_selection: bool = False) -> str:
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
            lines.append(f"  {shown:<{len(NOT_ASSESSED)}}  {requirement.text_es} [{requirement.id}]")
    for ref in recommendation.unclassified_clause_refs:
        lines.append(f"  ⚠ Inciso sin clasificar (se evalúa igualmente): {ref}")
    for ref in recommendation.unknown_clause_refs:
        lines.append(f"  ⚠ Inciso desconocido descartado: {ref}")
    if recommendation.errors:
        lines += ["", "Errores de la API (la revisión continuó sin esas respuestas):"]
        for error in recommendation.errors:
            subject = f" [{error.subject}]" if error.subject else ""
            lines.append(f"  ⚠ {error.step}{subject}: {error.message}")
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
    if show_selection:
        if not_applicable:
            lines += ["", "Secciones que no aplican:"]
            lines += [f"  {r.section_id}: {r.reason}" for r in not_applicable]
        width = max(map(len, CATEGORIES_ES.values()))
        lines += ["", "Selección de incisos:"]
        for selection in recommendation.selection:
            category = CATEGORIES_ES[selection.category]
            lines.append(f"  {category:<{width}}  {selection.ref}: {selection.reason}")
    return "\n".join(lines)
