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
    return parser


def main(argv: list[str] | None = None, pipeline: RecommendationSource | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "review":
        return _review(args.file, pipeline)
    print(f"'{args.command}' is not implemented yet.", file=sys.stderr)
    return 1


def _review(path: Path, pipeline: RecommendationSource | None) -> int:
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
    print(format_recommendation(recommendation))
    return 0


def _real_pipeline() -> RecommendationSource:
    from openai import OpenAI

    from case_mgmt_ai_demo.adapters import build_review_pipeline
    from case_mgmt_ai_demo.config import load_settings

    settings = load_settings()  # exports .env, so before OpenAI()
    return build_review_pipeline(settings, OpenAI())


def format_recommendation(recommendation: Recommendation) -> str:
    request = recommendation.request
    score = recommendation.approval_score
    verdict = VERDICTS_ES[recommendation.verdict]
    if recommendation.borderline:
        verdict += " (caso límite: revisar con detenimiento)"
    lines = [
        f"Recomendación — Solicitud {request.request_type} de {request.applicant_name}",
        f"Puntuación de aprobación: {'sin puntuación' if score is None else f'{score:.2f}'}",
        f"Veredicto: {verdict}",
        "",
        "Notas:",
    ]
    for number, note in enumerate(recommendation.notes, start=1):
        curable = " (subsanable)" if note.curable else ""
        lines.append(f"  {number}. {note.text} [{', '.join(note.section_ids)}]{curable}")
        for section_id in note.unretrieved_section_ids:
            lines.append(
                f"     ⚠ Cita no verificada: {section_id} no está entre las Secciones recuperadas"
            )
    return "\n".join(lines)
