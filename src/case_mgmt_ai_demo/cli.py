"""The single `uv run case-mgmt-ai-demo` entry point."""

import argparse
import sys

COMMANDS = {
    "review": "Produce a Recommendation for a Permit Request.",
    "ingest": "Chunk and index the Reglamento, then evaluate retrieval.",
    "serve": "Run the Reviewer web UI.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="case-mgmt-ai-demo", description="Permit Review Assistant demo."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in COMMANDS.items():
        subcommands.add_parser(name, help=help_text)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"'{args.command}' is not implemented yet.", file=sys.stderr)
    return 1
