# Permit Review Assistant demo

An AI assistant that reviews business Permit Requests against the Puerto Rico Reglamento and produces a cited Recommendation for a human Reviewer. See [`docs/design.md`](docs/design.md) and [`CONTEXT.md`](CONTEXT.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.13 is installed automatically).

```sh
uv sync
cp .env.example .env   # then set OPENAI_API_KEY
uv run pytest          # real-model tests skipped; RUN_REAL_MODEL_TESTS=1 to include them
uv run case-mgmt-ai-demo --help   # subcommands: review, ingest, serve
```

## Review a Permit Request

```sh
uv run case-mgmt-ai-demo review examples/pcoc-buena.json
uv run case-mgmt-ai-demo review examples/permiso-unico-buena.json
```

Prints the Recommendation in Spanish: Approval Score (the weakest requirement's probability, named), verdict, every Requirement with its probability, and Notes with their cited Section IDs. `--selection` lists the retrieved Sections that do not apply, with reasons, and every Clause with its category. Both Request Types are accepted (PCOC and Permiso Único, new or renewal). Sections come from a 12-chunk RC-2023-E fixture (`src/case_mgmt_ai_demo/fixtures/`) until ingestion exists. `[…]` in the fixture marks text the PDF extraction garbles or drops.

Criteria are curated in `src/case_mgmt_ai_demo/fixtures/criteria/` (one YAML file per Request Type).

Model IDs, the Approval Score cutoff, the Borderline band and paths default in `src/case_mgmt_ai_demo/config.py` and can be overridden with `PRA_*` variables (listed in `.env.example`).
