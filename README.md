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

Prints the Recommendation in Spanish: Approval Score (the weakest requirement's probability, named), verdict, every Requirement with its probability, and Notes with their cited Section IDs. `--selection` lists the retrieved Sections that do not apply, with reasons, and every Clause with its category. Both Request Types are accepted (PCOC and Permiso Único, new or renewal).

Sections come from the Chroma index that `ingest` builds (below), so ingest first: the 8 chunks of the body of RC-2023-E nearest the request (`PRA_RETRIEVAL_TOP_K`) among those tagged with its Request Type, plus the Sections its Criteria pin. `PRA_RETRIEVER=fixture` reads a 12-chunk RC-2023-E fixture instead (`src/case_mgmt_ai_demo/fixtures/`; `[…]` in it marks text the PDF extraction garbles or drops), which needs no index.

`--trace` prints the pipeline trace the Recommendation carries: each retrieved Section with its similarity to the request and whether it applies, and why; every Clause with its category and reason; each Requirement with its Clauses; and what each node of the pipeline returned. With `LANGSMITH_API_KEY` set, each review is also traced in LangSmith (as `permit_review`); without it nothing is sent.

The example requests were written against the fixture. Against the real index the good ones did not reach the cutoff (one run each): retrieval brings Sections they have no fields to evidence. See *Retrieval from the real index* in `docs/design.md`.

## Ingest the Reglamento

```sh
uv run case-mgmt-ai-demo ingest              # extract, chunk, embed and index both documents
uv run case-mgmt-ai-demo ingest --no-index   # chunks and reports only: no API calls
uv run case-mgmt-ai-demo ingest --doc COBROS-OA-2022-07   # one document only
```

Ingests RC-2023-E (Tomos I–XI by Section, and the Tomo XII glossary one chunk per term) and the fee order COBROS OA-2022-07 (one chunk per ordinal of the order and per item of Anejo A). Builds the Chroma index in `data/chroma` and writes, next to it, for each document, `data/<doc>.ingestion_report.json` (the configuration used, counts, and every chunk flagged by the extraction checks) and `data/<doc>.chunks.jsonl` (every chunk, to read or diff). The index is synced one document at a time and only chunks that changed are embedded, so running it again, or ingesting one document, leaves the rest of the index as it was. Page ranges, the split length and the Request Type tags are in `src/case_mgmt_ai_demo/fixtures/ingestion/` (`rc_2023_e.yaml`, `cobros_oa_2022_07.yaml`).

The RC-2023-E PDF draws some text as outlines instead of text (mostly lines with typographic quotes, on about 160 pages), which plain extraction skips. With Tesseract's Spanish model installed (`spa.traineddata`, e.g. the `tesseract-data-spa` package; `PRA_TESSDATA_DIR` if it is not in the system tessdata directory) ingestion reads those runs with OCR, which takes about five minutes; without it, or with `--no-ocr`, each run is left as `[…]` and its chunk is flagged. The report lists every OCR reading so it can be checked against the PDF.

Criteria are curated in `src/case_mgmt_ai_demo/fixtures/criteria/` (one YAML file per Request Type).

Model IDs, the Approval Score cutoff, the Borderline band and paths default in `src/case_mgmt_ai_demo/config.py` and can be overridden with `PRA_*` variables (listed in `.env.example`).
