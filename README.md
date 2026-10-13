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

## Review in the browser

```sh
uv run case-mgmt-ai-demo serve               # http://127.0.0.1:8000 (--host, --port)
```

Starts the Reviewer web UI (FastAPI, Jinja and HTMX, in Spanish). The page of a Permit Request shows its fields and its Recommendation: the Approval Score on a gauge with the cutoff and the Borderline band, the verdict (or "Sin puntuación — revisión manual"), every Requirement with its probability (the weakest one marked; "no evaluado" where the question was refused), the retrieved Sections that do not apply, the Notes (curable ones marked "Subsanable"), the Fee Check, and the pipeline trace, collapsed.

Each citation is a link to the Reglamento PDF, which the app serves, on the cited Section's page (`#page=N`, from `page_start` in the index; for a Clause of a split Section, the part that holds the Clause). It opens in a panel beside the page, with a link to open it in a new tab. A citation whose Section is not in the index is plain text, so ingest first to get the links.

Permit Requests, Recommendations and Determinations are kept in SQLite (`data/app.sqlite`). The first time it runs, `serve` stores the Permit Requests in `examples/` (`PRA_EXAMPLES_DIR`). A request without a Recommendation has a button that reviews it (real model calls, as `review` makes) and stores the Recommendation; stored Recommendations are shown without an API key.

The start page is the queue: every Permit Request with its Request Type, Applicant, verdict, Approval Score, Borderline flag and Determination. On the case page the Reviewer records the Determination, once: accept the Recommendation, override it (a reason is required), or issue a Cure Request, whose list of deficiencies starts from the curable Notes. A request with a Determination is not reviewed again.

"Nueva Solicitud de Permiso" on the start page opens a form that creates a Permit Request: choosing the Request Type swaps in that type's fields and its documents checklist. Submitting validates it as `review` does (each error is shown under its field), stores it, reviews it (real model calls) and opens its case page.

## Ingest the Reglamento

```sh
uv run case-mgmt-ai-demo ingest              # extract, chunk, embed and index both documents
uv run case-mgmt-ai-demo ingest --no-index   # chunks and reports only: no API calls
uv run case-mgmt-ai-demo ingest --doc COBROS-OA-2022-07   # one document only
uv run case-mgmt-ai-demo ingest --ocr unlimited-ocr       # read outlined text with Unlimited-OCR
```

Ingests RC-2023-E (Tomos I–XI by Section, and the Tomo XII glossary one chunk per term) and the fee order COBROS OA-2022-07 (one chunk per ordinal of the order and per item of Anejo A). Builds the Chroma index in `data/chroma` and writes, next to it, for each document, `data/<doc>.ingestion_report.json` (the configuration used, counts, and every chunk flagged by the extraction checks) and `data/<doc>.chunks.jsonl` (every chunk, to read or diff). The index is synced one document at a time and only chunks that changed are embedded, so running it again, or ingesting one document, leaves the rest of the index as it was. Each Section's Clauses are extracted the first time a review needs them and kept in `data/clauses.sqlite`; `ingest` forgets those of Sections that changed or are gone, and with `--clear-clauses` every one of the documents ingested (deleting the file forgets them all). Page ranges, the split length and the Request Type tags are in `src/case_mgmt_ai_demo/fixtures/ingestion/` (`rc_2023_e.yaml`, `cobros_oa_2022_07.yaml`).

The RC-2023-E PDF draws some text as outlines instead of text (mostly lines with typographic quotes, on about 160 pages), which plain extraction skips. With Tesseract's Spanish model installed (`spa.traineddata`, e.g. the `tesseract-data-spa` package; `PRA_TESSDATA_DIR` if it is not in the system tessdata directory) ingestion reads those runs with OCR, which takes about five minutes; without it, or with `--no-ocr`, each run is left as `[…]` and its chunk is flagged. The report lists every OCR reading so it can be checked against the PDF. On a terminal, a bar shows how many pages have been extracted and then how many chunks embedded.

`--ocr unlimited-ocr` (or `PRA_OCR_ENGINE=unlimited-ocr`) reads those runs with [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR), a 3B vision model, instead of Tesseract. It needs a GPU and `uv sync --extra unlimited-ocr`; the first run downloads about 7 GB of weights. The model's own Python code is run from Hugging Face, pinned to one revision (`REVISION` in `unlimited_ocr.py`). `pyproject.toml` takes PyTorch from the ROCm index (AMD GPUs); on an NVIDIA machine remove the three `tool.uv` tables marked there. Only the outlined runs are read, each as an image of the run with the printed text beside it, never whole pages. It has been tried on 41 runs of RC-2023-E, not yet on the whole document: see `docs/design.md`.

Criteria are curated in `src/case_mgmt_ai_demo/fixtures/criteria/` (one YAML file per Request Type).

Model IDs, the Approval Score cutoff, the Borderline band and paths default in `src/case_mgmt_ai_demo/config.py` and can be overridden with `PRA_*` variables (listed in `.env.example`).
