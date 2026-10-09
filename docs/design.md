# Permit Review Assistant — Demo Design

Domain vocabulary lives in [`CONTEXT.md`](../CONTEXT.md). This document records the decisions reached during design grilling (2026-10-09).

## Purpose

A locally-run demo for stakeholders of an AI component in a case management system. A **Reviewer** opens a business **Permit Request**; the assistant produces a **Recommendation** (Approve / Not approve, **Approval Score**, cited **Notes**); the Reviewer records a **Determination**.

The corpus — Puerto Rico's Reglamento Conjunto (RC-2023-E) and the OGPe fee order (COBROS OA-2022-07), both in `docs_to_use/` — is a stand-in for the regional regulations of the eventual real use case. The design must make swapping the corpus a matter of re-ingestion plus new Criteria/fee YAML.

## Domain decisions

- The AI **recommends**; the Reviewer **decides**. Determination options: accept, override (reason required), Cure Request.
- Incomplete or fixable problems are expressed as **curable** Notes, mirroring the Reglamento's *subsanación* (Secciones 2.1.9.2(m), 3.1.3.2), not as denials.
- Content, Notes, and Request data are in **Spanish**; UI offers an **English toggle**.
- Request Types in scope: **Permiso de Construcción (PCOC)** and **Permiso Único**.

## Scope

### Request Types and key Sections

| Request Type | Sections | Fee items (COBROS Anejo A) |
|---|---|---|
| PCOC | 2.1.9.2 filing checklist, 2.1.9.3 work value, 2.1.9.5 standing, 2.2.1.3 ministerial parameters, 2.2.6 extensions | 6(a)–(c) |
| Permiso Único | Regla 3.7.1 (3.7.1.1–3.7.1.4), 4.1.4.1 licence documents | 28, 30, 41 |

The COBROS exemption table (TERCERO.3) applies to both.

### Permit Request fields (~10 each)

Shared core:
1. Applicant name
2. Applicant type — private / government / cooperative / non-profit (drives exemptions)
3. Applicant role — owner / lessee / option-holder / representative, plus standing document type (2.1.9.5)
4. Municipio + número de catastro
5. Calificación (zoning district)
6. Attached-documents checklist (per Request Type)

PCOC adds: work description; estimated work value ($); floors/height + gross floor area; declared fee ($).

Permiso Único adds: proposed use / business type; employees, parking, hours; licences requested (sanitaria, CPI, alcohol) + validity term (1–3 years); declared fee ($).

### Synthetic dataset

8 curated Permit Requests, each with an expected outcome and rationale: 3 clear approve, 2 clear not-approve (e.g. renewal > 2 years after expiry; use not allowed in the district), 2 borderline, 1 wrong fee. They double as the demo script and a regression suite.

## Pipeline (LangGraph)

```
intake ──► fee_check ─────────────────────────┐
   └────► retrieve ──► decide ──► write_notes ├─► validate_citations ──► END (Recommendation)
                                     ▲                │
                                     └── retry once ──┘
```

| Node | Behaviour |
|---|---|
| `intake` | Validate the Permit Request fields against its Request Type. |
| `fee_check` | Deterministic, from fee-rules YAML. Emits its own curable Note citing the Anejo A item. Does **not** affect the Approval Score. |
| `retrieve` | Pinned Sections for each Criterion + Chroma semantic top-k, metadata-filtered by Request Type. |
| `decide` | One OpenAI Decisions API call (`client.decisions.create`, model `gpt-6-luna`): one overall `predicate` (= Approval Score) plus one `predicate` per Criterion. Context goes in `input` (no system prompt; instructions are per question). |
| `write_notes` | `gpt-6-luna` with structured output: list of `{text, section_ids[], curable}`. Inputs: Request fields, retrieved Sections, Approval Score, Criterion probabilities, fee-check result. |
| `validate_citations` | Reject any citation whose Section prefix was not retrieved; retry `write_notes` once, then drop offending Notes. |

- **Verdict:** configurable cutoff on the Approval Score plus a configurable **Borderline** band (e.g. 0.4–0.6) flagged in the UI.
- **Failure handling:** a refused overall predicate yields "No score — manual review" (Notes still generated); a refused Criterion predicate is shown as "not assessed". API errors surface in the trace.
- **Determination is outside the graph** — plain app state in SQLite. (Revisit LangGraph `interrupt()` only if post-Determination steps are added.)
- Graph state per node feeds the in-app pipeline trace.

## Models

| Use | Model |
|---|---|
| Approval Score + Criterion probabilities | `gpt-6-luna` via Decisions API (public beta; requires `openai>=3.26.0`) |
| Notes | `gpt-6-luna` (structured output); model ID configurable |
| Embeddings | `text-embedding-3-large` |

All model IDs live in config. Credentials: the developer's own `OPENAI_API_KEY` in `.env` (no Azure).

Known unknowns about the Decisions API, to be resolved by the spike: Spanish support is not documented; probabilities are not documented as calibrated; individual questions can be refused without reason. Fallback if the spike fails: `gpt-6-luna` structured output with a self-reported probability.

## Data and configuration

### Chunk contract

The ingestion output every downstream component builds against:

| Field | Example / values |
|---|---|
| `section_id` | `RC-2023-E:3.7.1.2`, `COBROS:AnejoA:6` (canonical, unique per Section) |
| `doc` | `RC-2023-E` \| `COBROS-OA-2022-07` |
| `level` | `tomo` \| `capitulo` \| `regla` \| `seccion` \| `anejo_item` \| `glossary_term` |
| `heading` | `REQUISITOS DE PRESENTACIÓN` |
| `parents` | `["Tomo III", "Capítulo 3.7", "Regla 3.7.1"]` |
| `page_start`, `page_end` | source PDF pages |
| `text` | full Section text, incisos preserved |
| `part` | `1-of-N` when a long Section is split at inciso boundaries (same `section_id`) |

- Chunk at **Sección** level (Regla level when a Regla has no Secciones). Strip running headers/footers; skip the RC table of contents; index the Tomo XII glossary one chunk per term.
- Citations may go down to inciso (`RC-2023-E:2.1.9.2(m)`); validation matches on Section prefix.
- Implemented as a Pydantic model, with a ~10-chunk hand-typed fixture so the graph and UI can be built before real ingestion.

### Retrieval evaluation (part of ingestion)

Ingestion is not complete until retrieval has been evaluated; this validates both the chunking strategy and the PDF text extraction.

- **Golden set** — hand-curated Spanish queries, each labelled with the `section_id`(s) that answer it. Labels are Section IDs, not chunk IDs, so the same golden set compares different chunking strategies fairly (a split Section counts as a hit if any of its parts is retrieved). Coverage: every Criterion's pinned Sections, COBROS Anejo A items (tests table reconstruction), glossary terms, and queries phrased like Permit Request fields.
- **Metrics** — hit rate@k and recall@k (k = 5, 10) and MRR, broken down by document and by `level`.
- **Output** — a report printed after ingestion and saved alongside the index with the chunking configuration, so runs of different strategies can be compared.
- **Gate** — ingestion fails if recall@10 falls below a configured threshold.

### Curated YAML

- **Criteria** — 5–8 per Request Type: ID, Spanish + English text, pinned Section IDs, Decisions API predicate instruction.
- **Fee rules** — COBROS 6(a)–(c) (PCOC), 28, 30, 41 (Permiso Único), plus the TERCERO.3 exemption table.

## Stack

- Python 3.13, uv; single `uv run` entry point.
- LangGraph orchestration.
- Chroma (embedded) vector store, built by a one-off `ingest` command; index cached so the demo starts instantly.
- FastAPI + Jinja + HTMX UI.
- SQLite for Permit Requests, Recommendations, and Determinations.
- Recommendations for curated cases cached so the live demo cannot fail.
- LangSmith tracing enabled only when `LANGSMITH_API_KEY` is set; the in-app trace panel is primary.

## UI

1. **Queue** — Permit Requests with Request Type, Applicant, verdict badge, score.
2. **Case view** — Request fields; Approval Score gauge and verdict (Borderline flag); Notes with clickable citations opening the Section text in a side panel; fee check result; collapsible pipeline trace (retrieved Sections + scores, Criterion probabilities); Determination buttons (Accept / Override with reason / Cure Request).
3. **New request form** — create Permit Requests live.

## Ownership

- **Chunker / ingestion** — written together (developer + Claude), against the chunk contract.
- **Claude drafts, developer reviews** — Criteria YAML, fee-rules YAML, the 8 curated Permit Requests, the retrieval golden set.

## Build order

1. Decisions API spike: Spanish input, separation between clearly good and bad requests, refusal behaviour.
2. Chunk contract (Pydantic) + stub fixture.
3. Chunker/ingestion together, including the retrieval evaluation and its golden set; in parallel, Criteria YAML, fee-rules YAML, curated requests.
4. LangGraph pipeline against the fixture.
5. FastAPI + HTMX UI.
6. Swap in the real index; run the curated requests as a regression check.
