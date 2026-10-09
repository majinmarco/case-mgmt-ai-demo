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

PCOC adds: work description; estimated work value ($) and how it was calculated (OGPe guides or another method, 2.1.9.3(b)); floors/height + gross floor area; the district parameters of 2.2.1.3(d) (lot size, density, occupation area, yards, parking, loading), each beside the district's limit; how the plans are certified (professional, licence, drafter, sheet certifications, compliance certification, file format, north orientation; 2.1.9.2(k)); whether the OGPe catastral records match (2.1.9.2(j)); declared fee ($).

The shared core also records whether the owner authorises a lessee's or option-holder's action (2.1.9.5(b)).

**Evidence rule.** A Permit Request carries a field for every requirement that applies to its Request Type and that the record could evidence. The model judges an unevidenced requirement as unmet, so a good request that lacks the field to show compliance scores as a bad one (see *Requirement selection* below).

Permiso Único adds: proposed use / business type and whether it is ministerial or has an approved variation (3.7.1.1(f)); employees, parking (against the district's requirement), hours; licences requested (sanitaria, CPI, alcohol) + validity term (1–3 years, 3.7.1.3(a)); the site sketch and what it shows (area, internal layout, access road, parking, dimensions; 3.7.1.2(b)(3)); whether the site is in a flood zone (3.7.1.1(k)); the documents of 2.1.9.2(h)–(i), 3.7.1.2(b) and 4.1.4.1(b) in the attached-documents checklist; declared fee ($). A renewal adds the permit being renewed, its expiry and filing dates (shown as days before or after expiry), and the owner's or operator's certification that the use conforms (3.7.1.4(k)).

### Synthetic dataset

8 curated Permit Requests, each with an expected outcome and rationale: 3 clear approve, 2 clear not-approve (e.g. renewal > 2 years after expiry; use not allowed in the district), 2 borderline, 1 wrong fee. They double as the demo script and a regression suite.

## Pipeline (LangGraph)

```
intake ──► fee_check ───────────────────────────────────────────────┐
   └────► retrieve ──► select_requirements ──► decide ──► write_notes ├─► validate_citations ──► END (Recommendation)
                                                             ▲                │
                                                             └── retry once ──┘
```

| Node | Behaviour |
|---|---|
| `intake` | Validate the Permit Request fields against its Request Type. |
| `fee_check` | Deterministic, from fee-rules YAML. Emits its own curable Note citing the Anejo A item. Does **not** affect the Approval Score. |
| `retrieve` | Chroma semantic top-k, metadata-filtered by Request Type, plus the pinned Sections of each applicable Criterion fetched by ID (a pin that matches no retrieved Clause is a configuration error). Code splits each retrieved chunk into Clauses (top-level `a. `, `b. ` …), each identified as `section_id(letter)`, e.g. `RC-2023-E:2.1.9.2(k)`. |
| `select_requirements` | `gpt-6-luna` with structured output classifies **every** Clause for this Permit Request as `{ref, category, reason}`: `aplica` (a requirement the Applicant or the request must meet; it stays even when the request does not meet or mention it, because then it is a deficiency), `tramite_del_sistema` (a step the system or agency performs: SUI filing, fee payments, validation, agency deadlines), or `no_aplica` (does not apply to this request). The model only picks references; code copies the Clause text word for word. Unknown references are dropped and reported; Clauses the model leaves out are reported and kept for `decide`. General Clauses the model is unsteady on have their category fixed by configuration (`fixtures/fixed_clause_categories.json`) and are not asked. Clauses a Criterion pins are not asked either: they are always `aplica` (reason: the Criterion), whatever a fixed category says. |
| `decide` | OpenAI Decisions API (`client.decisions.create`, model `gpt-6-luna`), one call per question, made concurrently: the overall `predicate` (= Approval Score) over the `aplica` Clauses, and one `predicate` per applicable Criterion over that Criterion's pinned Clauses alone. Each `input` holds its Clauses verbatim, grouped under their Section heading, then the Request fields (no system prompt; instructions are per question). The overall question names the Request Type. |
| `write_notes` | `gpt-6-luna` with structured output: list of `{text, section_ids[], curable}`. Inputs: Request fields, the full retrieved Sections, Approval Score, Criterion probabilities, fee-check result. |
| `validate_citations` | Reject any citation whose Section prefix was not retrieved; retry `write_notes` once, then drop offending Notes. |

- **Adapters:** the Section retriever, the requirement selector, the Decisions API client and the Notes LLM are injected into the graph, so tests run it with fakes.
- **Selection in the trace:** the full classification (all three categories, with reasons) is kept in graph state and in the Recommendation; `review` prints how many Clauses were set aside and lists them with `--selection`.
- **Verdict:** configurable cutoff on the Approval Score plus a configurable **Borderline** band (e.g. 0.4–0.6) flagged in the UI.
- **Failure handling:** a refused overall predicate yields "No score — manual review" (Notes still generated); a refused Criterion predicate is shown as "not assessed". API errors surface in the trace.
- **Determination is outside the graph** — plain app state in SQLite. (Revisit LangGraph `interrupt()` only if post-Determination steps are added.)
- Graph state per node feeds the in-app pipeline trace.

### Requirement selection (CMAWD-4 finding)

In the first terminal slice (CMAWD-4), `decide` received whole retrieved Sections and the good PCOC example scored 0.24 (Not approve). Sección 2.1.9.2 alone has Clauses a–w, many of them procedural (SUI filing, the 10%/90% payments, validation deadlines) that a Permit Request record cannot evidence, and asked whether the request meets "todos los requisitos de las Secciones citadas" the model counts anything unevidenced as unmet. No single Section caused it, and rewording the question topped out at 0.46. Two changes were both needed: selecting the Clauses that apply (`select_requirements`), and Permit Request fields that can evidence them (the evidence rule above). Without fixed categories, the selector still varied run to run on the general Clauses (SUI filing, fee payments, completeness 2.1.9.2(m), 2.2.1.3(a)–(b)), and one bad run was enough to pull the good request below the cutoff; those Clauses are now fixed by configuration. With all three, the good example scores at or above the cutoff and the bad one below 0.1 (opt-in real-model test, three repeats each).

## Models

| Use | Model |
|---|---|
| Approval Score + Criterion probabilities | `gpt-6-luna` via Decisions API (public beta; requires `openai>=3.26.0`) |
| Requirement selection | `gpt-6-luna` (structured output); model ID configurable |
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
| `text` | full Section text, Clauses preserved |
| `part` | `1-of-N` when a long Section is split at Clause boundaries (same `section_id`) |

- Chunk at **Sección** level (Regla level when a Regla has no Secciones). Strip running headers/footers; skip the RC table of contents; index the Tomo XII glossary one chunk per term.
- Citations may go down to a Clause (`RC-2023-E:2.1.9.2(m)`); validation matches on Section prefix.
- Implemented as a Pydantic model, with a ~10-chunk hand-typed fixture so the graph and UI can be built before real ingestion.

### Retrieval evaluation (part of ingestion)

Ingestion is not complete until retrieval has been evaluated; this validates both the chunking strategy and the PDF text extraction.

- **Golden set** — hand-curated Spanish queries, each labelled with the `section_id`(s) that answer it. Labels are Section IDs, not chunk IDs, so the same golden set compares different chunking strategies fairly (a split Section counts as a hit if any of its parts is retrieved). Coverage: every Criterion's pinned Sections, COBROS Anejo A items (tests table reconstruction), glossary terms, and queries phrased like Permit Request fields.
- **Metrics** — hit rate@k and recall@k (k = 5, 10) and MRR, broken down by document and by `level`.
- **Output** — a report printed after ingestion and saved alongside the index with the chunking configuration, so runs of different strategies can be compared.
- **Gate** — ingestion fails if recall@10 falls below a configured threshold.

### Curated YAML

- **Criteria** — 5–8 per Request Type (`fixtures/criteria/<type>.yaml`): ID (also the Decisions API question name), Spanish + English text, pinned Clauses (or whole Sections), Decisions API predicate instruction, and the application kinds it applies to (new request, renewal, or both). PCOC: standing, environmental determination, site identification, plans, work value, district parameters; 2.2.6 (extensions) has no Criterion because a PCOC request here is a new request. Permiso Único: ministerial use, standing, components, memorial/sketch/photos and 4.1.4.1 certifications for new requests; 4.1.4.1 certifications, renewal deadline and renewal requirements for renewals.
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
