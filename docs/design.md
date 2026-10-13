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

PCOC adds: work description; estimated work value ($) and how it was calculated (OGPe guides or another method, 2.1.9.3(b)); floors/height + gross floor area; the district parameters of 2.2.1.3(d) (lot size, density, occupation area, yards, parking, loading), each beside the district's limit; how the plans are certified (professional, licence, drafter, sheet certifications, compliance certification, file format, north orientation; 2.1.9.2(k)); whether the OGPe catastral records match (2.1.9.2(j)); declared fee ($); whether the works are carried out wholly by the Government (COBROS TERCERO.3, read only by the Fee Check).

The shared core also records whether the owner authorises a lessee's or option-holder's action (2.1.9.5(b)).

**Evidence rule.** A Permit Request carries a field for every requirement that applies to its Request Type and that the record could evidence. The model judges an unevidenced requirement as unmet, so a good request that lacks the field to show compliance scores as a bad one (see *Requirement selection* below).

Permiso Único adds: proposed use / business type and whether it is ministerial or has an approved variation (3.7.1.1(f)); employees, parking (against the district's requirement), hours; licences requested (sanitaria, CPI, alcohol) + validity term (1–3 years, 3.7.1.3(a)); the site sketch and what it shows (area, internal layout, access road, parking, dimensions; 3.7.1.2(b)(3)); whether the site is in a flood zone (3.7.1.1(k)); the documents of 2.1.9.2(h)–(i), 3.7.1.2(b) and 4.1.4.1(b) in the attached-documents checklist; declared fee ($); the establishment type that sets the Licencia Sanitaria rate (COBROS Anejo A, 30, read only by the Fee Check). A renewal adds the permit being renewed, its expiry and filing dates (shown as days before or after expiry), and the owner's or operator's certification that the use conforms (3.7.1.4(k)).

### Synthetic dataset

8 curated Permit Requests, each with an expected outcome and rationale: 3 clear approve, 2 clear not-approve (e.g. renewal > 2 years after expiry; use not allowed in the district), 2 borderline, 1 wrong fee. They double as the demo script and a regression suite.

## Pipeline (LangGraph)

```
intake ──► fee_check ─────────────────────────────────────────────────────────────────────┐
   └────► retrieve ──► validate_sections ──► select_requirements ──► decide ──► write_notes ├─► validate_citations ──► END (Recommendation)
                                                                                   ▲                │
                                                                                   └── retry once ──┘
```

| Node | Behaviour |
|---|---|
| `intake` | Validate the Permit Request fields against its Request Type. |
| `fee_check` | Deterministic, from fee-rules YAML, in parallel with retrieval. Calculates the fee owed for the Request Type and Applicant type (each charge reduced by the TERCERO.3 exemption) and compares it with the declared fee. The result (declared fee, expected fee, each charge, exemption applied, cited items) is in the Recommendation and printed by `review`. A mismatch adds one curable Note citing the Anejo A item(s), plus `COBROS:TERCERO` when an exemption applied; so does a fee the request lacks the data to calculate (no work value, establishment type or floor area). Does **not** affect the Approval Score, and is not shown to `write_notes`. (CMAWD-6) |
| `retrieve` | Chroma semantic top-k (`PRA_RETRIEVAL_TOP_K`, 8 chunks), metadata-filtered by Request Type and limited to the body of RC-2023-E, plus the pinned Sections of each applicable Criterion fetched by ID (a pin that matches no retrieved Clause is a configuration error). A Section found by one of its parts comes with all of them. `PRA_RETRIEVER=fixture` reads the hand-typed fixture instead of the index. (CMAWD-11) Each retrieved Section's Clauses, each identified as `section_id(letter)`, e.g. `RC-2023-E:2.1.9.2(k)`, are read from the Clause store (below), which extracts them the first time the Section is needed. (CMAWD-20) |
| `validate_sections` | `gpt-6-luna` with structured output `{applies, reason}`, one call per retrieved Section (all its parts), concurrently: does this Section set a requirement this Permit Request must meet, given its Request Type and whether it is new or a renewal? Sections pinned by an applicable Criterion always apply and are not asked. A Section that does not apply stays in the Recommendation and trace with its reason, but its Clauses never reach `select_requirements` or `decide`, Notes are not given it, and a Note citing it fails `validate_citations`. (CMAWD-19) |
| `select_requirements` | `gpt-6-luna` with structured output classifies **every** Clause for this Permit Request as `{ref, category, reason}`: `aplica` (a requirement the Applicant or the request must meet; it stays even when the request does not meet or mention it, because then it is a deficiency), `no_verificable` (a requirement that applies, but that no field of the form could evidence: the prompt lists every field and what it can declare, from `form_es`, including the only documents the form admits; a field that exists and was left empty is still `aplica`), `tramite_del_sistema` (a step the system or agency performs: SUI filing, fee payments, validation, agency deadlines), or `no_aplica` (does not apply to this request). The model only picks references; code copies the Clause text word for word. Unknown references are dropped and reported; Clauses the model leaves out are reported and kept for `decide`. General Clauses the model is unsteady on have their category fixed by configuration (`fixtures/fixed_clause_categories.json`, optionally for some Request Types only, e.g. 2.1.9.2(k) sealed construction plans is `no_aplica` for Permiso Único) and are not asked. Clauses a Criterion pins are not asked either: they are always `aplica` (reason: the Criterion), whatever a fixed category says. |
| `decide` | OpenAI Decisions API (`client.decisions.create`, model `gpt-6-luna`), one `predicate` per **Requirement**, one call each, made concurrently. A Requirement is an applicable Criterion (over its pinned Clauses) or an applicable Section (over its remaining `aplica` Clauses that no Criterion pins; question named `seccion_<number>`, worded "¿Cumple esta Solicitud de {Request Type} con los requisitos citados de la Sección …?"). Each `input` holds its Clauses verbatim, grouped under their Section heading, then the Request fields (no system prompt; instructions are per question). **Approval Score = the lowest Requirement probability**: approval needs every requirement met, and the weakest one explains the score. (CMAWD-19) |
| `write_notes` | `gpt-6-luna` with structured output: list of `{text, section_ids[], curable}`. Inputs: Request fields, the Sections that apply, every Requirement with its probability, the Approval Score. One Note per Requirement below 0.5 or not assessed, then a few confirming what is met. |
| `validate_citations` | A Note is valid when it cites at least one Section and every citation's Section prefix (`RC-2023-E:2.1.9.2(m)` → `RC-2023-E:2.1.9.2`) is among the Sections that apply. If any Note is invalid, `write_notes` runs once more with the original prompt plus a correction listing each offending Note and its bad citations; its output replaces the first attempt's. Notes still invalid after the retry are dropped and kept in the Recommendation's `dropped_notes` (with the citations that failed), and `notes_retried` records the retry; `review` prints both. A failed retry keeps the first attempt's valid Notes. (CMAWD-7) |

- **Adapters:** the Section retriever, the Section validator, the requirement selector, the Decisions API client and the Notes LLM are injected into the graph, so tests run it with fakes.
- **Clause store:** a Section's Clauses do not depend on the Permit Request, so they are extracted once and kept in `data/clauses.sqlite` (`clause_store.py`), one entry per Section ID holding its Clauses (`ref`, `section_id`, `heading`, `text`) in Reglamento order, over all the Section's parts. An entry is used only while its fingerprint matches: a hash of the Section's parts (text and heading) and the extractor's `version`, so changed Section text or a changed extractor extracts again. The extractor is an injected adapter (`ClauseExtractor`: `version` and `extract(section)`); today it is the regex over top-level lettered paragraphs (`a. `, `b. ` …, `split_clauses`), and it is the seam for a model-based extractor where a Reglamento's layout defeats the regex, without paying for that extraction on every review. An extraction that fails (`ModelError`) stores nothing. When `ingest` syncs the index it also prunes the store: the Clauses stored for Sections of that document that are gone or changed are forgotten and the rest are kept, so a re-ingest does not throw away extractions that are still good. `ingest --clear-clauses` forgets every stored Clause of the documents ingested, and deleting the file forgets them all. Whether a Section applies (`validate_sections`) and how each Clause is classified (`select_requirements`) depend on the Applicant's input and are never stored. (CMAWD-20)
- **Selection in the trace:** the full classification (all four categories, with reasons) is kept in graph state and in the Recommendation; `review` prints how many Clauses were set aside and lists them with `--selection`.
- **Pipeline trace:** the Recommendation's `trace` holds the retrieved Sections (similarity to the request, and the Criteria that pin each) and every node's output in the order the nodes ran, as JSON, with Sections, Clauses and Criteria by their IDs. `review --trace` prints it: each retrieved Section with its similarity and its `validate_sections` verdict and reason (and, when it does not apply, the Clauses that were therefore never selected or scored), every Clause with its category and reason, each Requirement with its Clauses and probability, then each node's output. (CMAWD-11)
- **Verdict:** configurable cutoff on the Approval Score plus a configurable **Borderline** band (e.g. 0.4–0.6) flagged in the UI.
- **Failure handling:** any refused Requirement yields "No score — manual review" (Notes still generated), and that Requirement is shown as "not assessed"; so does a request with no applicable requirement. A model call that fails (adapters raise `ModelError` for OpenAI errors and missing or malformed structured output) is recorded as `{step, subject, message}` in the Recommendation's `errors` and printed by `review`; the verdict is "No score — manual review" (any score is still shown, never Borderline), and the review carries on with the safest reading: a failed Decisions call leaves its Requirement not assessed, a failed Section validation keeps the Section, a failed selection leaves every Clause unclassified (so kept for `decide`), and failed Notes leave none (a failed citation retry keeps the first attempt's valid Notes). (CMAWD-9) A failed embedding of the request leaves only the pinned Sections, which need no embedding. (CMAWD-11) A failed Clause extraction is recorded under `retrieve` with its Section, and that Section's lettered paragraphs are read by the regex instead, for this review only. (CMAWD-20)
- **Determination is outside the graph** — plain app state in SQLite. (Revisit LangGraph `interrupt()` only if post-Determination steps are added.)
- Graph state per node feeds the in-app pipeline trace (the Recommendation's `trace`).

### Requirement selection (CMAWD-4 finding)

In the first terminal slice (CMAWD-4), `decide` received whole retrieved Sections and the good PCOC example scored 0.24 (Not approve). Sección 2.1.9.2 alone has Clauses a–w, many of them procedural (SUI filing, the 10%/90% payments, validation deadlines) that a Permit Request record cannot evidence, and asked whether the request meets "todos los requisitos de las Secciones citadas" the model counts anything unevidenced as unmet. No single Section caused it, and rewording the question topped out at 0.46. Two changes were both needed: selecting the Clauses that apply (`select_requirements`), and Permit Request fields that can evidence them (the evidence rule above). Without fixed categories, the selector still varied run to run on the general Clauses (SUI filing, fee payments, completeness 2.1.9.2(m), 2.2.1.3(a)–(b)), and one bad run was enough to pull the good request below the cutoff; those Clauses are now fixed by configuration. With all three, the good example scores at or above the cutoff and the bad one below 0.1 (opt-in real-model test, three repeats each).

### Per-requirement scoring (CMAWD-5 finding, CMAWD-19)

With Criteria in place (CMAWD-5), the good Permiso Único example still scored 0.16–0.61 over three runs on the single overall predicate, while every Criterion scored 0.89–1.00. Leaving out one kept Clause at a time moved the score by at most 0.16: the overall question stays pessimistic over many requirements, as in CMAWD-4. Focused per-requirement predicates had been stable and correct throughout, so the overall predicate was replaced by per-Requirement predicates with the lowest probability as the Approval Score, and a `validate_sections` step was added so Sections retrieved but unrelated to the request never reach scoring. Taking the minimum makes one wrongly kept Clause decisive: the good Permiso Único scored 0.16 whenever the selector kept 2.1.9.2(k) (sealed construction plans), so that Clause is fixed `no_aplica` for Permiso Único. After that: good PCOC 0.68, good Permiso Único 0.62 (three of three runs), both bad examples 0.00.

### Retrieval from the real index (CMAWD-11)

`review` reads the Chroma index by default. The query is the request as its labelled Spanish fields, embedded with the model the index was built with; similarity is cosine similarity (1 minus Chroma's distance).

- **Only the body of RC-2023-E is searched.** Unrestricted, 7 of the 15 chunks nearest the good PCOC example were COBROS Anejo A items, and 8 of 15 for the good Permiso Único. Fees are the Fee Check's and never part of the Approval Score, and a glossary term defines a word rather than setting a requirement, so both are left out of the search. They stay in the index for citations.
- **The request as one query retrieves by topic, not by applicability.** Similarities are bunched (0.57–0.63 for the top 8 of both good examples), and what comes back shares the request's vocabulary: for the good PCOC, Secciones 3.2.1.2, 9.5.2.2, 9.2.3.2 and 10.2.11.7 (presentation requirements for construction works, electrical works and certain recommendations); for the good Permiso Único, 2.2.3.6, 9.12.4.3, 3.7.1.6, 3.7.1.9, 10.2.11.7 and 3.1.3. Of the Sections the Criteria pin, retrieval found two of the four for Permiso Único (3.7.1.2, 4.1.4.1) and none of the four for PCOC, so the pins are doing the work the design gave them.
- **The good examples no longer reach the cutoff** (one run each, 2026-10-13). `validate_sections` set aside 3 of the 8 retrieved Sections for each. The good PCOC scored **0.01**: its Criteria scored 0.59–1.00, but Secciones 3.2.1.2 (0.22), 9.5.2.2 and 9.2.3.2 (0.33) and 10.2.11.7 (0.01) ask for a flood certification, a soil study, a recycling plan, a georeferenced DXF and technical specifications, none of which the request has a field for. The good Permiso Único got **no score**: its Criteria scored 0.90–1.00, Sección 10.2.11.7 scored 0.06 and the question on Sección 3.1.3 was refused. This is the evidence rule again (an unevidenced requirement is judged unmet), now with Sections nobody curated. The trace shows it Clause by Clause. Not addressed in CMAWD-11; the options are Permit Request fields and Criteria for the Sections that do apply, tighter Request Type tags, or a similarity floor.
- **A Section's opening paragraph is often left unclassified.** The text before a Section's first Clause is cited by the Section ID alone, and in the good PCOC run the selector left it out in 9.5.2.2, 9.2.3.2 and 10.2.11.7, so it was kept for `decide` and reported.
- **Requirements the form cannot evidence are not scored.** Those Sections ask for things no field of the Permit Request can carry, so no Applicant could meet them through the form, and with the minimum as the score one such Clause sinks any request. The selector now classifies them `no_verificable`: they are set aside from `decide`, and `review` counts them above the Requirements and lists each with its reason after the Notes, since they remain requirements the Reviewer must check. `review` also names, under every Requirement, the Clauses it was assessed on; for a Requirement the Decisions API refused, which comes with no reason, it lists each of those Clauses with the selector's reason for keeping it. One run each (2026-10-13, real index): the good PCOC went from 0.01 to **0.56** (16 Clauses `no_verificable`, among them the flood certification, soil study, recycling plan and georeferenced polygon; weakest Requirements 9.5.2.2 and 10.2.11.7 at 0.56); the good Permiso Único's Sección 10.2.11.7 went from 0.06 to 0.81, though the run still had no score because the Decisions API refused Sección 3.1.3; both bad examples stayed at **0.00** (9 and 1 Clauses `no_verificable`). Clauses a Criterion pins are never asked, so they cannot be set aside this way.
- The opt-in real-model tests that check the good and bad examples' scores run on the fixture retriever, where those scores were established; one more reviews the good PCOC against the real index and checks the trace.

## Models

| Use | Model |
|---|---|
| Approval Score + Criterion probabilities | `gpt-6-luna` via Decisions API (public beta; requires `openai>=3.26.0`) |
| Requirement selection | `gpt-6-luna` (structured output); model ID configurable |
| Section validation | `gpt-6-luna` (structured output); model ID configurable |
| Notes | `gpt-6-luna` (structured output); model ID configurable |
| Embeddings | `text-embedding-3-large` |

All model IDs live in config. Credentials: the developer's own `OPENAI_API_KEY` in `.env` (no Azure).

Known unknowns about the Decisions API, to be resolved by the spike: Spanish support is not documented; probabilities are not documented as calibrated; individual questions can be refused without reason. Fallback if the spike fails: `gpt-6-luna` structured output with a self-reported probability.

## Data and configuration

### Chunk contract

The ingestion output every downstream component builds against:

| Field | Example / values |
|---|---|
| `section_id` | `RC-2023-E:3.7.1.2`, `COBROS:AnejoA:6` (canonical, unique per Section); `RC-2023-E:2.2.6` for a Regla, `RC-2023-E:1.2` for a Capítulo, `RC-2023-E:Tomo-II` for a Tomo; `RC-2023-E:Glosario:A.1` for a glossary term (letter and number as printed); `COBROS:TERCERO` for an ordinal of the fee order |
| `doc` | `RC-2023-E` \| `COBROS-OA-2022-07` |
| `level` | `tomo` \| `capitulo` \| `regla` \| `seccion` \| `anejo_item` \| `glossary_term` |
| `heading` | `REQUISITOS DE PRESENTACIÓN`; the term for a glossary term (`Acarreo`); the product for an Anejo A item |
| `parents` | `["Tomo III", "Capítulo 3.7", "Regla 3.7.1"]` |
| `page_start`, `page_end` | source PDF pages: physical, 1-based page index (not the printed footer number), also stored in the index metadata so a citation links to `<pdf>#page=N` |
| `text` | full Section text, Clauses preserved |
| `part` | `1-of-N` when a long Section is split at Clause boundaries (same `section_id`) |

- Chunk at **Sección** level (Regla level when a Regla has no Secciones). Strip running headers/footers; skip the RC table of contents; index the Tomo XII glossary one chunk per term.
- A Regla, Capítulo or Tomo is also a chunk for the text it holds directly (a Capítulo without Reglas, or a Regla's text before its first Sección), at its own `level`.
- `request_types` come from Section number prefixes in the ingestion configuration (`fixtures/ingestion/rc_2023_e.yaml`, `cobros_oa_2022_07.yaml`); a Section no prefix covers is tagged with both.

### Ingestion of RC-2023-E (CMAWD-8)

`ingest` extracts pp. 53–824 (Tomos I–XI), chunks them, embeds each chunk (parents, heading and text) with `text-embedding-3-large` and syncs the Chroma collection `reglamento`: only new or changed chunks are embedded, chunks no longer produced are removed. Each chunk's metadata holds every contract field except the text, so a Section ID resolves to its document and pages from the index alone. The configuration, counts and flagged chunks are written beside the index.

Findings about the PDF:

- **Text drawn as outlines.** The PDF was made with "Microsoft: Print To PDF", which drew some runs of text as vector outlines rather than text: 375 runs on about 160 body pages, nearly all on lines with typographic quotes. Text extraction returns nothing for them, which is the gap in Sección 2.1.9.5(a) on p. 80 (`…lo dispuesto en` / `Certi` / `designados en dicha Ley.`). Ingestion finds each run from the page's drawings and reads it from a full-page OCR (Tesseract, Spanish), keeping the PDF's own text everywhere else; a run OCR cannot read, or every run when OCR is unavailable, becomes `[…]`. OCR readings are not guaranteed (e.g. `(8')` read as `(87)` on p. 355), so the report lists them all.
- **A second OCR engine: Unlimited-OCR.** `ingest --ocr unlimited-ocr` reads the outlined runs with `baidu/Unlimited-OCR` (a 3B vision model, run in process on a GPU) instead of Tesseract. The model gives the text of an image without saying where each word is, so a run cannot be picked out of a page reading by position as with Tesseract. Each run is read from an image of the run together with the printed text on either side of it, and its reading is what lies between those two texts; when the model does not read the printed neighbours back as the PDF has them, the run stays `[…]`, so the neighbours check the reading. Straight double quotes in a reading are turned into the typographic ones the Reglamento prints. Tried on 41 runs (the 10 Tesseract leaves unread and every fourteenth of the rest), about 0.7 s per run on a Radeon 8060S: read with their line, 6 of the 10 came out right (`“AsBuilt”`, `“stemming”`, `Breakfast”`) and it corrected Tesseract misreadings (`7' 6''` where Tesseract read `7' 67`); the other 4 are table cells such as `24-29"` that the model returns as a LaTeX formula, which is taken as unread. Read on its own, without its neighbours, a run was often misread (`"loyers"` for `“layers”`), which is why the line is read. **Not yet run on the whole document**, so its counts of read and unread runs, and its flags, are not known; Tesseract stays the default until they are.
- **The running header** (`REGLAMENTO CONJUNTO | 25`) is printed at the foot of the page and is dropped by position.
- **Tomo cover pages** hold only outlined text and are skipped.
- **Numbering mistakes are kept as printed.** Regla 6.1.16 appears twice (I-P and ARD districts) and so does Sección 6.1.9.2; the second use gets a `-bis` suffix (`RC-2023-E:6.1.16.1-bis`, parent `Regla 6.1.16-bis`) so Section IDs stay unique. Secciones 4.3.10.2–4 sit under Regla 4.3.12 and `116.1.7` under Regla 11.6.1; these keep their printed number and are flagged.
- **Heading spellings** `CAPITULO`, `SECCION`, `SECCCIÓN` and a number printed as `6.1. 28.2` are accepted.
- **Tables** (about 140 body pages, mostly the district parameter tables of Tomo VI) rule off their rows but not always their columns, and a label cell is centred beside several lines of description, so reading straight across interleaves the cells. Ingestion takes a table's rows from its horizontal rules and its columns from the gutters no text crosses, and reads it cell by cell: each cell is its own line of chunk text (`Densidad poblacional` / `  a. Solares menores…`). Tables with three or more columns of free text are still not always read cleanly (e.g. Secciones 2.8.2.2 and 6.1.1.3) and are flagged.
- **Lettered sub-items** under another list item (`e. … / a. … / b. …`) or in a table cell are indented two spaces in the chunk text so they are not read as Clauses of the Section.

Extraction checks (warnings in the report, never a failure): text left unread (`[…]`); a line ending on a bare word followed by a line starting in lower case; running-header remnants; Clause letters out of order within a Section; Secciones out of sequence within a Regla; chunks under 40 characters. Wrapped lines are joined into paragraphs before the checks run, so text dropped in the middle of a paragraph shows up as `[…]`, not as a truncated line. The run with OCR on RC-2023-E raises 52 flags: 5 chunks with unread text, 15 truncated lines (multi-column tables), 24 lettering flags and 8 sequence flags. The lettering and sequence flags looked at so far are the Reglamento's own numbering (e.g. Regla 2.1.4 letters `a.`, `a.`, `b.`); the rest have not been checked against the PDF.
- Citations may go down to a Clause (`RC-2023-E:2.1.9.2(m)`); validation matches on Section prefix.
- Implemented as a Pydantic model, with a ~10-chunk hand-typed fixture so the graph and UI can be built before real ingestion.

### Ingestion of COBROS OA-2022-07 and the glossary (CMAWD-10)

`ingest` covers both documents, each with its own configuration, report and chunks file, and syncs the index one document at a time: ingesting COBROS adds its chunks and leaves RC-2023-E's alone, and RC-2023-E's body chunks are unchanged by the glossary being added, so nothing already indexed is embedded again.

**COBROS** (pp. 4–34; 68 chunks, no extraction flags):

- **One chunk per ordinal** of the order, PRIMERO to NOVENO (`COBROS:TERCERO`, level `seccion`, parent `Orden Administrativa OGPe 2022-07`), and **one per numbered item of Anejo A**, 1 to 57 (`COBROS:AnejoA:6`, level `anejo_item`, parents `Anejo A` and the division the item is under, e.g. `EDIFICABILIDAD`, or `DIVISION DE USOS`, `Usos` where the Anejo names a division over two rows). The recitals (POR CUANTO) before PRIMERO and the signature after NOVENO are not indexed.
- **Tables are read from their rules.** Anejo A and the exemption table of TERCERO.3 are ruled grids whose cells wrap over many lines, so text extraction interleaves the columns (and the reader CMAWD-8 wrote for RC-2023-E, which takes columns from gutters, does not see these cells). A row is the text between two horizontal rules; its cells are bounded by the vertical rules drawn beside that row, so a heading that spans the table is one cell. Each row is one line of chunk text, cells in column order and named after the table's header row: `a. Radicación (Cargo por servicio | Derechos a cobrarse: $ 0.35 | Concepto: Por cada $1,000 del valor de la obra o fracción. Pago mínimo de $100.00 Lo que resulte mayor`. The row's label (and the `#` column) are not named; empty cells are left out.
- **A lettered row is a Clause** of its item (`COBROS:AnejoA:6(a)`); a row without a letter (a further charge, a fee band) stays in the Clause above it. An item of a single row (46–48, 56) is that row.
- **Rows cut by a page break are put back together.** The first row on a page continues the row above when it has no rule over it (TERCERO) or when it has neither a label nor anything in the first fee column (Anejo A rules off the top of every page, so the rule tells nothing there). The second test is a judgement: a cut row whose fee cell also wraps onto the next page would be read as a new row. None does in this order.
- **Page range.** `page_start` is the page of the ordinal or of the item's heading row, `page_end` the page of its last row, so item 6 (heading and row (a) at the foot of p. 13, the rest on p. 14) is `13–14`.
- **Header and footer** are dropped by position: the order's number, title and page number above y=70, the office's address at or below y=938. The extraction checks also flag either if it turns up in a chunk.
- **Footnotes** (below a short rule at the foot of pp. 5, 30 and 34) are placed after the row or paragraph that carries their superscript number, not left at the page break where they would split a row.
- Text is kept as printed, including the order's own slips (`Radicación (Cargo por servicio` has no closing parenthesis; `Construccion`).

**Glossary** (Tomo XII, III. Glosario de términos, pp. 829–955; 1,431 chunks):

- **One chunk per term**, level `glossary_term`: the heading is the term and the text its definition (`RC-2023-E:Glosario:A.1`, heading `Acarreo`, parents `Tomo XII`, `Glosario de términos`). Terms are numbered from 1 under each letter, so the Section ID is the letter and the number as printed. Numbers the glossary uses twice under a letter (D.2, S.2, S.3, Z.1) get a `-bis` suffix, as in the body.
- **The term is the bold text the entry opens with**; extraction now records each line's runs of bold text. A name too long for its line continues on the next, and a dash or colon between term and definition belongs to neither. A term's number is not always at the margin (D.30 is printed further in), so a term is any numbered line left of the definitions' hanging indent.
- **Not indexed:** the introduction (I) and the list of acronyms (II, pp. 825–828), and the analysis that follows the terms (IV, from p. 957), which is not regulation.
- **Seven terms have no name found** and are flagged `unnamed_term`, with the name left at the start of the text: in C.31, E.25, E.38, E.102, S.92 and T.1 the name is partly drawn as outlines (read by OCR, or `[…]`), and I.3 is not set in bold. Without OCR a name cut short by unread text is not always caught: E.38 comes out named `E`, in a chunk already flagged `unread_text`.
- 66 runs of outlined text in these pages; OCR read all of them in the run made for this ticket. Without OCR 43 terms hold a `[…]`. About two dozen definitions are under 40 characters and flagged `short_chunk` (`Ver No Conforme.`); they are as printed.

### Retrieval evaluation (part of ingestion)

Ingestion is not complete until retrieval has been evaluated; this validates both the chunking strategy and the PDF text extraction.

- **Golden set** — hand-curated Spanish queries, each labelled with the `section_id`(s) that answer it. Labels are Section IDs, not chunk IDs, so the same golden set compares different chunking strategies fairly (a split Section counts as a hit if any of its parts is retrieved). Coverage: every Criterion's pinned Sections, COBROS Anejo A items (tests table reconstruction), glossary terms, and queries phrased like Permit Request fields.
- **Metrics** — hit rate@k and recall@k (k = 5, 10) and MRR, broken down by document and by `level`.
- **Output** — a report printed after ingestion and saved alongside the index with the chunking configuration, so runs of different strategies can be compared.
- **Gate** — ingestion fails if recall@10 falls below a configured threshold.

### Curated YAML

- **Criteria** — 5–8 per Request Type (`fixtures/criteria/<type>.yaml`): ID (also the Decisions API question name), Spanish + English text, pinned Clauses (or whole Sections), Decisions API predicate instruction, and the application kinds it applies to (new request, renewal, or both). PCOC: standing, environmental determination, site identification, plans, work value, district parameters; 2.2.6 (extensions) has no Criterion because a PCOC request here is a new request. Permiso Único: ministerial use, standing, components, memorial/sketch/photos and 4.1.4.1 certifications for new requests; 4.1.4.1 certifications, renewal deadline and renewal requirements for renewals.
- **Fee rules** — `fixtures/fee_rules.yaml`: COBROS 6(a)–(c) (PCOC: per $1,000 of work value or fraction, with the filing minimum), 28 (fire prevention, per year, by floor area), 30 (Licencia Sanitaria, per year, by establishment type), 41 (commercial use, plus alcohol sales) for Permiso Único, plus the TERCERO.3 exemption table as the percentage of each kind of charge (licences, services, tariffs, stamps) each Applicant type pays. Not curated: late renewal 42(c), 28(a) over 100,000 p.c., and 30(f)–(x); TERCERO.3 has no row for non-profits, so they pay in full.

## Stack

- Python 3.13, uv; single `uv run` entry point.
- LangGraph orchestration.
- Chroma (embedded) vector store, built by a one-off `ingest` command; index cached so the demo starts instantly. OCR of text the PDF draws as outlines needs Tesseract's Spanish model (optional), or the Unlimited-OCR model on a GPU (optional extra `unlimited-ocr`).
- FastAPI + Jinja + HTMX UI.
- SQLite for Permit Requests, Recommendations, and Determinations.
- Recommendations for curated cases cached so the live demo cannot fail.
- LangSmith tracing enabled only when `LANGSMITH_API_KEY` is set (loading the settings switches it off otherwise, whatever `LANGSMITH_TRACING` says); each review is one `permit_review` run. The in-app trace panel is primary.

## UI

1. **Queue** — Permit Requests with Request Type, Applicant, verdict badge, score.
2. **Case view** — Request fields; Approval Score gauge and verdict (Borderline flag); Notes with clickable citations opening the source PDF on the cited page (`#page=N` from the chunk's `page_start`) in a side panel; fee check result; collapsible pipeline trace (retrieved Sections + scores, Criterion probabilities); Determination buttons (Accept / Override with reason / Cure Request).
3. **New request form** — create Permit Requests live.

### Case view as built (CMAWD-13)

- **Storage.** A Permit Request and its Recommendation are each stored whole, as the JSON of their models, in `data/app.sqlite` (`store.py`). `serve` stores the requests in `examples/` when it has none.
- **Review on request.** A stored request without a Recommendation has a button; HTMX posts to `/solicitudes/{id}/recomendacion`, which runs the pipeline, stores the Recommendation and returns it as a fragment. The pipeline is built at the first review, so stored Recommendations are shown without an API key or an index.
- **Citations.** The app serves the two PDFs under `/reglamento/`. A citation's page is looked up in the index when the page is rendered, not stored: the `page_start` of the cited Section or, for a Clause of a split Section, of the part whose Clauses include it (a sub-item such as `3.7.1.2(b)(3)` goes by its Clause, `(b)`). A Section the index does not hold is plain text. The side panel is a new `iframe` on each click, because changing only `#page=N` of a PDF already shown does not move it; without JavaScript the link opens the PDF itself.
- **Gauge.** Drawn with the cutoff and Borderline band of the running configuration, which may differ from those a stored Recommendation's verdict was reached with.
- Spanish only; the English toggle is not built.

### New request form as built (CMAWD-15)

- **Fields.** `request_form.py` lists the form's fields for each Request Type. A field is named by the path of the Permit Request field it fills (`renewal.expiry_date`), so the form posts what a request file holds and an intake error names its input. A field left blank is not declared; a nested field (district parameters, plans certification, site sketch, renewal) is declared when any of its inputs is filled.
- **Request Type.** Choosing one makes HTMX fetch that type's fields (`/solicitudes/nueva/campos`) and swap them in; the shared fields stay as typed.
- **Validation.** The submission is validated by the same parser as pipeline intake, and each of its Spanish messages is shown under its field (status 422, the values kept). Nothing is stored and no review runs.
- **Submission.** A valid request is stored and reviewed in the same call, with an indeterminate progress indicator on the form while it runs (the pipeline does not report its steps), and then the case page opens. A review that fails leaves the request stored: its case page says why, once, and offers the review again.

### Queue and Determinations as built (CMAWD-14)

- **Queue.** `/` lists every stored Permit Request with its Request Type, Applicant, verdict badge, Approval Score, Borderline flag and Determination status ("Pendiente" until one is recorded; "Sin revisar" without a Recommendation), each linking to its case page.
- **Determination.** Recorded on the case page, in a card right after the Recommendation, by a plain form post to `/solicitudes/{id}/determinacion` that redirects back to the case page; a rejected one re-renders the page with the error (422). Stored whole as JSON in its own table, with the verdict of the Recommendation it was made on (`determination.py`, `store.py`).
  - **Accept** determines the recommended verdict.
  - **Override** needs a reason and determines the other verdict.
  - **Cure Request** lists deficiencies, one per line, in a text box pre-filled with the curable Notes and their citations, which the Reviewer may edit; it must list at least one.
  - A Recommendation without a verdict (manual review) cannot be accepted: the Reviewer names the outcome, Approve or Not approve, with a reason (or issues a Cure Request).
- **Final.** A Permit Request has one Determination: a second is refused (409), and so is reviewing the request again, since the Determination was made on the stored Recommendation. A request without a Recommendation cannot be determined. The store enforces both, so a review still running when the Determination is recorded does not replace the Recommendation either.

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
