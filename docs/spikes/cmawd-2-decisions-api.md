# CMAWD-2 spike: Decisions API with Spanish Permit Requests

**Date:** 2026-10-09 · **Model:** `gpt-6-luna` via `client.decisions.create` (openai SDK 3.27.0) · **Script:** `uv run python -m case_mgmt_ai_demo.spikes.decisions_api [--repeat N] [--wording-probe] [--refusal-probe] [--raw]`

Every number below comes from one run of `--repeat 3 --wording-probe --refusal-probe --raw` against the committed script, unless it is marked as from the earlier draft run.

## Verdict: GO

The Decisions API can produce the Approval Score and Criterion probabilities for Spanish Permit Requests. A clearly good and a clearly bad PCOC request separate widely (0.86 vs 0.00), every call returns the same result, and Criterion answers are confident and correct. The structured-output fallback is not needed and was not built.

One condition carries into the build: **the overall Approval Score is sensitive to small changes in the context**, while the Criterion scores are not. Restoring one phrase of one Section excerpt moved the good request from 0.53 to 0.86 (see [Context sensitivity](#context-sensitivity)). Treat the overall score as uncalibrated, and set the cutoff and Borderline band from the 8 curated Permit Requests with the real retrieval output, not with assumed values.

## Setup

- **Input** (one string): four RC-2023-E Section excerpts followed by ~9 Permit Request fields in Spanish. The excerpts are 2.1.9.2 g/i/j/k/m, 2.1.9.3 a–b, 2.1.9.5 a–b and 2.2.1.3 c–d. Each is word for word from the PDF text with line breaks joined, except that 2.1.9.2(g) keeps only its first sentence and 2.1.9.5(a) omits one sentence that pymupdf extracts garbled (`…lo dispuesto en Certi / designados en dicha Ley.`).
- **Good request:** owner with escritura pública, matching catastro, sealed plans, value certification, environmental determination, 2-floor house in an R-I district (max 2 floors). Lot size, density and setbacks are stated only as "según la hoja de calificación".
- **Bad request:** a neighbour with no standing document, mismatched catastro, unsealed sketch, no value certification, 4-floor mechanic's workshop in the same R-I district. It fails on many counts at once, so it is an easy negative; the 8 curated Permit Requests (which include Borderline cases) are the real test.
- **Questions:** an overall `aprobacion` predicate plus three Criterion predicates (`legitimacion_activa` → 2.1.9.5, `valor_estimado` → 2.1.9.3, `parametros_ministeriales` → 2.2.1.3). Each set was asked once with Spanish and once with English instructions.

## Results

| Question | Good, ES | Bad, ES | Good, EN | Bad, EN |
|---|---|---|---|---|
| `aprobacion` (overall) | **0.86** | 0.00 | **0.90** | 0.00 |
| `legitimacion_activa` | 1.00 | 0.00 | 1.00 | 0.00 |
| `valor_estimado` | 1.00 | 0.00 | 1.00 | 0.00 |
| `parametros_ministeriales` | 0.84 | 0.00 | 0.87 | 0.00 |

The three repeats were identical, and probabilities come back rounded to 2 decimals.

### Does Spanish input work?

Yes. Spanish Section text, Spanish request fields and Spanish question instructions all work. With the same request, Spanish instructions scored the two judgement questions 0.03–0.04 below English. That gap is small next to the [context sensitivity](#context-sensitivity) below. Language and exact phrasing can't be separated with two requests, so this spike can't say whether the gap comes from language or from wording.

### Does the good request score clearly higher than the bad one?

Yes: 0.86 vs 0.00 in Spanish and 0.90 vs 0.00 in English. The `real_model` test `tests/test_spike_decisions_api.py` guards this by requiring a gap of at least 0.5. With the context in the script, the call is deterministic and the margin is 0.36. A context change can eat that margin (see below), so treat a failure as a prompt to re-examine, not as flakiness.

### Context sensitivity

My first draft of the script had a mis-transcribed 2.1.9.5(a): it dropped the phrase "…por el Jefe de la Agencia o su representante autorizado en caso de un proyecto público." and added a full stop after "JP". That was the only difference. These numbers come from the earlier draft run, and that version of the excerpt was not committed.

| | Draft excerpt | Corrected excerpt |
|---|---|---|
| Good `aprobacion`, ES | 0.53 | 0.86 |
| Good `aprobacion`, EN | 0.78 | 0.90 |
| Good `parametros_ministeriales`, ES / EN | 0.77 / 0.85 | 0.84 / 0.87 |
| Good `legitimacion_activa`, `valor_estimado` | 1.00 | 1.00 |
| Bad, every question | 0.00 | 0.00 |
| Overall wording variants, good (ES) | 0.50–0.78 | 0.84–0.90 |

A small edit to text that the request doesn't even depend on (it is not a public project) moved the overall score by 0.33. With the draft excerpt, a clean request would have landed in the Borderline band (0.4–0.6). The well-defined Criterion questions barely moved, and the bad request stayed at 0.00.

### Wording sensitivity

Alternative phrasings of the overall predicate on the corrected context, run with `--wording-probe` as a single call:

| Name | Instruction | Good | Bad |
|---|---|---|---|
| `es_tal_como` (script default) | Según las Secciones del Reglamento citadas, ¿debe aprobarse esta Solicitud … tal como fue presentada? | 0.86 | 0.00 |
| `es_sin_tal_como` | Según las Secciones del Reglamento citadas, ¿debe aprobarse esta Solicitud …? | 0.84 | 0.00 |
| `es_cumple` | ¿Cumple esta Solicitud … con todos los requisitos de las Secciones del Reglamento citadas? | 0.90 | 0.00 |
| `es_solo_secciones` | Evaluando únicamente los requisitos de las Secciones … proporcionadas, ¿debe aprobarse …? | 0.88 | 0.00 |
| `en_only_sections` | Considering only the requirements in the provided Reglamento Sections, should this PCOC Permit Request be approved? | 0.75 | 0.00 |

On this context the Spanish phrasings agree within 0.06. The compliance form (`es_cumple`) scored highest in both runs. It asks about the cited requirements (closed-world) rather than whether to approve (open-world), and it is the suggested starting point for the `decide` node. The variants differ in meaning as well as wording, so this table can't rank them reliably.

### What a refusal looks like

A refused question keeps its position in `answers` and carries its `name`, or `null` if the question was unnamed. It has no probability and no reason, and the other questions in the call are unaffected. Raw response (`--refusal-probe --raw`, good request):

```json
{"answers":[{"name":"origen_nacional","type":"refusal"},{"name":"soborno","probability":0.0,"type":"predicate"},{"name":null,"type":"refusal"}],"model":"gpt-6-luna","usage":{"input_tokens":1355,"input_tokens_details":{"cache_write_tokens":0,"cached_tokens":0},"output_tokens":0,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":1355}}
```

| Probe question | Answer |
|---|---|
| Inferring national origin from the Applicant's name | refusal (both requests) |
| An unnamed question unrelated to the input (tomorrow's weather in Caguas) | refusal (both requests) |
| Would the reviewer accept a bribe | answered: 0.00 good / 0.03 bad |

None of the real Permit Request questions was refused in any run. **Always name questions**: an unnamed refusal can only be matched by its position. The design's handling stands: a refused overall predicate gives "No score — manual review", and a refused Criterion is shown as "not assessed".

## Token usage and cost per call

| Call | Input tokens | Output tokens | Cost | Latency |
|---|---|---|---|---|
| 4 questions, good request (ES) | 1,901 | 0 | ≈ $0.00019 | 0.5 s (3.8 s first call) |
| 4 questions, bad request (ES) | 1,880 | 0 | ≈ $0.00019 | 0.5–0.6 s |
| 4 questions, good request (EN) | 1,865 | 0 | ≈ $0.00019 | 0.5–2.8 s |
| 5 questions (wording probe), good request | 2,038 | 0 | ≈ $0.00020 | 1.2 s |
| 3 short questions (refusal probe), good request | 1,355 | 0 | ≈ $0.00014 | 1.1 s |

- Output tokens are always 0, and the response reports 0 reasoning tokens.
- Each question adds input tokens roughly in line with the length of its instructions, while the shared input is billed once per call. Batching all of a request's questions into one call is therefore the cheapest way to run them.
- `cached_tokens` stayed at 0 at these input sizes.
- **Pricing is unverified.** Cost is computed at $0.10 per 1M input tokens with output unbilled (`USD_PER_M_*` in the script). That rate comes from secondary reports ([eesel.ai](https://www.eesel.ai/blog/openai-decisions-api-pricing), [OpenRouter](https://openrouter.ai/compare/openai/gpt-6-luna-decisions)); [OrcaRouter](https://www.orcarouter.ai/blog/openai-decisions-api-gpt-6-luna) says OpenAI has not published a Decisions rate. Even at 10× that rate, a `decide` call with ~10 retrieved Sections (~5–8k tokens) would cost under a cent.
- The 16-call run above cost ≈ $0.003, by the script's own total.

## Consequences for the build

- The `decide` node uses the Decisions API as designed. No fallback is needed.
- Name every question after its Criterion ID. Start the overall predicate from the compliance wording (`es_cumple`).
- The overall Approval Score moves with the context, so the cutoff and Borderline band have to be calibrated on the 8 curated Permit Requests using the real retrieved Sections. Re-check them whenever chunking or retrieval changes. The Criterion probabilities are the more stable signal for Notes and the trace.
- Section text must be extracted faithfully, because extraction errors move the score. This is one more reason for the retrieval evaluation gate in ingestion.
