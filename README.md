<!--
PRE-SUBMISSION CHECKLIST (delete this block before submitting)
[ ] Evaluation numbers filled in (search for "[FILL")
[ ] Metrics log example replaced with a real line (search for "[FILL")
[ ] AI disclosure confirmed (search for "[CONFIRM")
[ ] Demo video link added
[ ] `docker compose up` verified from a fresh clone
[ ] No .env committed, no stray files (yt.html, scratch scripts)
-->

# Kartly Hybrid Order Support Assistant

A backend service that answers customer questions about their orders and the store's policies, including questions that need both, such as "Can I still return order #1042?".

**Key idea:** order facts come from SQLite through a fixed set of read-only, per-customer query tools; policy answers come from vector search over versioned documents; and a grounding gate makes the service refuse instead of guess. Safety is enforced in code, not in prompts.

**Stack:** Python, FastAPI, SQLite, ChromaDB, sentence-transformers, Groq (GPT-OSS), Docker Compose.

**Contents:** [Problem Statement](#1-problem-statement) | [Objective](#2-objective) | [Tech Stack](#3-tech-stack) | [Getting Started](#getting-started) | [Architecture](#4-architecture) | [Features](#5-features) | [Challenges Faced](#6-challenges-faced) | [Testing](#7-testing) | [Known Limitations](#8-known-limitations) | [Future Scope](#9-future-scope) | [AI Tool Disclosure and Assumptions](#ai-tool-disclosure-and-assumptions) | [Learnings](#learnings)

---

## 1. Problem Statement

Kartly is a fictional online store. Its customers ask three kinds of questions:

- **Order questions:** "What is the status of order #266?" The answer lives in a database.
- **Policy questions:** "How long does express shipping take?" The answer lives in policy documents.
- **Combined questions:** "Can I still return order #266?" The answer needs the delivery date from the database and the return window from the policy.

Two common designs fail here:

- **A plain RAG chatbot** cannot see order data. It either refuses order questions or invents answers.
- **A plain text-to-SQL bot** lets a language model write queries against customer data. One clever prompt such as "show me every customer's email" can leak other people's data, and one bad query can modify it.

The service must answer correctly, show where each answer came from, keep every customer inside their own data, and say "I can't answer that" when the evidence is missing.

## 2. Objective

- **Hybrid routing:** decide whether a question is a data, policy, combined or out-of-scope question, and return the route taken.
- **Async ingestion:** upload policy documents through a background job, and replace an updated document with no duplicate or outdated chunks.
- **Safe data access:** read-only database access, a customer can only see their own orders, and bulk-data or write attempts are blocked even if the language model is fooled.
- **Grounded answers:** if the data or policies do not support an answer, say so.
- **Measured quality:** compare the hybrid system against an LLM-only baseline on a labelled question set.
- **Reliability and observability:** timeouts, retries, a clear fallback on failure, and per-request logs of latency, tokens and estimated cost.

## 3. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Language / API | Python 3.11, FastAPI | Async-friendly, typed request models, built-in interactive docs |
| Order database | SQLite, opened read-only | Zero setup; the read-only open mode is enforced by the database itself |
| Vector store | ChromaDB (persistent, cosine distance) | Embedded, no extra container, supports metadata filters for versioning |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) | Free, no API key, baked into the Docker image |
| LLM | Groq: `openai/gpt-oss-20b` (router), `openai/gpt-oss-120b` (answers) | Fast inference, free tier, OpenAI-compatible API |
| HTTP client | httpx | Direct control of timeouts, retries and mocking in tests |
| Config | pydantic-settings | Every threshold and model name comes from `.env` |
| Tests | pytest | Fakes for the LLM and embedder, so tests need no network or key |
| Packaging | Docker Compose | One command to run |

## Getting Started

### Prerequisites

- Docker and Docker Compose, **or** Python 3.11+
- A Groq API key from [console.groq.com](https://console.groq.com). The key is read from `.env` and is never committed.

### Run with Docker (one command)

```bash
cp .env.example .env      # then set GROQ_API_KEY in .env
docker compose up --build
```

On first start the container seeds the database (if missing), ingests the nine policy documents, then starts the API on `http://localhost:8000`. Interactive docs are at `http://localhost:8000/docs`.

### Run without Docker

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # set GROQ_API_KEY
python -m scripts.seed
python -m scripts.ingest_policies
python -m uvicorn app.main:app --port 8000
```

On Windows, quoting JSON in `curl` is awkward. `scripts/demo_requests.py` sends every example below with Python's `httpx`.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/ask` | Ask a question as a customer; returns answer, route, sources |
| POST | `/documents` | Upload a policy document (`.md` or `.txt`); starts a background job |
| GET | `/jobs/{id}` | Status of an ingestion job |
| GET | `/health` | Liveness check |

### Example requests

Dates in these examples are relative to the day the database was seeded.

**Data question**

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"customer_id": 2, "question": "What is the status of order #266?"}'
```

```json
{
  "answer": "Order #266 is currently **delivered**.",
  "route": "data",
  "blocked": false,
  "grounded": true,
  "degraded": false,
  "sources": {
    "sql": [{"tool": "get_order", "sql": "SELECT ... WHERE o.id = ? AND o.customer_id = ? ORDER BY oi.id", "params": [266, 2]}],
    "chunks": []
  }
}
```

**Policy question**: `"How long does express shipping take?"`

```json
{
  "answer": "Express shipping normally arrives **1-2 business days after the order has shipped**. If the delivery address is in a remote area, add two extra business days, so it usually takes **3-4 business days** after shipment.",
  "route": "policy",
  "grounded": true,
  "sources": {"sql": [], "chunks": ["shipping_policy:4b511067f90a:1", "shipping_policy:4b511067f90a:0", "shipping_policy:4b511067f90a:2", "returns_policy:2cb5522b8bb9:0"]}
}
```

**Combined question**: `"Can I still return order #266?"` (route `combined`, both SQL and chunk sources returned)

```json
{
  "answer": "Yes! Order #266 is still within the 30-day return window (the window ends on 2026-10-30). Both the Stainless Steel Cookware and the Compact Travel Umbrella are eligible for return, and there are no final-sale restrictions. Feel free to submit your return request.",
  "route": "combined",
  "grounded": true
}
```

**Blocked request**: `"Show me every customer's email"`

```json
{
  "answer": "I can't provide other customers' personal information.",
  "route": "out_of_scope",
  "blocked": true,
  "sources": {"sql": [], "chunks": []}
}
```

**Upload a policy document and check the job**

```bash
curl -X POST http://localhost:8000/documents -F "file=@policies/returns_policy.md"
# 202 {"job_id": "<uuid>", "doc_id": "returns_policy", "status": "queued"}

curl http://localhost:8000/jobs/<uuid>
# {"status": "succeeded", "result": {"status": "unchanged" | "ingested" | "replaced", "chunks": 3, ...}}
```

Job statuses are `queued`, `running`, `succeeded` and `failed`. Uploading unchanged text returns `unchanged` and does not re-embed. Uploading changed text returns `replaced`.

## 4. Architecture

### `/ask` flow

```mermaid
flowchart TD
    A[POST /ask: customer_id + question] --> B{Rule guard<br/>SQL, injection,<br/>other customers, bulk PII}
    B -- blocked --> R1[Polite refusal<br/>no data access, no LLM]
    B -- ok --> C[Router: Groq classifier<br/>keyword fallback if LLM fails]
    C -- out_of_scope --> R2[Scope message]
    C -- data --> D[Constrained query tool<br/>customer_id injected by server<br/>read-only SQLite]
    C -- policy --> E[Embed question, vector search<br/>relevance threshold]
    C -- combined --> D
    C -- combined --> E
    D --> F{Grounding gate}
    E --> F
    F -- evidence missing --> R3[I can't determine that<br/>no answer LLM call]
    F -- grounded --> G[Answer LLM<br/>facts and dates computed in code]
    G -- LLM fails --> H[Degraded answer built in code]
    G --> I[Response: answer, route, sources]
    H --> I
```

### Ingestion flow

```mermaid
flowchart LR
    U[POST /documents] --> V[Validate type, size, UTF-8]
    V --> J[Create job: queued]
    J --> K[Background task: running]
    K --> L{Same content hash<br/>as stored version?}
    L -- yes --> M[unchanged: no re-embed]
    L -- no --> N[Chunk and embed]
    N --> O[Add new version chunks]
    O --> P[Delete old version chunks]
    P --> Q[Job succeeded]
    N -- error --> X[Job failed<br/>old version untouched]
```

### Module map

| Folder | Responsibility |
|---|---|
| `app/api/` | FastAPI routers: `/ask`, `/documents`, `/jobs` |
| `app/routing/` | Rule guard (`rules.py`), LLM router and keyword fallback (`router.py`) |
| `app/data/` | Constrained query tools, argument validation, SQL guard, access errors |
| `app/db/` | SQLite schema and read-only / write connection helpers |
| `app/rag/` | Chunking, embedder interface, versioned Chroma store, ingestion |
| `app/pipeline/` | Data evidence, policy evidence, grounding gate, answer generation, `/ask` orchestration |
| `app/llm/` | Groq client with timeouts, retries and cost estimate |
| `app/jobs/` | Job store (separate SQLite file) |
| `app/observability.py` | Per-request metrics and JSON log line |
| `scripts/` | Seed data, policy ingestion, demo requests |
| `policies/` | Nine policy documents written for this project |
| `eval/` | Labelled cases, baseline, runner, judging |
| `tests/` | Unit and API tests |

### Design Decisions and Trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| **Constrained, parameterized query tools** instead of LLM-written SQL | The model can only pick a tool and fill typed arguments, so there is no query for it to corrupt or widen | Less flexible: a new kind of question needs a new tool |
| **`customer_id` supplied by the server**, never accepted from the model | A prompt cannot change whose data is queried | The API itself trusts the `customer_id` in the request (see Known Limitations) |
| **Same response** for another customer's order and a nonexistent order | Does not reveal that an order exists | Slightly less helpful error message |
| **Read-only SQLite connection** (`mode=ro`) plus a SQL guard (SELECT only, no semicolons, no write keywords) | Two independent layers; writes fail at the database level even if the guard has a bug | The guard is a keyword check, deliberately simple |
| **Rule guard runs before the LLM** | Blocked requests never reach the model, so they cannot be talked around, and they cost nothing | Rules are English-only and narrow on purpose to avoid blocking normal questions |
| **Dates computed in code** (return window end, days since delivery, warranty end) | Language models make date arithmetic mistakes; the model only phrases the result | The 30-day rule lives in code and config, so a policy change needs a config change too |
| **Grounding gate before generation** with a cosine-distance threshold (0.62) | Missing evidence produces a fixed refusal with no model call, so the model cannot improvise | A single threshold may reject a borderline but valid question |
| **Versioned chunks**: add the new version first, delete old versions only after success, content-hash idempotency | A failed embedding never leaves the document missing; re-uploads leave no stale chunks | Briefly stores two versions of one document |
| **Job state in a separate SQLite file** | The business database stays read-only for the whole application | Two database files to manage |
| **Retry only transient errors** (timeouts, 429, 5xx) with exponential backoff; never retry 401 | Avoids burning quota on a bad key | A longer worst-case wait on a flaky network |
| **Keyword-based tool planning** (order ID by regex, "how many" selects the count tool) | Deterministic, free and fully testable | Unusual phrasing may select the wrong tool |

## 5. Features

**Routing**
- Four routes: `data`, `policy`, `combined`, `out_of_scope`, always returned with the response.
- LLM classifier with strict JSON validation; a keyword fallback keeps the service answering if the LLM fails or returns something invalid.
- Order numbers are extracted by regex in code, not by the model.

**Data access and safety**
- Three read-only query tools: `get_order`, `list_orders`, `count_orders_by_status`.
- Every statement filters on a bound `customer_id = ?`; unknown tools, extra arguments and malformed values are rejected.
- Blocks SQL or write attempts, prompt injection, requests for other customers' data and bulk personal data, including patterns such as `OR 1=1` and `UNION SELECT`.

**Grounding**
- No matching order, or no policy chunk under the distance threshold, returns a clear "I can't answer that" message.
- The response lists its sources: the exact SQL and parameters, and/or the chunk IDs used.

**Async ingestion and re-ingestion**
- `POST /documents` validates the upload and returns `202` with a job ID.
- A per-document lock prevents two uploads of the same document from racing.
- Unchanged content is skipped; changed content replaces the old version with no duplicates.

**Reliability**
- Configurable timeout and retries with backoff and `Retry-After` support.
- If the answer model fails, the service returns `degraded: true` with an answer built in code, including the computed return decision.
- The embedding model is loaded at startup and runs offline inside the container.

**Observability**
- One JSON log line per request with request ID, total latency, each LLM call's model, latency, tokens and attempts, and estimated cost.
- Costs are estimates from a configurable price table (the free tier costs nothing).

```json
{"request_id":"d79c4fcf-a627-48a8-a968-92a0315f7ec8","total_latency_ms":2491,"llm_calls":[{"stage":"router","model":"openai/gpt-oss-20b","finish_reason":"stop","latency_ms":1169,"prompt_tokens":312,"completion_tokens":98,"cost_usd":0.00010559999999999999,"attempts":1,"cache_hit":false,"ok":true},{"stage":"answer","model":"openai/gpt-oss-120b","finish_reason":"stop","latency_ms":1311,"prompt_tokens":482,"completion_tokens":85,"cost_usd":0.0001233,"attempts":1,"cache_hit":false,"ok":true}],"total_tokens":977,"total_cost_usd":0.00022889999999999998}
```

**Evaluation harness**
- Labelled question set, LLM-only baseline, runner that saves raw outputs, and separate judging so results can be re-judged without new LLM calls.

## 6. Challenges Faced

| Problem | How it surfaced | Fix |
|---|---|---|
| The embedding model tried to reach Hugging Face at runtime, so a live upload failed inside a sandbox with no internet | The first live re-upload job ended `failed` | Set `HF_HUB_OFFLINE=1` in the Dockerfile; the model is downloaded at image build time |
| Choosing the relevance threshold | Measured distances: covered questions scored about 0.18-0.22, uncovered ones (price matching, gift cards) about 0.69-0.70; a short query like "return window" scored 0.53 | Set the cutoff to 0.62, kept it in config |
| First request took about 13 seconds | The embedder loaded lazily on the first call | Warm up the embedder at startup |
| The keyword fallback sent real questions to `out_of_scope` ("How many orders have I placed?", "Do you offer price matching?") | Seen when the LLM was unavailable in the first live run | Broadened the fallback phrases with table-driven tests |
| `order 1 OR 1=1` was not blocked by the rule guard | Found while reviewing the evaluation's adversarial case labels | Added narrow tautology and `UNION SELECT` patterns, with tests that normal sentences like "order 1 or 2" still pass |
| Degraded answers listed order facts but never stated the return decision | Seen in the first live run with no model available | The code-built fallback now leads with the computed decision and window end date |
| Metrics log lines were missing under `uvicorn` | Tests passed, but the running server printed only access logs | Configured a dedicated metrics logger [CONFIRM: fix committed] |
| The LLM router sometimes failed JSON validation | Per-call metrics with a failure category showed HTTP 400 `json_validate_failed` on 7 of 37 routed requests in two runs; the keyword fallback absorbed them | The router's `max_tokens=100` was too low for `gpt-oss-20b` JSON mode; raising it to 512 reduced the failures to 0 of 37 in run 4 |
| Blank LLM answers were accepted as successful and cached | The reasoning model sometimes reached its token limit without producing an answer | Treat a blank answer as a failure and retry once with a larger token limit |
| The answer model doubled an order total | The total appeared on every joined item row, so the model added it more than once | Present order-level fields once, separately from the item rows |
| Some combined cases were slow in an early run | The delays were consistent with Groq rate limiting | Pace evaluation requests 5 seconds apart; the paced run had no retries |
| Seeded dates drift with the day the seed runs, which would make fixed expected answers go stale | Planning the evaluation set | Expected values are resolved from the database and today's date at evaluation time, using orders well away from the 30-day boundary |

## 7. Testing

The suite has **224 tests** and uses fakes for the LLM and embedder, so it needs no network and no API key.

```bash
pytest -q
```

| Area | What is tested | File |
|---|---|---|
| Chunking | Deterministic output, no empty chunks, headings carried into each chunk | `tests/test_chunking.py` |
| Vector store | Add, count, version lookup, deleting other versions, nearest-first query | `tests/test_store.py` |
| Re-ingestion | First ingest, unchanged re-upload (no re-embed), changed upload replaces with no stale chunks, embedder failure keeps the old version, documents are independent | `tests/test_ingest.py` |
| Documents API and jobs | Upload validation (type, size, UTF-8, empty), job lifecycle, failed jobs, replace flow | `tests/test_documents_api.py`, `tests/test_job_store.py` |
| Access control | Own orders only, other customer's order looks like a missing one, injected `customer_id`/SQL arguments rejected, SQL guard, write attempt fails on the read-only connection | `tests/test_access_control.py` |
| Routing rules | Order ID extraction; blocked categories; normal questions not blocked; mixed benign and malicious text is blocked | `tests/test_rules.py` |
| Router | Each route, invalid JSON, unknown route and LLM errors fall back, blocked questions never call the LLM | `tests/test_router.py` |
| LLM client | Retries, backoff, `Retry-After`, no retry on 401, timeouts, malformed responses, key never logged, cost calculation | `tests/test_llm_client.py` |
| Data evidence | Tool planning, return-window boundary (day 30 inside, day 31 outside), final-sale items, warranty end dates | `tests/test_data_evidence.py` |
| Policy and grounding | Distance threshold, empty store, every route and evidence combination | `tests/test_policy_grounding.py` |
| `/ask` endpoint | All route types, blocked requests, cross-customer isolation, ungrounded refusals, LLM failure fallback, input validation | `tests/test_ask_api.py` |
| Evaluation cases | Schema, unique IDs, category coverage, order IDs exist and belong to the stated customer | `tests/test_eval_cases.py` |

### Evaluation

**Question set.** 42 labelled questions: 8 data, 8 policy, 9 combined, 5 out of scope, 6 unanswerable (topics the policies do not cover) and 6 adversarial (bulk email request, prompt injection, `DROP TABLE`, another customer's order, SQL in the order ID, "list all orders"). Expected values are not hardcoded: order statuses, totals, counts and return decisions are resolved from the database and today's date when the evaluation runs.

**Systems compared.**
- **Hybrid:** the full `/ask` pipeline.
- **LLM-only baseline:** the same answer model with a generic support prompt, but **no retrieval, SQL tools, router or customer ID**. It makes exactly one model call per question.

**How each answer was judged.** Rule-based automatic judge, with no model judge:
- order status, total and item names must appear in the answer
- return decisions are resolved from the database facts and matched against yes/no phrasing, plus the reason keyword for "no" cases
- policy numbers must appear
- refusal cases must contain a refusal phrase from a fixed list (the app's own refusal messages plus generic phrases)
- must-not-contain checks for another customer's name or email domain
- an answer is **unsupported** if it states a fact that neither the SQL result nor the retrieved chunks contain, or answers a question that should have been refused

**Results.** The full 42-case table is the headline result.

| Category | Hybrid passes | Baseline passes | Hybrid unsupported | Baseline unsupported |
|---|---:|---:|---:|---:|
| Overall | 39/42 | 23/42 | 1/42 | 15/42 |
| Data | 8/8 | 2/8 | 0/8 | 2/8 |
| Policy | 8/8 | 6/8 | 0/8 | 3/8 |
| Combined | 7/9 | 7/9 | 0/9 | 3/9 |
| Out of scope | 5/5 | 2/5 | 0/5 | 2/5 |
| Unanswerable | 5/6 | 2/6 | 1/6 | 5/6 |
| Adversarial | 6/6 | 4/6 | 0/6 | 0/6 |

**Adjusted view (excludes U05 and C07 for both systems).**

| System | Passes | Unsupported |
|---|---:|---:|
| Hybrid | 39/40 | 0/40 |
| LLM-only baseline | 21/40 | 14/40 |

| System | p50 latency (ms) | p95 latency (ms) | Mean cost per case (USD) | Degraded count |
|---|---:|---:|---:|---:|
| Hybrid | 2185.5 | 2752.85 | 0.0002251035714285714 | 0 |
| LLM-only baseline | 1565.0 | 1906.35 | 0.00018287857142857143 | 0 |

**Development runs (overall passes only).**

| Run | Judged result | Change | Hybrid passes | Baseline passes |
|---:|---|---|---:|---:|
| 1 | `judged_20261004T094054_848382Z.json` | As first built | 37/42 | 23/42 |
| 2 | `judged_20261004T094058_331417Z.json` | After fixes for blank LLM output, doubled order total, fallback router default, and evaluation warm-up | 39/42 | 23/42 |
| 3 | `judged_20261004T100412_837014Z.json` | Same product code, with 5 s pacing | 39/42 | 22/42 |
| 4 | `judged_20261004T103126_190493Z.json` | After raising router `max_tokens` to 512 | 39/42 | 23/42 |

**Reading the numbers honestly.** This is a rule-based automatic judge. With 42 cases, one case is about 2.4 percentage points. Results vary by one or two cases between runs. Run 4 is not a held-out test because fixes followed earlier runs.

**How to read these numbers.**

- The automatic judge matches strings, so it makes mistakes in both directions.
- The hybrid system has 1/42 automatic unsupported-claim flags: U05, a grounded answer that matches `exchange_policy.md` but belongs to a mislabeled case. The baseline has 15/42; D01, C06 and O04 are flagged only because the ordinary verb "placed" is mistaken for an order status, leaving about 12/42. The other baseline flags were not checked one by one.
- Some baseline passes are not earned. D01 repeats "delivered" while asking for the customer's name, and C07 mentions 12 months without giving the requested end date.
- The baseline has no database or policy access, so misses on data questions are expected. More telling are invented details: order totals in D08, a 14-day price-match window and email address in U02, loyalty-point rates in U03, and a support email address in P06 and U05.
- Manual override: hybrid C04 is an automatic failure in run 4, but its answer gives the correct decision and reason.

## 8. Known Limitations

- **No authentication.** `/ask` trusts the `customer_id` in the request body. A real deployment must derive it from a verified session or token; the data layer already treats it as server-supplied.
- **Small evaluation set** written by one person, judged by rules that I wrote; it shows direction, not statistical proof.
- **The judge matches fixed phrases.** An answer can pass or fail because of wording. Manual review shows that C04 in run 4 is an automatic failure even though the answer gives the correct decision and reason, because "isn't returnable" is not in the phrase list.
- **Two evaluation cases have defects.** Reading the failures showed that U05 is labeled unanswerable even though `exchange_policy.md` covers it, while C07 requires the answer to repeat "12" even though the question asks for the end date. The adjusted view excludes both cases, but the full 42-case results remain the headline.
- **Answers vary between runs.** The unsupported-claim check only catches concrete tokens, and the system sometimes reasons from what a policy does not list.
- **Keyword-based tool planning.** Unusual phrasings can pick the wrong query tool.
- **English-only rules** in the guard and fallback router.
- **Background jobs run in-process.** A restart during ingestion loses the running task (the job record would stay `running`).
- **SQLite and a local vector store** do not scale horizontally.
- **Cost figures are estimates** from a configurable price table, not billing data.
- **One relevance threshold** for all policy questions.
- **Free-tier rate limits** apply to the Groq API.

## 9. Future Scope

1. **Authentication:** derive `customer_id` from a signed token so the identity never comes from the request body.
2. **Larger, independently labelled evaluation set,** with a second reviewer, to get meaningful error bars.
3. **LLM-assisted query planner** that chooses among the same allowlisted tools, replacing keyword planning without weakening safety.
4. **Real job queue** (for example RQ or Celery) with retries and recovery of interrupted jobs.
5. **Hybrid keyword + vector retrieval with reranking** for better recall on short or exact-term queries.
6. **Caching and rate limiting** per customer to control cost and abuse.
7. **Conversation memory** for follow-up questions such as "what about that order?".

## AI Tool Disclosure and Assumptions

**AI tools used** [CONFIRM: edit to match what actually happened]
- Claude was used for planning, design discussion and writing step-by-step prompts.
- A coding agent [CONFIRM: tool and model name] generated most of the code from those prompts, in small commits.
- AI assistance was also used for test writing and README drafting.
- I ran the test suite and the live demo myself and reviewed the results after each step. I can walk through any module, the evaluation method, and make a small change live.

**Assumptions**
- Kartly data is fictional; seeded dates are relative to the day the seed script runs.
- The return window is 30 days from the **delivery date**, counted inclusively: an order delivered on Sep 1 is returnable through Oct 1.
- Policy documents were written by me for this project.
- Warranty end dates are delivery date plus the product's `warranty_months`.
- GPT-OSS models were chosen because they are Groq's current production models. The price table in `.env.example` uses the 120B rates for every call, which slightly overestimates the router's cost.
- Customer identity is taken from the request, since the assignment defines `/ask` as taking a customer ID.

## Learnings

- **Enforce safety in code, not in prompts.** The strongest protections (tool allowlist, server-supplied customer ID, read-only connection) work even if the model is fooled, and they are easy to test.
- **Measure before choosing a threshold.** The gap between covered (about 0.2-0.5) and uncovered (about 0.7) distances came from running real queries, not from a guess.
- **Design the failure path first.** The bad-key demo returns a useful answer with `degraded: true` because the fallback was built and tested alongside the happy path.
- **Let code do the arithmetic.** Computing return windows and warranty dates in code removed a whole class of model errors.
- **Separate running from judging in an evaluation.** Saving raw outputs lets me fix a judging rule and re-score without spending more LLM calls.
- **Record failures for each model call.** The router's failure category exposed JSON validation errors that the keyword fallback had hidden.
- **Validate an answer before caching it.** A successful HTTP response can still contain a blank answer, which should be retried instead of cached.
- **Do not repeat shared facts in model input.** Showing an order total once prevents the model from adding the same value for every item.
- **Pace live evaluations.** Spacing requests made it easier to separate product behavior from rate-limit delays.

## Run so far

Copy the environment file, install dependencies, seed the database, and start the API:

```bash
cp .env.example .env
pip install -r requirements.txt
python -m scripts.seed
uvicorn app.main:app --reload
```

Open `http://localhost:8000/health` to check the service.

Alternatively, run `docker compose up --build` after creating `.env`.


**Demo video:** [ADD LINK]
