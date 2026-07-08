# 100-App Toolkit-Readiness Research Agent

An agent that researches **100 named apps** for how build-ready they are as agent tools —
auth, self-serve vs gated, API surface, MCP-readiness, buildability — finds the patterns,
**verifies its own accuracy with a measured Pass 1 → Pass 2 delta**, and ships one
self-explanatory HTML page.

Built for the Composio AI Product Ops take-home. Stack: **Python · OpenAI (structured
outputs) · Composio SDK+MCP · Vercel**.

- **The page** (`site/index.html`) — headline-first: the patterns, the 100-row matrix,
  the agent + where a human stepped in, the live proof, and the verification delta.
- **The agent** (`agent/`) — async, checkpointed, resume-safe, swappable backend
  (Composio primary, free fallback) and four verification loops.
- **The proof** (`agent/proof_demo.py`) — an OpenAI agent that calls a real Composio tool
  end-to-end.

---

## Quickstart

```bash
# 1. Install (Python 3.11–3.14)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env       # then paste your keys into .env
#   OPENAI_API_KEY   — required (extraction, cross-check, proof demo)
#   COMPOSIO_API_KEY — primary search/scrape backend + proof demo
#   USE_FALLBACK     — true to force the free (no-Composio) backend

# 3. Preflight (checks keys + smoke-tests both backends)
python scripts/check_env.py

# 4. Run the agent
python -m agent.orchestrator --pass 1 --scope all      # naive baseline, all 100
python -m agent.orchestrator --pass 2 --scope all      # verified (loops L1–L4), all 100

# 5. Score the accuracy delta on the hand-labeled sample
python -m agent.verify                                  # writes data/accuracy_report.json

# 6. Build the page (inlines results + accuracy into site/index.html)
python -m agent.build_site

# 7. The live proof
python -m agent.proof_demo --mode app --app github      # OpenAI agent → real GitHub toolkit
python -m agent.proof_demo --mode search                # zero-setup: agent → Composio web search
```

> **Faster / cheaper:** swap `--scope all` for `--scope sample` in step 4 to run only the
> 20-app verification sample — enough to produce the accuracy delta and a populated page.

Runs are **resume-safe**: re-running continues from `data/checkpoints/`. Pass `--fresh` to
recompute. Raw pages are cached under `data/cache/`, so re-extraction and the Pass 2 re-run
are nearly free and quota-safe.

---

## How it works

```
apps.json (100)
      │
      ▼
┌──────────────── Orchestrator ────────────────┐
│ async · concurrency-capped · retry+backoff    │
│ checkpoint per app · resume-safe               │
└───────────────┬───────────────────────────────┘
                │  per app
                ▼
   A. DISCOVER  targeted queries → docs/auth/pricing/API/MCP  ── search backend
   B. RETRIEVE  fetch + scrape those pages (cached)            ── scrape backend
   C. EXTRACT   OpenAI structured output → strict schema, a quote + URL per field
   D. PERSIST   raw pages + extracted row → data/cache, results.json

Backends (interchangeable, PRD §7):
  • Composio SDK   → COMPOSIO_SEARCH_WEB + COMPOSIO_SEARCH_FETCH_URL_CONTENT  (no-auth, API key only)
  • Plain pipeline → DuckDuckGo search + httpx fetch + BeautifulSoup           (no key needed)
  Auto-fallback: the moment Composio raises a quota/rate limit, the run degrades to the
  free pipeline and finishes — the ceiling itself becomes page content.
```

### Pass 1 vs Pass 2 (why the delta is real, not asserted)

| | Pass 1 — naive baseline | Pass 2 — verified |
|---|---|---|
| Discover | 1 generic search | targeted docs/auth/pricing queries |
| Retrieve | 1 page | up to 6 prioritized pages |
| Extract | ungrounded, "best guess" | **L1** grounded — quote per field or answer *unknown* |
| Checks | none | **L2** blind cross-check · **L3** MCP-existence probe · **L4** confidence gate → human queue |

Both passes are scored against the same hand-labeled **stratified sample**
(`data/ground_truth.json`, 20 apps: ≥1 per category + every gated app + every disambiguation
trap). `agent/verify.py` writes the per-field Pass 1 → Pass 2 delta and lists every remaining
miss honestly, tagged with its pre-registered failure mode (H1–H4).

### Where a human steps in (only where required, PRD §8)
1. **Ground-truth labeling** of the 20-app sample — this *is* the accuracy measurement.
2. **Low-confidence / cross-check-disagreement rows** — routed to a review queue
   (`needs_human_review` + `human_reason`).
3. **Gated / login-walled apps** (DealCloud, PitchBook, Paygent, iPayX, fanbasis,
   Waterfall.io) — a human confirms "gated + evidence", the correct finding.

---

## Repo layout

```
apps.json                    the 100 apps (name, category, hint_url, disambiguation)
requirements.txt  .env.example  vercel.json
scripts/check_env.py         preflight: keys + backend smoke test
agent/
  schema.py                  pydantic models (strict schema + LLM extraction target)
  config.py                  env/settings
  cache.py                   raw-page cache + per-app checkpoints (atomic writes)
  llm.py                     OpenAI structured extraction + L2 cross-check
  backends/                  base.py · composio_backend.py · fallback_backend.py
  research.py                A/B/C/D pipeline · resilient retriever · Pass 1 / Pass 2 loops
  orchestrator.py            async run + checkpoint/resume + results.json assembly
  verify.py                  scoring vs ground truth → accuracy_report.json
  sample.py                  the 20-app stratified verification sample
  proof_demo.py              OpenAI agent → real Composio tool
  build_site.py              inline results/accuracy into site/index.html
data/
  cache/  checkpoints/  results.json  ground_truth.json  accuracy_report.json
site/index.html              the deliverable (self-contained, inline data)
```

---

## Deploy the page (Vercel)

The page is a single static file — no build step.

```bash
python -m agent.build_site          # make sure data is inlined
npx vercel deploy site --prod       # or drag the site/ folder into the Vercel dashboard
```

`vercel.json` sets `site/` as the output directory and serves `index.html` at `/`.

---

## Notes / honesty

- **No fabricated confidence.** Every row carries an evidence URL, per-field confidence, and
  a verification status (`unverified` / `agent_verified` / `human_verified`) you can audit.
- **Radical honesty is a correct finding.** "Partner-gated, here's the evidence" is scored as
  right; a confident guess on a gated app is the failure mode. During ground-truth labeling
  several PRD assumptions were honestly corrected (fanbasis/Waterfall/DealCloud *do* publish
  public docs; iPayX has no public developer API; OpenAI now requires prepaid billing).
- **The proof is real.** `proof_demo.py` runs an OpenAI agent that actually calls a Composio
  tool. App mode (GitHub) uses a one-time Composio OAuth connect link; search mode needs no
  connection and runs with just the two API keys.
