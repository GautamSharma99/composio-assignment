# 100-App Toolkit-Readiness Research Agent

An agent that researches 100 apps for how build-ready they are as agent tools (auth,
self-serve vs gated, API surface, MCP-readiness, buildability), then **verifies its own
accuracy with a measured Pass 1 → Pass 2 delta** and ships one self-contained HTML page.

- **Live page:** _<add your Vercel URL>_
- **Result:** decision accuracy **67% → 78% (+11 pts)** on a hand-labeled 20-app sample; all
  100 apps researched, every field backed by an evidence URL.

Stack: Python · OpenAI (structured outputs) · Composio SDK+MCP · Vercel.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # add OPENAI_API_KEY and COMPOSIO_API_KEY
python scripts/check_env.py # verify keys + backends

python -m agent.orchestrator --pass 1 --scope sample   # naive baseline
python -m agent.orchestrator --pass 2 --scope all      # verified pass (loops L1–L4), all 100
python -m agent.verify                                 # score the Pass 1 → Pass 2 delta
python -m agent.build_site                             # inline results into site/index.html

open site/index.html        # view the page
```

Use `--scope sample` (20 apps) for a fast/cheap run. Runs are resume-safe (checkpointed);
add `--fresh` to recompute. Set `USE_FALLBACK=true` to run with no Composio (free
DuckDuckGo + httpx backend). Data is pre-generated, so `open site/index.html` works as-is.

### Live proof — OpenAI agent calling a real Composio tool
```bash
python -m agent.proof_demo --mode search              # zero setup
python -m agent.proof_demo --mode app --app github    # prints a Composio OAuth link, then calls GitHub
```

## How it works

Per app: **discover** (targeted search) → **retrieve** (fetch + scrape, cached) →
**extract** (OpenAI structured output, a quote + URL per field) → **verify** (L1 evidence
grounding · L2 blind cross-check · L3 MCP-existence probe · L4 confidence gate → human queue).
Backend is swappable — Composio primary, free pipeline fallback on quota.

## Layout

```
apps.json                the 100 apps            data/results.json        100 structured rows
agent/                   the pipeline            data/ground_truth.json   20-app hand-labeled sample
  research.py            discover/retrieve/extract   data/accuracy_report.json  Pass 1 vs Pass 2
  verify.py              scoring vs ground truth
  proof_demo.py          OpenAI → Composio tool  site/index.html          the deliverable page
```

## Deploy

```bash
python -m agent.build_site && npx vercel deploy site --prod
```
