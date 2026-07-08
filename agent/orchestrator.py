"""Async orchestrator: concurrency-capped, resume-safe, checkpoint-per-app (PRD §7).

CLI::

    python -m agent.orchestrator --pass 1 --scope all       # naive baseline, all 100
    python -m agent.orchestrator --pass 2 --scope sample    # loops, on the 20-app sample
    python -m agent.orchestrator --pass 2 --scope all       # loops, all 100
    python -m agent.orchestrator --assemble                 # (re)write data/results.json

Re-runs resume from checkpoints; pass ``--fresh`` to recompute. Retry/backoff lives in
``llm.py`` (OpenAI) and ``research.py`` (backend auto-fallback); this layer owns
concurrency, checkpointing and progress.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from .cache import Cache, utcnow
from .config import APPS_JSON, RESULTS_JSON, SETTINGS
from .llm import LLM
from .research import ResearchAgent, Retriever
from .sample import SAMPLE_SET
from .schema import AppSeed

log = logging.getLogger("agent.orchestrator")


def load_seeds() -> list[AppSeed]:
    data = json.loads(APPS_JSON.read_text(encoding="utf-8"))
    return [AppSeed(**row) for row in data]


def _select(seeds: list[AppSeed], scope: str, limit: int | None) -> list[AppSeed]:
    if scope == "sample":
        seeds = [s for s in seeds if s.id in SAMPLE_SET]
    if limit:
        seeds = seeds[:limit]
    return seeds


async def run_pass(pass_label: str, seeds: list[AppSeed], agent: ResearchAgent,
                   cache: Cache, *, concurrency: int, fresh: bool) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    total = len(seeds)
    done = 0

    async def worker(seed: AppSeed) -> dict:
        nonlocal done
        if not fresh and cache.has_checkpoint(pass_label, seed.id):
            done += 1
            log.info("[%s] %3d/%d  #%-3d %-24s (cached)", pass_label, done, total, seed.id, seed.name)
            return cache.get_checkpoint(pass_label, seed.id)

        async with sem:
            fn = agent.run_pass1 if pass_label == "pass1" else agent.run_pass2
            result = await fn(seed)

        row = result.model_dump(mode="json")
        cache.put_checkpoint(pass_label, seed.id, row)
        done += 1
        flags = ""
        if row.get("needs_human_review"):
            flags += " [HUMAN]"
        if row.get("error"):
            flags += " [ERROR]"
        log.info("[%s] %3d/%d  #%-3d %-24s -> %s / %s%s", pass_label, done, total, seed.id,
                 seed.name, row.get("buildability"), row.get("self_serve"), flags)
        return row

    return await asyncio.gather(*[worker(s) for s in seeds])


def assemble_results(cache: Cache, seeds: list[AppSeed], backend_mode: str = "") -> dict:
    """Merge checkpoints into data/results.json — prefer pass2 (verified) over pass1."""
    apps: list[dict] = []
    n_pass2 = n_pass1 = n_missing = 0
    for seed in seeds:
        row = cache.get_checkpoint("pass2", seed.id)
        if row is not None:
            n_pass2 += 1
        else:
            row = cache.get_checkpoint("pass1", seed.id)
            if row is not None:
                n_pass1 += 1
        if row is None:
            n_missing += 1
            continue
        apps.append(row)

    payload = {
        "meta": {
            "generated_at": utcnow(),
            "count": len(apps),
            "from_pass2": n_pass2,
            "from_pass1_only": n_pass1,
            "missing": n_missing,
            "backend_mode": backend_mode,
        },
        "apps": apps,
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote %s (%d apps: %d pass2, %d pass1-only, %d missing)",
             RESULTS_JSON, len(apps), n_pass2, n_pass1, n_missing)
    return payload


def build_agent(prefer: str | None = None) -> tuple[Cache, Retriever, ResearchAgent]:
    cache = Cache()
    retriever = Retriever(cache, prefer=prefer or SETTINGS.resolve_backend())
    agent = ResearchAgent(retriever, LLM())
    return cache, retriever, agent


async def _amain(args: argparse.Namespace) -> None:
    seeds = load_seeds()

    if args.assemble and args.pass_num is None:
        assemble_results(Cache(), seeds)
        return

    if not SETTINGS.has_openai:
        raise SystemExit("OPENAI_API_KEY is not set in .env — extraction needs it. Fill it in and re-run.")

    cache, retriever, agent = build_agent(prefer=args.backend)
    selected = _select(seeds, args.scope, args.limit)
    pass_label = f"pass{args.pass_num}"
    log.info("Running %s on %d apps (scope=%s, backend=%s, concurrency=%d)",
             pass_label, len(selected), args.scope, retriever.mode, args.concurrency)

    await run_pass(pass_label, selected, agent, cache,
                   concurrency=args.concurrency, fresh=args.fresh)

    retriever.close()
    if retriever.degraded:
        log.warning("NOTE: backend degraded to fallback during this run (Composio ceiling hit).")

    # Always refresh results.json so the site reflects the newest checkpoints.
    assemble_results(cache, seeds, backend_mode=retriever.mode)


def _quiet_noisy_loggers() -> None:
    for noisy in ("httpx", "httpcore", "composio", "openai", "urllib3", "composio_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _quiet_noisy_loggers()
    p = argparse.ArgumentParser(description="100-app toolkit-readiness research agent")
    p.add_argument("--pass", dest="pass_num", type=int, choices=(1, 2), default=None,
                   help="which pass to run (1=naive baseline, 2=verified with loops)")
    p.add_argument("--scope", choices=("all", "sample"), default="all")
    p.add_argument("--limit", type=int, default=None, help="cap number of apps (testing)")
    p.add_argument("--fresh", action="store_true", help="ignore checkpoints and recompute")
    p.add_argument("--backend", choices=("composio", "fallback"), default=None,
                   help="force a backend (default: composio unless USE_FALLBACK / no key)")
    p.add_argument("--concurrency", type=int, default=SETTINGS.concurrency)
    p.add_argument("--assemble", action="store_true", help="write results.json from checkpoints")
    args = p.parse_args()

    if args.pass_num is None and not args.assemble:
        p.error("specify --pass {1,2} and/or --assemble")

    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
