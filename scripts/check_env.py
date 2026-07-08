"""Preflight: verify keys and smoke-test both backends before a full run.

    python scripts/check_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import SETTINGS  # noqa: E402


def main() -> int:
    print("== Keys ==")
    print(f"  OPENAI_API_KEY   : {'set' if SETTINGS.has_openai else 'MISSING'}")
    print(f"  COMPOSIO_API_KEY : {'set' if SETTINGS.has_composio else 'missing'}")
    print(f"  USE_FALLBACK     : {SETTINGS.use_fallback}")
    print(f"  resolved backend : {SETTINGS.resolve_backend()}")

    ok = True

    if SETTINGS.has_composio and not SETTINGS.use_fallback:
        print("\n== Composio backend smoke test ==")
        try:
            from agent.backends.composio_backend import ComposioBackend

            b = ComposioBackend()
            sr = b.search("GitHub REST API authentication", num=3)
            print(f"  search  -> {len(sr)} results" + (f" (e.g. {sr[0].url})" if sr else ""))
            if sr:
                p = b.fetch(sr[0].url)
                print(f"  fetch   -> ok={p.ok}, {len(p.text)} chars")
        except Exception as e:
            ok = False
            print(f"  Composio FAILED: {type(e).__name__}: {e}")

    print("\n== Fallback backend smoke test ==")
    try:
        from agent.backends.fallback_backend import FallbackBackend

        b = FallbackBackend()
        sr = b.search("Linear GraphQL API", num=2)
        print(f"  search  -> {len(sr)} results")
        if sr:
            p = b.fetch(sr[0].url)
            print(f"  fetch   -> ok={p.ok}, {len(p.text)} chars")
        b.close()
    except Exception as e:
        ok = False
        print(f"  Fallback FAILED: {type(e).__name__}: {e}")

    if SETTINGS.has_openai:
        print("\n== OpenAI smoke test ==")
        try:
            import asyncio

            from agent.backends.base import FetchedPage
            from agent.llm import LLM
            from agent.schema import AppSeed

            seed = AppSeed(id=0, name="GitHub", category="Dev & Infrastructure")
            page = FetchedPage(url="https://docs.github.com/rest", title="GitHub REST",
                               text="GitHub REST API. Authenticate with a personal access token "
                                    "(Bearer). OAuth apps are also supported. Public docs.", ok=True)
            ext = asyncio.run(LLM().extract(seed, [page], grounded=True))
            print(f"  extract -> auth={[a.value for a in ext.auth_methods]}, self_serve={ext.self_serve.value}")
        except Exception as e:
            ok = False
            print(f"  OpenAI FAILED: {type(e).__name__}: {e}")
    else:
        print("\n(OpenAI key missing — extraction/proof demo will not run until it's set.)")

    print("\n" + ("ALL GOOD ✅" if ok else "SOME CHECKS FAILED ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
