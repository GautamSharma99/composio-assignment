"""Inline results.json + accuracy_report.json into site/index.html (PRD §12, §16.3).

Data is inlined (not fetched) so the page is one self-contained, reviewer-proof file
that also deploys as a static Vercel asset with no build step. Re-run after any agent
or verify run to refresh the page.

Usage::

    python -m agent.build_site
"""

from __future__ import annotations

import json
import re

from .config import ACCURACY_JSON, RESULTS_JSON, SITE_DIR

INDEX = SITE_DIR / "index.html"
_MARKER = re.compile(
    r'(<script id="app-data" type="application/json">)(.*?)(</script>)', re.S
)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run() -> None:
    if not INDEX.exists():
        raise SystemExit(f"{INDEX} not found.")
    results = _load(RESULTS_JSON)
    accuracy = _load(ACCURACY_JSON)

    payload = json.dumps({"results": results, "accuracy": accuracy}, ensure_ascii=False)
    # Neutralize any '<' (e.g. '</script>' inside an evidence quote) so it can't break
    # the HTML parser. '<' round-trips cleanly through JSON.parse.
    safe = payload.replace("<", "\\u003c")

    html = INDEX.read_text(encoding="utf-8")
    if not _MARKER.search(html):
        raise SystemExit("data marker <script id='app-data'> not found in index.html")
    html = _MARKER.sub(lambda m: m.group(1) + "\n" + safe + "\n" + m.group(3), html)
    INDEX.write_text(html, encoding="utf-8")

    n = len(results["apps"]) if results and results.get("apps") else 0
    delta = None
    if accuracy and accuracy.get("overall"):
        delta = accuracy["overall"].get("delta")
    print(f"Injected into {INDEX}: {n} apps"
          + (f", accuracy delta = {delta}" if delta is not None else ", no accuracy report yet"))


if __name__ == "__main__":
    run()
