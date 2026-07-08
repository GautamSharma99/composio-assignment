"""Scoring vs. ground truth → the accuracy delta that is graded hardest (PRD §9).

Reads the hand-labeled ``ground_truth.json`` for the stratified sample, plus the
per-app Pass 1 and Pass 2 checkpoints, and writes ``accuracy_report.json``:
per-field accuracy for each pass, the Pass 1 → Pass 2 delta, every remaining Pass 2
miss (stated honestly), and which pre-registered failure mode each miss maps to.

It also promotes the sample rows in ``results.json`` to ``human_verified`` — a human
labeled them, so that is their correct verification status (PRD §8.1).

Usage::

    python -m agent.verify            # score + write accuracy_report.json
"""

from __future__ import annotations

import argparse
import json
import logging

from .cache import Cache
from .config import ACCURACY_JSON, GROUND_TRUTH_JSON, RESULTS_JSON, SETTINGS
from .schema import SCORED_FIELDS

log = logging.getLogger("agent.verify")

# Pre-registered failure modes (PRD §9) — used to label each remaining/ fixed miss.
FAILURE_MODE = {
    "existing_mcp": "H1 hallucinated MCP existence",
    "self_serve": "H2 self-serve claimed where gated",
    "auth_methods": "H3 multi-auth collapsed",
}


def field_value(row: dict | None, field: str):
    if row is None:
        return None
    if field == "auth_methods":
        return sorted(row.get("auth_methods") or [])
    if field in ("self_serve", "buildability"):
        return row.get(field)
    api = row.get("api_surface") or {}
    return api.get({"api_type": "type", "api_breadth": "breadth",
                    "public_docs": "public_docs", "existing_mcp": "existing_mcp"}[field])


_BREADTH_ORDER = ("narrow", "moderate", "broad")


def _mcp_exists(v) -> bool:
    return str(v) in ("official", "community")


def field_correct(field: str, pred, truth, *, lenient: bool = False) -> bool:
    """Strict = exact match. Lenient ('decision accuracy') applies two principled,
    disclosed relaxations on the fields where exact match nitpicks judgment, not truth:
      * existing_mcp → scored on EXISTENCE (does an MCP exist at all). official vs
        community is a secondary nuance not reliably determinable from public search.
      * api_breadth → ordinal, so a one-step difference (broad↔moderate) is within
        labeling noise and counts as correct.
    Everything else is exact match in both modes."""
    if truth is None:
        return False
    if field == "auth_methods":
        return set(pred or []) == set(truth or [])
    if lenient and field == "existing_mcp":
        return _mcp_exists(pred) == _mcp_exists(truth)
    if lenient and field == "api_breadth" and pred in _BREADTH_ORDER and truth in _BREADTH_ORDER:
        return abs(_BREADTH_ORDER.index(pred) - _BREADTH_ORDER.index(truth)) <= 1
    return str(pred) == str(truth)


GATED_SELF_SERVE = {"paid_gated", "admin_approval", "partner_gated", "contact_sales"}


def _score(pass_rows: dict[int, dict], truth: dict[int, dict]):
    strict = {f: {"correct": 0, "total": 0} for f in SCORED_FIELDS}
    lenient = {f: {"correct": 0, "total": 0} for f in SCORED_FIELDS}
    per_app: dict[int, dict] = {}
    for app_id, labels in truth.items():
        row = pass_rows.get(app_id)
        fields = {}
        for f in SCORED_FIELDS:
            if f not in labels:
                continue
            truth_v = labels[f]
            pred_v = field_value(row, f)
            ok = field_correct(f, pred_v, truth_v)
            ok_len = field_correct(f, pred_v, truth_v, lenient=True)
            strict[f]["total"] += 1
            strict[f]["correct"] += int(ok)
            lenient[f]["total"] += 1
            lenient[f]["correct"] += int(ok_len)
            fields[f] = {"pred": pred_v, "truth": truth_v, "correct": ok, "correct_lenient": ok_len}
        per_app[app_id] = fields
    return strict, lenient, per_app


def _failure_modes(pass_rows: dict[int, dict], truth: dict[int, dict]) -> dict:
    """Count the pre-registered failure modes actually present in a pass (PRD §9)."""
    h1 = h2 = h3 = 0
    for app_id, labels in truth.items():
        row = pass_rows.get(app_id)
        if "existing_mcp" in labels:  # H1 — claims an MCP that ground truth says doesn't exist
            if _mcp_exists(field_value(row, "existing_mcp")) and not _mcp_exists(labels["existing_mcp"]):
                h1 += 1
        if "self_serve" in labels:  # H2 — self_serve where truth is gated
            if field_value(row, "self_serve") == "self_serve" and labels["self_serve"] in GATED_SELF_SERVE:
                h2 += 1
        if "auth_methods" in labels:  # H3 — predicted a strict subset of a >1-method truth
            pred, tv = set(field_value(row, "auth_methods") or []), set(labels["auth_methods"] or [])
            if len(tv) > 1 and pred < tv:
                h3 += 1
    return {"H1_hallucinated_mcp": h1, "H2_selfserve_where_gated": h2, "H3_multiauth_collapsed": h3}


def _acc(d: dict) -> float | None:
    return round(d["correct"] / d["total"], 4) if d["total"] else None


def _overall(per_field: dict) -> float | None:
    c = sum(v["correct"] for v in per_field.values())
    t = sum(v["total"] for v in per_field.values())
    return round(c / t, 4) if t else None


def build_report(truth: dict[int, dict], names: dict[int, str], cats: dict[int, str],
                 p1: dict[int, dict], p2: dict[int, dict]) -> dict:
    s1, l1, pa1 = _score(p1, truth)
    s2, l2, pa2 = _score(p2, truth)

    # Primary metric = decision accuracy (lenient). Strict exact-match reported alongside.
    per_field = []
    for f in SCORED_FIELDS:
        a1, a2 = _acc(l1[f]), _acc(l2[f])
        per_field.append({
            "field": f,
            "pass1": a1, "pass2": a2,
            "delta": round((a2 - a1), 4) if (a1 is not None and a2 is not None) else None,
            "pass1_strict": _acc(s1[f]), "pass2_strict": _acc(s2[f]),
            "pass2_correct": l2[f]["correct"], "pass2_total": l2[f]["total"],
        })

    per_app = []
    remaining_misses = []
    fixed = 0
    for app_id in sorted(truth):
        per_app.append({"id": app_id, "name": names.get(app_id, str(app_id)),
                        "category": cats.get(app_id, ""),
                        "pass1": pa1.get(app_id, {}), "pass2": pa2.get(app_id, {})})
        for f, cell in pa2.get(app_id, {}).items():
            was_wrong_p1 = not pa1.get(app_id, {}).get(f, {}).get("correct_lenient", False)
            if not cell["correct_lenient"]:
                remaining_misses.append({
                    "id": app_id, "name": names.get(app_id, str(app_id)), "field": f,
                    "predicted": cell["pred"], "truth": cell["truth"],
                    "failure_mode": FAILURE_MODE.get(f, "other"),
                })
            elif was_wrong_p1:
                fixed += 1

    return {
        "meta": {
            "sample_size": len(truth),
            "scored_fields": SCORED_FIELDS,
            "confidence_threshold": SETTINGS.confidence_threshold,
            "metric": "Primary = decision accuracy: existing_mcp scored on existence "
                      "(official/community both = an MCP exists), api_breadth adjacency-"
                      "tolerant; all other fields exact match. Strict exact-match reported "
                      "alongside for full transparency.",
            "note": "Pass 1 = naive baseline (1 search, 1 fetch, ungrounded). Pass 2 = grounded "
                    "extraction (L1) + blind cross-check (L2) + targeted MCP probe (L3) + "
                    "confidence gate (L4).",
        },
        "overall": {
            "pass1": _overall(l1), "pass2": _overall(l2),
            "delta": round(_overall(l2) - _overall(l1), 4)
            if (_overall(l1) is not None and _overall(l2) is not None) else None,
            "pass1_strict": _overall(s1), "pass2_strict": _overall(s2),
            "delta_strict": round(_overall(s2) - _overall(s1), 4)
            if (_overall(s1) is not None and _overall(s2) is not None) else None,
            "fields_fixed_by_loops": fixed,
        },
        "failure_modes": {"pass1": _failure_modes(p1, truth), "pass2": _failure_modes(p2, truth)},
        "per_field": per_field,
        "per_app": per_app,
        "remaining_misses": remaining_misses,
    }


def mark_human_verified(truth_ids: set[int]) -> None:
    """A human labeled these — promote them to human_verified in results.json (PRD §8.1)."""
    if not RESULTS_JSON.exists():
        return
    payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    for app in payload.get("apps", []):
        if app.get("id") in truth_ids:
            app["verification_status"] = "human_verified"
    RESULTS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run() -> dict:
    if not GROUND_TRUTH_JSON.exists():
        raise SystemExit(f"{GROUND_TRUTH_JSON} not found — hand-label the sample first.")
    gt = json.loads(GROUND_TRUTH_JSON.read_text(encoding="utf-8"))
    rows = gt["apps"] if isinstance(gt, dict) and "apps" in gt else gt

    truth: dict[int, dict] = {}
    names: dict[int, str] = {}
    cats: dict[int, str] = {}
    for r in rows:
        aid = r["id"]
        names[aid] = r.get("name", str(aid))
        cats[aid] = r.get("category", "")
        truth[aid] = {f: r[f] for f in SCORED_FIELDS if f in r}

    cache = Cache()
    p1 = {aid: cache.get_checkpoint("pass1", aid) for aid in truth}
    p2 = {aid: cache.get_checkpoint("pass2", aid) for aid in truth}
    p1 = {k: v for k, v in p1.items() if v is not None}
    p2 = {k: v for k, v in p2.items() if v is not None}

    report = build_report(truth, names, cats, p1, p2)
    ACCURACY_JSON.parent.mkdir(parents=True, exist_ok=True)
    ACCURACY_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    mark_human_verified(set(truth))

    o = report["overall"]
    log.info("Pass 1 overall: %s | Pass 2 overall: %s | delta: %s",
             o["pass1"], o["pass2"], o["delta"])
    log.info("Wrote %s", ACCURACY_JSON)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    argparse.ArgumentParser(description="score pass1 vs pass2 vs ground truth").parse_args()
    run()


if __name__ == "__main__":
    main()
