"""The live proof (PRD §11): an OpenAI agent that calls a REAL Composio tool end-to-end.

Two modes:

* ``--mode search`` (zero setup, guaranteed to run with just the two API keys): the
  agent is handed Composio's no-auth ``COMPOSIO_SEARCH_WEB`` tool and answers a live
  question by actually calling it. Proves the OpenAI⇄Composio tool loop with no OAuth.

* ``--mode app`` (default; ``PROOF_APP``, e.g. github): the closed loop the analysis
  points at — an easy-win app from the validated cluster. The agent calls a real
  toolkit tool (a read-only task). If the user hasn't connected the app yet, we mint a
  Composio Connect link, print it, and wait for the OAuth to complete.

Connection uses ``connected_accounts.link`` first (the legacy ``initiate``/``authorize``
OAuth path was retired 2026-07-03), falling back to ``authorize`` for non-OAuth toolkits.

Usage::

    python -m agent.proof_demo                    # app mode, PROOF_APP (github)
    python -m agent.proof_demo --mode search      # no-auth search agent
    python -m agent.proof_demo --mode app --app notion
"""

from __future__ import annotations

import argparse
import json

from .config import SETTINGS

APP_TASKS = {
    "github": "Look up the authenticated GitHub user: their login and name, and the names "
              "of their 3 most recently updated repositories. Then give a one-line summary.",
    "notion": "List up to 3 items (pages or databases) the integration can access in this "
              "Notion workspace, with their titles. Then give a one-line summary.",
    "linear": "List up to 3 of my Linear teams or recent issues with their titles, then "
              "give a one-line summary.",
}
DEFAULT_TASK = "Use the available tool(s) to fetch a small, read-only piece of real data " \
               "from this account and summarize what you found in one line."


def _require_keys() -> None:
    missing = [k for k, v in (("OPENAI_API_KEY", SETTINGS.has_openai),
                              ("COMPOSIO_API_KEY", SETTINGS.has_composio)) if not v]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)} in .env — the proof demo needs both.")


def _clients():
    from composio import Composio
    from composio_openai import OpenAIProvider
    from openai import OpenAI

    composio = Composio(api_key=SETTINGS.composio_api_key, provider=OpenAIProvider())
    openai_client = OpenAI(api_key=SETTINGS.openai_api_key)
    return composio, openai_client


def ensure_connection(composio, toolkit: str, user_id: str, timeout: float = 180.0):
    """Return an ACTIVE connection for (user, toolkit), creating one via OAuth if needed."""
    toolkit = toolkit.upper()
    existing = composio.connected_accounts.list(
        user_ids=[user_id], toolkit_slugs=[toolkit], statuses=["ACTIVE"]
    )
    items = getattr(existing, "items", None) or []
    if items:
        print(f"✓ Using existing connected {toolkit} account for user '{user_id}'.")
        return items[0]

    print(f"No connected {toolkit} account for '{user_id}'. Creating a Composio Connect link…")
    conn_req = _initiate_connection(composio, toolkit, user_id)
    if getattr(conn_req, "redirect_url", None):
        print("\n  👉 Open this URL and authorize, then come back:\n")
        print(f"     {conn_req.redirect_url}\n")
    print(f"Waiting up to {int(timeout)}s for the connection to complete…")
    account = conn_req.wait_for_connection(timeout=timeout)
    print(f"✓ Connected {toolkit}.")
    return account


def _initiate_connection(composio, toolkit: str, user_id: str):
    # link() is the current path (legacy OAuth initiate/authorize retired 2026-07-03).
    try:
        auth_config_id = composio.toolkits._get_auth_config_id(toolkit=toolkit)
        return composio.connected_accounts.link(user_id=user_id, auth_config_id=auth_config_id)
    except Exception as link_err:
        try:
            return composio.toolkits.authorize(user_id=user_id, toolkit=toolkit)
        except Exception as auth_err:
            raise RuntimeError(
                f"Could not start a connection for {toolkit}. "
                f"link() failed: {link_err}; authorize() failed: {auth_err}"
            ) from auth_err


def _agent_loop(composio, openai_client, tools, user_id: str, task: str, max_steps: int = 6) -> str:
    messages = [
        {"role": "system", "content": "You are an agent. Use the provided tools to complete "
                                       "the task with real data, then answer concisely."},
        {"role": "user", "content": task},
    ]
    for step in range(1, max_steps + 1):
        resp = openai_client.chat.completions.create(
            model=SETTINGS.openai_model, tools=tools, messages=messages
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return msg.content or "(no content)"

        called = ", ".join(tc.function.name for tc in msg.tool_calls)
        print(f"  step {step}: model called → {called}")
        results = composio.provider.handle_tool_calls(response=resp, user_id=user_id)
        for tc, result in zip(msg.tool_calls, results):
            content = result if isinstance(result, str) else json.dumps(result, default=str)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content[:6000]})
    return "(reached max steps without a final answer)"


def run_search(question: str) -> None:
    _require_keys()
    composio, openai_client = _clients()
    uid = SETTINGS.composio_user_id
    print(f"\n=== PROOF (search mode) — OpenAI agent calling Composio COMPOSIO_SEARCH_WEB ===")
    print(f"Question: {question}\n")
    tools = composio.tools.get(user_id=uid, tools=["COMPOSIO_SEARCH_WEB"])
    answer = _agent_loop(composio, openai_client, tools, uid, question)
    print("\n--- Agent answer ---")
    print(answer)


def run_app(app: str, task: str | None) -> None:
    _require_keys()
    composio, openai_client = _clients()
    uid = SETTINGS.composio_user_id
    toolkit = app.upper()
    task = task or APP_TASKS.get(app.lower(), DEFAULT_TASK)
    print(f"\n=== PROOF (app mode) — OpenAI agent calling the real Composio {toolkit} toolkit ===")
    print(f"Task: {task}\n")

    ensure_connection(composio, toolkit, uid)
    tools = composio.tools.get(user_id=uid, toolkits=[toolkit])
    print(f"Loaded {len(tools)} {toolkit} tools for the agent.\n")
    answer = _agent_loop(composio, openai_client, tools, uid, task)
    print("\n--- Agent answer ---")
    print(answer)


def main() -> None:
    p = argparse.ArgumentParser(description="Live proof: OpenAI agent calls a real Composio tool.")
    p.add_argument("--mode", choices=("app", "search"), default="app")
    p.add_argument("--app", default=SETTINGS.proof_app, help="toolkit for app mode (default from PROOF_APP)")
    p.add_argument("--task", default=None, help="override the agent task/question")
    p.add_argument("--question", default="What is Composio and what does its Python SDK do? Cite a source.",
                   help="question for search mode")
    args = p.parse_args()

    if args.mode == "search":
        run_search(args.task or args.question)
    else:
        run_app(args.app, args.task)


if __name__ == "__main__":
    main()
