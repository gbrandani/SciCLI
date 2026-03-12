"""
test_together.py — Smoke tests for the Together AI provider.

Tests (in order):
  1. Raw API reachability — plain chat, no tools
  2. Tool calling with a minimal schema — does gpt-oss-20b support it?
  3. SciCLI TogetherProvider.send() with search=off (reread+get_paper_references only)
  4. SciCLI TogetherProvider.send() with search=on (full agentic pipeline, 1 iteration)

Run:
  python tests/test_together.py              # all tests
  python tests/test_together.py --test 1     # only test 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from repo root or tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from config import SessionState

PROVIDER   = "togetherai"
MODEL_SHORT = "gpt-oss-20b"
MODEL_FULL  = config.get_model_full_id(PROVIDER, MODEL_SHORT)
BASE_URL    = "https://api.together.xyz/v1"


def _api_key() -> str:
    key = os.environ.get("TOGETHER_API_KEY", "").strip()
    if not key:
        # Try loading from settings.json
        key = config.get_api_key(PROVIDER)
    if not key:
        sys.exit("TOGETHER_API_KEY not set — cannot run tests.")
    return key


def _client():
    from openai import OpenAI
    return OpenAI(api_key=_api_key(), base_url=BASE_URL)


def _state(search_mode: str = "off") -> SessionState:
    s = SessionState(provider=PROVIDER, model=MODEL_SHORT)
    object.__setattr__(s, "reread_registry", {})
    object.__setattr__(s, "search_contexts", {})
    object.__setattr__(s, "search_snap_counter", 0)
    object.__setattr__(s, "search_snap_registry", [])
    s.search_mode = search_mode
    s.search_depth = "shallow"
    s.research_pipeline = True
    s.target_papers = 0
    s.target_searches = 0
    s.force_answer_at = 0
    s.max_iterations = 0
    s.use_think_phase = -1
    return s


# ── Test 1: plain chat, no tools ──────────────────────────────────────────────

def test_1_plain_chat():
    print("\n=== Test 1: plain chat (no tools) ===")
    client = _client()
    resp = client.chat.completions.create(
        model=MODEL_FULL,
        messages=[{"role": "user", "content": "What is 1+1? Answer with just the number."}],
        max_tokens=20,
    )
    answer = resp.choices[0].message.content or ""
    finish = resp.choices[0].finish_reason
    print(f"  Model: {MODEL_FULL}")
    print(f"  finish_reason: {finish}")
    print(f"  Response: {answer!r}")
    # Model may return empty on very short max_tokens; just confirm no exception and finish_reason
    assert finish in ("stop", "length", "eos"), f"Unexpected finish_reason: {finish}"
    print("  PASS")


# ── Test 2: single tool calling ───────────────────────────────────────────────

def test_2_tool_calling():
    print("\n=== Test 2: tool calling (minimal schema) ===")
    client = _client()
    tool_schema = [{
        "type": "function",
        "function": {
            "name": "get_answer",
            "description": "Return the answer to a simple question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The answer"}
                },
                "required": ["answer"],
            },
        },
    }]
    resp = client.chat.completions.create(
        model=MODEL_FULL,
        messages=[{"role": "user", "content": "What is 2+2? Use the get_answer tool."}],
        tools=tool_schema,
        tool_choice="auto",
        max_tokens=100,
    )
    choice = resp.choices[0]
    msg = choice.message
    has_tool_call = bool(getattr(msg, "tool_calls", None))
    print(f"  finish_reason: {choice.finish_reason}")
    print(f"  tool_calls present: {has_tool_call}")
    if has_tool_call:
        tc = msg.tool_calls[0]
        print(f"  tool called: {tc.function.name}({tc.function.arguments})")
    else:
        print(f"  content: {msg.content!r}")
    print("  PASS (no exception)")


# ── Test 3: SciCLI send() with search=off ────────────────────────────────────

def test_3_scicli_search_off():
    print("\n=== Test 3: SciCLI TogetherProvider — search=off ===")
    from providers import TogetherProvider
    prov = TogetherProvider(api_key=_api_key())
    state = _state(search_mode="off")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content":
            "In one sentence: what does the ATPase domain of a chromatin remodeler do?"},
    ]
    bundle = prov.send(
        messages=messages,
        model=MODEL_SHORT,
        state=state,
        max_output_tokens=200,
        use_tools=True,
    )
    print(f"  input_tokens:  {bundle.input_tokens}")
    print(f"  output_tokens: {bundle.output_tokens}")
    print(f"  text (first 200 chars): {bundle.text[:200]!r}")
    assert bundle.text.strip(), "Empty reply"
    print("  PASS")


# ── Test 4: SciCLI send() with search=auto (agentic, shallow) ────────────────

def test_4_scicli_search_auto():
    print("\n=== Test 4: SciCLI TogetherProvider — search=auto (agentic shallow) ===")
    from providers import TogetherProvider
    prov = TogetherProvider(api_key=_api_key())
    state = _state(search_mode="auto")
    state.search_depth = "shallow"
    messages = [
        {"role": "system", "content": "You are a helpful scientific assistant."},
        {"role": "user", "content":
            "What is the current consensus on CTCF loop extrusion in mammals? "
            "Answer in 2-3 sentences."},
    ]

    tool_calls_seen = []

    def on_tool_start(name, args):
        tool_calls_seen.append(name)
        print(f"  [tool] {name}({list(args.keys())})")

    def on_status(msg):
        print(f"  [status] {msg}")

    bundle = prov.send(
        messages=messages,
        model=MODEL_SHORT,
        state=state,
        max_output_tokens=800,
        use_tools=True,
        on_tool_start=on_tool_start,
        on_status=on_status,
    )
    print(f"  tool calls made: {tool_calls_seen}")
    print(f"  input_tokens:  {bundle.input_tokens}")
    print(f"  output_tokens: {bundle.output_tokens}")
    print(f"  text (first 300 chars): {bundle.text[:300]!r}")
    assert bundle.text.strip(), "Empty reply"
    print("  PASS")


# ── Runner ────────────────────────────────────────────────────────────────────

TESTS = {
    "1": test_1_plain_chat,
    "2": test_2_tool_calling,
    "3": test_3_scicli_search_off,
    "4": test_4_scicli_search_auto,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=list(TESTS.keys()), help="Run only this test number")
    args = parser.parse_args()

    to_run = [TESTS[args.test]] if args.test else list(TESTS.values())
    failed = []
    for fn in to_run:
        try:
            fn()
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback; traceback.print_exc()
            failed.append(fn.__name__)

    print(f"\n{'='*50}")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print(f"All {len(to_run)} test(s) passed.")
