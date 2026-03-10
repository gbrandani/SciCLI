#!/usr/bin/env python3
"""
test_citation_debug.py — Comprehensive citation pipeline diagnostic.

Runs a full agentic search and traces every step:
  - Think phase output
  - Each tool call (name + args summary)
  - Each tool result (access_level, ref_key, char count)
  - Compaction events
  - Force-answer prompt (full SOURCE INVENTORY)
  - Raw synthesis text (before citation replacement)
  - apply_references key matching analysis
  - Final output with replacements

Usage:
    python test_citation_debug.py                   # deepseek-chat
    python test_citation_debug.py --kimi            # kimi
    python test_citation_debug.py --depth deep      # deep search
    python test_citation_debug.py --question "..."  # custom question
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ──────────────────────────────────────────────────────────────────────────────

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    CYAN    = "\033[36m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    RED     = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE    = "\033[34m"

def hdr(title: str, char: str = "─", width: int = 70) -> str:
    return f"\n{C.BOLD}{C.CYAN}{char * 3} {title} {char * max(0, width - len(title) - 5)}{C.RESET}"

def sub(text: str) -> str:
    return f"  {C.DIM}{text}{C.RESET}"

def ok(text: str) -> str:
    return f"  {C.GREEN}✓ {text}{C.RESET}"

def warn(text: str) -> str:
    return f"  {C.YELLOW}⚠ {text}{C.RESET}"

def err(text: str) -> str:
    return f"  {C.RED}✗ {text}{C.RESET}"

def kv(key: str, val: str) -> str:
    return f"  {C.DIM}{key}:{C.RESET} {val}"


# ──────────────────────────────────────────────────────────────────────────────
# Shared trace log
# ──────────────────────────────────────────────────────────────────────────────

_trace: list[str] = []
_iteration = [0]   # mutable counter


def _log(line: str) -> None:
    _trace.append(line)
    print(line)


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────────────

def on_think_draft(draft: str) -> None:
    _log(hdr("THINK PHASE OUTPUT"))
    for line in draft.splitlines():
        _log(f"  {C.DIM}{line}{C.RESET}")


def on_status(msg: str) -> None:
    _log(f"  {C.MAGENTA}[status]{C.RESET} {msg}")


_tool_call_count = [0]

def on_tool_start(tool_name: str, tool_args: dict) -> None:
    _tool_call_count[0] += 1
    n = _tool_call_count[0]
    _log(hdr(f"TOOL CALL #{n}: {tool_name}", "·"))
    if tool_name == "search_papers":
        q = tool_args.get("query") or " | ".join(tool_args.get("queries", []))
        _log(kv("query", q))
        _log(kv("limit", str(tool_args.get("limit", "default"))))
    elif tool_name == "read_paper":
        _log(kv("paper_id", tool_args.get("paper_id", "?")))
    elif tool_name in ("web_search",):
        _log(kv("query", tool_args.get("query", "?")))
    elif tool_name == "read_webpage":
        _log(kv("url", tool_args.get("url", "?")))
    elif tool_name == "get_paper_references":
        _log(kv("paper_id", tool_args.get("paper_id", "?")))
        _log(kv("direction", tool_args.get("direction", "?")))
    else:
        _log(kv("args", str(tool_args)[:120]))


def on_tool_result(
    tool_name: str, tool_args: dict, result: dict,
    char_count: int = 0, truncated_from: int = 0,
) -> None:
    if tool_name == "search_papers":
        n = result.get("returned", "?") if isinstance(result, dict) else "?"
        total = result.get("total_available", "?") if isinstance(result, dict) else "?"
        _log(ok(f"returned {n} / {total} papers, {char_count:,} chars"
                + (f" (truncated from {truncated_from:,})" if truncated_from else "")))
    elif tool_name == "read_paper":
        if isinstance(result, dict) and result.get("success"):
            al = result.get("access_level", "?")
            title = (result.get("title") or "")[:60]
            _log(ok(f"{al} — \"{title}\" — {char_count:,} chars"
                    + (f" (truncated from {truncated_from:,})" if truncated_from else "")))
        else:
            _log(warn(f"failed: {result.get('error', '?') if isinstance(result, dict) else result}"))
    elif tool_name == "get_paper_references":
        n = result.get("returned", "?") if isinstance(result, dict) else "?"
        _log(ok(f"returned {n} references, {char_count:,} chars"))
    elif tool_name == "read_webpage":
        title = (result.get("title") or "")[:60] if isinstance(result, dict) else ""
        _log(ok(f"\"{title}\" — {char_count:,} chars"))
    elif tool_name == "auto_cite_graph":
        citing = result.get("citing", 0) if isinstance(result, dict) else 0
        referenced = result.get("referenced", 0) if isinstance(result, dict) else 0
        _log(ok(f"auto cite graph: {citing} citing, {referenced} referenced"))
    else:
        status = "ok" if (isinstance(result, dict) and result.get("success")) else "?"
        _log(ok(f"status={status}, {char_count:,} chars"))


# ──────────────────────────────────────────────────────────────────────────────
# Monkey-patching for deep trace
# ──────────────────────────────────────────────────────────────────────────────

def install_patches() -> None:
    """Patch key internal functions to emit trace output."""
    import records as _records_mod
    import agentic as _agentic_mod
    from records import _CITE_PAT, _split_keys, get_ordered_sources

    _orig_apply = _records_mod.apply_references

    def _traced_apply(text, source_details, citation_style="numbered"):
        _log(hdr("apply_references CALLED", "▶"))

        # Analyse keys in text
        found_in_text: list[str] = []
        for m in re.finditer(_CITE_PAT, text):
            for k in _split_keys(m.group(1)):
                if k not in found_in_text:
                    found_in_text.append(k)

        _log(kv("text length", f"{len(text):,} chars"))
        _log(kv("citation_style", citation_style))

        if found_in_text:
            _log(kv("keys in text", str(found_in_text)))
        else:
            _log(warn("NO [BibTeXKey] patterns found in text"))
            _log(sub("(first 300 chars of text):"))
            _log(sub(text[:300].replace("\n", "↵")))

        # Analyse key_map
        ordered = get_ordered_sources(source_details)
        key_map = {s.ref_key: s for s in ordered if s.ref_key}
        _log(kv("records passed in", str(len(source_details))))
        _log(kv("records in key_map", str(len(key_map))))

        if key_map:
            _log(kv("key_map keys", str(sorted(key_map.keys()))))
        else:
            _log(warn("key_map is EMPTY — no keys available for replacement"))
            _log(sub("source_details access_levels: "
                     + str([s.access_level for s in source_details])))

        # Classify matches
        matched   = [k for k in found_in_text if k in key_map]
        unmatched = [k for k in found_in_text if k not in key_map]

        if matched:
            _log(ok(f"keys that WILL be replaced: {matched}"))
        if unmatched:
            _log(warn(f"keys NOT in key_map (won't be replaced): {unmatched}"))
            for k in unmatched:
                # Was it in source_details at all (maybe wrong access_level)?
                for s in source_details:
                    if s.ref_key == k:
                        _log(sub(f"  '{k}' found in records with access_level='{s.access_level}' "
                                 f"tool='{s.tool_name}'"))
                        break
                else:
                    _log(sub(f"  '{k}' NOT FOUND anywhere in source_details"))

        # Call original
        result = _orig_apply(text, source_details, citation_style)

        if result == text and found_in_text:
            _log(err("RESULT UNCHANGED — citation replacement produced no changes"))
        elif result != text:
            count = len(matched)
            _log(ok(f"replacement done — {count} key group(s) replaced"))

        return result

    # Patch both module-level and the name imported into agentic
    _records_mod.apply_references = _traced_apply
    _agentic_mod.apply_references = _traced_apply

    # Patch _build_force_answer_prompt to show full inventory
    _orig_force = _agentic_mod._build_force_answer_prompt

    def _traced_force(source_details, think_draft=None):
        result = _orig_force(source_details, think_draft=think_draft)
        _log(hdr("FORCE-ANSWER PROMPT — SOURCE INVENTORY"))
        # Extract just the inventory section
        inv_start = result.find("SOURCE INVENTORY")
        if inv_start >= 0:
            inv_section = result[inv_start:inv_start + 2000]
            for line in inv_section.splitlines():
                _log(f"  {C.CYAN}{line}{C.RESET}")
            if len(result) - inv_start > 2000:
                _log(sub("  ... (truncated for display)"))
        else:
            _log(warn("SOURCE INVENTORY section not found in prompt"))
        return result

    _agentic_mod._build_force_answer_prompt = _traced_force


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def load_question(path: str) -> str:
    with open(path) as f:
        content = f.read()
    m = re.search(r'<<<User>>>\s*(.*?)(?:<<<Assistant>>>|$)', content, re.DOTALL)
    if m:
        return m.group(1).strip()
    return content.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Citation pipeline diagnostic")
    parser.add_argument("--kimi",     action="store_true", help="Use kimi instead of deepseek")
    parser.add_argument("--depth",    default="shallow",   help="Search depth: shallow|deep")
    parser.add_argument("--question", default=None,        help="Override question")
    args = parser.parse_args()

    # ── Load question ─────────────────────────────────────────────────────────
    dataset = os.path.join(os.path.dirname(__file__), "search_dataset_model_QA1.txt")
    question = args.question or load_question(dataset)

    print(hdr("CITATION PIPELINE DIAGNOSTIC", "═", 70))
    print(kv("question", textwrap.shorten(question, 80)))
    print(kv("provider", "kimi" if args.kimi else "deepseek"))
    print(kv("depth",    args.depth))

    # ── Install patches ───────────────────────────────────────────────────────
    install_patches()

    # ── Set up provider and state ─────────────────────────────────────────────
    from config import SessionState
    from providers import DeepSeekProvider, KimiProvider

    state = SessionState()
    state.search_depth = args.depth
    state.research_pipeline = True
    state.citation_style = "numbered"

    if args.kimi:
        api_key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY") or ""
        if not api_key:
            print(err("KIMI_API_KEY / MOONSHOT_API_KEY not set"))
            sys.exit(1)
        provider = KimiProvider(api_key=api_key)
        model = "moonshot-v1-8k"
    else:
        api_key = os.environ.get("DEEPSEEK_API_KEY") or ""
        if not api_key:
            print(err("DEEPSEEK_API_KEY not set"))
            sys.exit(1)
        provider = DeepSeekProvider(api_key=api_key)
        model = "deepseek-chat"

    messages = [{"role": "user", "content": question}]

    # ── Run pipeline ──────────────────────────────────────────────────────────
    _log(hdr("PIPELINE START", "═", 70))
    t0 = time.time()

    bundle = provider.send(
        messages=messages,
        model=model,
        state=state,
        use_tools=True,
        on_tool_start=on_tool_start,
        on_tool_result=on_tool_result,
        on_status=on_status,
        on_think_draft=on_think_draft,
    )

    elapsed = time.time() - t0

    # ── Final output ──────────────────────────────────────────────────────────
    _log(hdr("FINAL OUTPUT", "═", 70))
    print(bundle.text)

    # ── Summary ───────────────────────────────────────────────────────────────
    _log(hdr("SUMMARY", "═", 70))

    from records import _CITE_PAT, _split_keys
    text = bundle.text

    # Check citations in final output
    remaining_keys = []
    for m in re.finditer(_CITE_PAT, text):
        for k in _split_keys(m.group(1)):
            # If it still looks like a BibTeX key (not a number), it wasn't replaced
            if not k.isdigit() and not re.match(r'^\d+(,\s*\d+)*$', k):
                remaining_keys.append(k)

    # Check for green-coloured citation markers (ANSI) — these were replaced
    replaced_count = len(re.findall(r'\033\[32m\[', text))

    print(kv("elapsed",          f"{elapsed:.1f}s"))
    print(kv("tool calls",       str(_tool_call_count[0])))
    print(kv("records in bundle", str(len(bundle.source_details or []))))
    print(kv("citations replaced", str(replaced_count)))

    if remaining_keys:
        print(warn(f"unreplaced BibTeX keys in output: {remaining_keys}"))
        print(sub("These keys appeared in text but were not in key_map."))
        print(sub("Likely cause: model cited papers it only saw in search results,"))
        print(sub("not papers it actually read (which have full_text/abstract_only access)."))
    else:
        print(ok("No unreplaced BibTeX keys in output"))

    # Check if References section was added
    if "---\nReferences" in text or "---\r\nReferences" in text:
        print(ok("References section present in output"))
    else:
        print(warn("No References section found in output"))
        if replaced_count == 0:
            print(sub("→ apply_references made no replacements (see trace above)"))

    print(kv("source_details access levels",
             str([s.access_level for s in (bundle.source_details or [])])))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
