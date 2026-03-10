#!/usr/bin/env python
"""
test_search_quality.py — Comprehensive search quality evaluation for chat_cli.

Tests the 3-phase research pipeline (Think → Search → Synthesize) across:
- 3 providers: openai/gpt-5-mini, deepseek/deepseek-chat, kimi/kimi-k2.5
- 2 depths: shallow, deep
= 6 configurations per question

Outputs are stored in test_outputs/ for manual review.
Automated evaluation checks: citations, source coverage, auto-references, quality.
"""

import subprocess
import sys
import os
import json
import re
import datetime
from pathlib import Path
from collections import Counter

CLI = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scicli.py")
OUTPUT_DIR = Path(os.path.dirname(os.path.dirname(__file__))) / "test_outputs"
TIMEOUT_SHALLOW = 180
TIMEOUT_DEEP = 300


# ----------------------------
# Test questions
# ----------------------------

QUESTIONS = [
    {
        "id": "QA1",
        "question": (
            "experiments report highly variable nucleosome-nucleosome stacking energies, "
            "from 2 to 15 kBT. do simulations offer any explanation for this discrepancy? "
            "search in the literature to answer this question"
        ),
        "key_terms": [
            "simulation", "molecular dynamics", "coarse-grained",
            "histone tail", "ionic", "free energy",
            "stacking", "nucleosome",
        ],
        "expected_papers_keywords": ["Lin", "Moller", "Brouwer", "de Pablo"],
    },
    {
        "id": "QA2",
        "question": (
            "what is the molecular mechanism used by the ATPase domain of chromatin remodellers "
            "to slide nucleosomal DNA Based experiments and simulations from the literature?"
        ),
        "key_terms": [
            "twist defect", "DNA translocation", "ATPase", "cryo-EM",
            "SWI/SNF", "ISWI", "CHD", "remodel",
        ],
        "expected_papers_keywords": ["Brandani", "Yan", "Liu", "Nodelman"],
    },
]


# ----------------------------
# Configuration matrix
# ----------------------------

CONFIGS = [
    {"provider": "openai", "model": "gpt-5-mini", "depth": "shallow", "extra_cmds": []},
    {"provider": "openai", "model": "gpt-5-mini", "depth": "deep", "extra_cmds": []},
    {"provider": "deepseek", "model": "deepseek-chat", "depth": "shallow", "extra_cmds": []},
    {"provider": "deepseek", "model": "deepseek-chat", "depth": "deep", "extra_cmds": []},
    {"provider": "kimi", "model": "kimi-k2.5", "depth": "shallow", "extra_cmds": []},
    {"provider": "kimi", "model": "kimi-k2.5", "depth": "deep", "extra_cmds": []},
    {"provider": "sakura", "model": "gpt-oss-120b", "depth": "shallow", "extra_cmds": []},
    {"provider": "sakura", "model": "gpt-oss-120b", "depth": "deep", "extra_cmds": []},
]


def get_env_key(provider: str) -> str:
    """Check if API key is available for a provider."""
    keys = {
        "openai": ["API_KEY_OPENAI", "OPENAI_API_KEY"],
        "deepseek": ["DEEPSEEK_API_KEY"],
        "kimi": ["KIMI_API_KEY"],
        "sakura": ["SAKURA_API_KEY"],
    }
    for k in keys.get(provider, []):
        if os.environ.get(k):
            return k
    return ""


# ----------------------------
# Runner
# ----------------------------

def run_question(config: dict, question: str) -> dict:
    """Run a question through chat_cli with given config and capture output."""
    commands = []

    # Provider setup
    if config["provider"] != "openai":
        commands.append(f"/provider {config['provider']}")
    if config["model"]:
        commands.append(f"/model {config['model']}")

    # Depth
    commands.append(f"/depth {config['depth']}")

    # Extra commands (e.g., provider-specific settings)
    commands.extend(config.get("extra_cmds", []))

    # Ask question
    commands.append(question)
    commands.append("/quit")
    commands.append("n")

    input_text = "\n".join(commands) + "\n"

    env = os.environ.copy()
    env["TERM"] = "dumb"
    env.pop("MallocStackLogging", None)

    timeout = TIMEOUT_DEEP if config["depth"] == "deep" else TIMEOUT_SHALLOW

    proc = subprocess.Popen(
        [sys.executable, CLI],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    try:
        stdout, stderr = proc.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        return {"stdout": stdout, "stderr": stderr + "\n[TIMEOUT]", "rc": -1, "timeout": True}

    return {"stdout": stdout, "stderr": stderr, "rc": proc.returncode, "timeout": False}


# ----------------------------
# Output extraction
# ----------------------------

def extract_response(stdout: str) -> str:
    """Extract assistant response from full output."""
    # With new UI, there's no "Assistant" header. Response follows the dim rule separator.
    # Look for content after tool calls / dim rules
    lines = stdout.splitlines()
    response_lines = []
    in_response = False

    for line in lines:
        # Skip command echoes and status lines
        if line.strip().startswith(">") and not in_response:
            continue
        if line.strip().startswith("/"):
            continue
        if line.strip().startswith("Chat CLI"):
            continue
        if "Enter to send" in line:
            continue
        if "Switched to" in line or "Search depth:" in line or "Model:" in line:
            continue
        if "OpenAI web search:" in line:
            continue

        # Tool call lines (dim)
        if line.strip().startswith("Searching papers:") or line.strip().startswith("Reading paper"):
            in_response = True
            continue
        if line.strip().startswith("Web search:") or line.strip().startswith("Reading webpage:"):
            in_response = True
            continue
        if line.strip().startswith("Thinking about"):
            in_response = True
            continue
        if line.strip().startswith("Search plan ready"):
            continue
        if line.strip().startswith("→"):
            continue

        # Source section indicators
        if "Papers read (full text)" in line or "Papers consulted" in line:
            break
        if "Web pages read" in line:
            break
        if "search results consulted" in line:
            break
        if "Save conversation" in line:
            break

        if in_response or (not line.strip().startswith(("─", "═"))):
            if line.strip() and not line.strip().startswith("─"):
                in_response = True
                response_lines.append(line)

    return "\n".join(response_lines).strip()


def extract_tool_info(stdout: str) -> dict:
    """Extract tool call information from output."""
    info = {
        "search_papers": 0,
        "read_paper": 0,
        "web_search": 0,
        "read_webpage": 0,
        "think_phase": False,
        "papers_full_text": 0,
        "papers_abstract_only": 0,
    }

    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("Searching papers:"):
            info["search_papers"] += 1
        elif line.startswith("Reading paper"):
            info["read_paper"] += 1
        elif line.startswith("Web search:"):
            info["web_search"] += 1
        elif line.startswith("Reading webpage:"):
            info["read_webpage"] += 1
        elif "Thinking about" in line:
            info["think_phase"] = True
        elif "[full_text]" in line.lower() or "[FULL_TEXT]" in line:
            info["papers_full_text"] += 1
        elif "[abstract_only]" in line.lower() or "[ABSTRACT_ONLY]" in line:
            info["papers_abstract_only"] += 1

    return info


# ----------------------------
# Evaluation
# ----------------------------

def evaluate(qa: dict, config: dict, result: dict) -> dict:
    """Evaluate a response comprehensively."""
    stdout = result["stdout"]
    response = extract_response(stdout)
    tool_info = extract_tool_info(stdout)

    ev = {
        "qa_id": qa["id"],
        "provider": config["provider"],
        "model": config["model"],
        "depth": config["depth"],
        "timeout": result.get("timeout", False),
        "response_length": len(response),
        "tool_info": tool_info,
        "scores": {},
        "issues": [],
    }

    # 1. Inline citations — detect [N], [Author, Journal, Year], and [AuthorYearWord] formats
    num_citations = set(int(m.group(1)) for m in re.finditer(r'\[(\d+)\]', response))
    # [Author, Journal, Year] style citations
    author_citations = set(m.group(0) for m in re.finditer(r'\[[A-Z][a-z]+,\s[^]]+,\s\d{4}[a-z]?\]', response))
    # [AuthorYearWord] BibTeX key style citations
    bibtex_citations = set(m.group(0) for m in re.finditer(r'\[[A-Z][a-z]+\d{4}[a-z]+\w*\]', response))
    all_citation_count = len(num_citations) + len(author_citations) + len(bibtex_citations)
    ev["num_citations"] = all_citation_count
    ev["citation_numbers"] = sorted(num_citations) if num_citations else sorted(author_citations | bibtex_citations)

    if all_citation_count >= 5:
        ev["scores"]["citations"] = 10
    elif all_citation_count >= 3:
        ev["scores"]["citations"] = 7
    elif all_citation_count >= 1:
        ev["scores"]["citations"] = 4
    else:
        ev["scores"]["citations"] = 0
        ev["issues"].append("No inline citations")

    # 2. Auto-references section
    # After Rich rendering, --- becomes ─── and **References** becomes "References" (bold stripped)
    # Check for either raw markdown or rendered form
    # Also detect [Author, Journal, Year] style refs from the cite tool
    has_auto_refs = bool(
        re.search(r'---\s*\n\*?\*?References\*?\*?', response)
        or re.search(r'─+\s*\nReferences', response)
        or (re.search(r'\nReferences\s*\n', response) and re.search(r'\[\d+\]\s+\S.*\.\s', response))
        or (re.search(r'\nReferences\s*\n', response) and re.search(r'\[[A-Z][a-z]+,\s', response))
        or (re.search(r'\nReferences\s*\n', response) and re.search(r'\[[A-Z][a-z]+\d{4}', response))
    )
    ev["has_auto_references"] = has_auto_refs
    if has_auto_refs:
        ev["scores"]["auto_refs"] = 10
    else:
        # Check if there are numbered refs or [Author, Journal, Year] at the end
        has_ref_list = bool(
            re.search(r'\n\[\d+\]\s+[A-Z].*\.\s*\(?\d{4}\)?', response)
            or re.search(r'\n\[[A-Z][a-z]+,\s.*\d{4}\]', response)
        )
        if has_ref_list:
            ev["scores"]["auto_refs"] = 8
        else:
            has_manual_refs = bool(re.search(r'(?i)#{1,3}\s*references', response))
            if has_manual_refs:
                ev["scores"]["auto_refs"] = 5
                ev["issues"].append("Model wrote its own References section (should be auto-generated)")
            else:
                ev["scores"]["auto_refs"] = 3
                ev["issues"].append("No References section at all")

    # 3. Key terms coverage
    terms_found = sum(1 for t in qa["key_terms"] if t.lower() in response.lower())
    term_coverage = terms_found / len(qa["key_terms"]) if qa["key_terms"] else 0
    ev["scores"]["key_terms"] = round(term_coverage * 10, 1)
    ev["terms_found"] = terms_found
    ev["terms_total"] = len(qa["key_terms"])

    # 4. Source depth
    total_reads = tool_info["read_paper"] + tool_info["read_webpage"]
    if config["depth"] == "deep":
        if total_reads >= 8:
            ev["scores"]["source_depth"] = 10
        elif total_reads >= 5:
            ev["scores"]["source_depth"] = 7
        elif total_reads >= 3:
            ev["scores"]["source_depth"] = 5
        else:
            ev["scores"]["source_depth"] = 2
            ev["issues"].append(f"Deep mode but only {total_reads} sources read")
    else:
        if total_reads >= 3:
            ev["scores"]["source_depth"] = 10
        elif total_reads >= 1:
            ev["scores"]["source_depth"] = 6
        else:
            ev["scores"]["source_depth"] = 2

    # 5. Think phase
    if config["depth"] in ("normal", "deep"):
        ev["scores"]["think_phase"] = 10 if tool_info["think_phase"] else 0
        if not tool_info["think_phase"]:
            ev["issues"].append("Think phase not triggered (expected for normal/deep)")
    else:
        ev["scores"]["think_phase"] = 10  # not expected for shallow

    # 6. Response quality (length-based heuristic + structure check)
    quality = 0
    if ev["response_length"] > 2000:
        quality += 3
    elif ev["response_length"] > 1000:
        quality += 2
    elif ev["response_length"] > 400:
        quality += 1

    # Check for structured sections
    if re.search(r'(?i)(overview|introduction|background)', response):
        quality += 1
    if re.search(r'(?i)(findings|results|evidence)', response):
        quality += 1
    if re.search(r'(?i)(synthesis|discussion|implications)', response):
        quality += 1
    if re.search(r'(?i)(limitation|uncertain|open question|future)', response):
        quality += 1
    if re.search(r'(?i)(contrast|contradict|disagree|however|in contrast)', response):
        quality += 1
    if re.search(r'(?i)(consistent with|confirms|supports|in agreement)', response):
        quality += 1

    ev["scores"]["quality"] = min(10, quality)

    # 7. Expected papers
    papers_mentioned = sum(
        1 for kw in qa.get("expected_papers_keywords", [])
        if kw.lower() in response.lower()
    )
    expected_total = len(qa.get("expected_papers_keywords", []))
    if expected_total:
        ev["scores"]["expected_papers"] = round((papers_mentioned / expected_total) * 10, 1)
    else:
        ev["scores"]["expected_papers"] = 5  # N/A

    # 8. QA1 key paper check
    if qa["id"] == "QA1":
        # Must find 2024 eLife paper by Bin Zhang about explicit-ion CG nucleosome stacking
        found_zhang = bool(
            re.search(r'(?i)zhang.*2024.*elife', response)
            or re.search(r'(?i)elife.*2024.*zhang', response)
            or re.search(r'(?i)explicit.ion.*coarse.grain.*stack', response)
            or (re.search(r'(?i)\bzhang\b', response) and re.search(r'(?i)explicit.ion', response))
        )
        ev["qa1_key_paper_found"] = found_zhang
        if found_zhang:
            ev["scores"]["key_paper"] = 10
        else:
            ev["scores"]["key_paper"] = 0
            ev["issues"].append("QA1: Did not find Zhang 2024 eLife explicit-ion CG paper")
    else:
        ev["scores"]["key_paper"] = 5  # N/A for non-QA1

    # Overall score (weighted average)
    weights = {
        "citations": 2,
        "auto_refs": 1,
        "key_terms": 2,
        "source_depth": 2,
        "think_phase": 1,
        "quality": 2,
        "expected_papers": 1,
        "key_paper": 1,
    }
    total_weight = sum(weights.values())
    weighted_sum = sum(ev["scores"].get(k, 0) * w for k, w in weights.items())
    ev["overall_score"] = round(weighted_sum / total_weight, 1)

    return ev


def save_output(qa_id: str, config: dict, result: dict, evaluation: dict):
    """Save test output to file for manual review."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{qa_id}_{config['provider']}_{config['depth']}_{ts}"

    # Save full stdout
    (OUTPUT_DIR / f"{name}_stdout.txt").write_text(result["stdout"], encoding="utf-8")

    # Save evaluation
    (OUTPUT_DIR / f"{name}_eval.json").write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Save response only
    response = extract_response(result["stdout"])
    (OUTPUT_DIR / f"{name}_response.md").write_text(response, encoding="utf-8")


# ----------------------------
# Display
# ----------------------------

def print_eval(ev: dict):
    """Print evaluation summary."""
    label = f"{ev['provider']}/{ev['model']} depth={ev['depth']}"
    print(f"\n{'─'*70}")
    print(f"  {ev['qa_id']} | {label}")
    print(f"  Overall: {ev['overall_score']}/10")
    print(f"{'─'*70}")

    print(f"  Scores:")
    for k, v in ev["scores"].items():
        bar = "█" * int(v) + "░" * (10 - int(v))
        print(f"    {k:20s} {bar} {v}/10")

    print(f"  Citations: {ev['num_citations']} inline refs {ev['citation_numbers'][:5]}...")
    print(f"  Auto-refs: {ev['has_auto_references']}")
    ti = ev["tool_info"]
    print(f"  Tools: {ti['search_papers']} searches, {ti['read_paper']} paper reads, "
          f"{ti['web_search']} web searches, {ti['read_webpage']} web reads")
    print(f"  Think phase: {ti['think_phase']}")
    print(f"  Key terms: {ev['terms_found']}/{ev['terms_total']}")
    print(f"  Response: {ev['response_length']} chars")

    if ev["issues"]:
        print(f"  Issues:")
        for i in ev["issues"]:
            print(f"    ! {i}")


def print_summary(all_evals: list):
    """Print overall summary and comparison."""
    print(f"\n{'═'*70}")
    print(f"  SUMMARY")
    print(f"{'═'*70}")

    # Group by provider
    by_provider = {}
    for ev in all_evals:
        key = f"{ev['provider']}/{ev['depth']}"
        by_provider.setdefault(key, []).append(ev)

    print(f"\n  {'Config':35s} {'Avg':>5s} | {'Citations':>9s} {'Sources':>7s} {'Quality':>7s} {'Terms':>5s}")
    print(f"  {'─'*35} {'─'*5} | {'─'*9} {'─'*7} {'─'*7} {'─'*5}")

    for key, evs in sorted(by_provider.items()):
        # Skip entries with no scores (e.g. timeouts)
        scored = [e for e in evs if e.get("scores")]
        if not scored:
            print(f"  {key:35s}   N/A | (all timed out)")
            continue
        avg = sum(e["overall_score"] for e in scored) / len(scored)
        avg_cite = sum(e["scores"].get("citations", 0) for e in scored) / len(scored)
        avg_src = sum(e["scores"].get("source_depth", 0) for e in scored) / len(scored)
        avg_qual = sum(e["scores"].get("quality", 0) for e in scored) / len(scored)
        avg_terms = sum(e["scores"].get("key_terms", 0) for e in scored) / len(scored)
        print(f"  {key:35s} {avg:5.1f} | {avg_cite:9.1f} {avg_src:7.1f} {avg_qual:7.1f} {avg_terms:5.1f}")

    # Common issues
    all_issues = []
    for ev in all_evals:
        all_issues.extend(ev["issues"])
    if all_issues:
        print(f"\n  Common issues:")
        for issue, count in Counter(all_issues).most_common():
            print(f"    ({count}x) {issue}")

    # Deep vs shallow comparison
    deep_scores = [e["overall_score"] for e in all_evals if e["depth"] == "deep" and e.get("scores")]
    shallow_scores = [e["overall_score"] for e in all_evals if e["depth"] == "shallow" and e.get("scores")]
    if deep_scores and shallow_scores:
        deep_avg = sum(deep_scores) / len(deep_scores)
        shallow_avg = sum(shallow_scores) / len(shallow_scores)
        delta = deep_avg - shallow_avg
        print(f"\n  Deep avg: {deep_avg:.1f} | Shallow avg: {shallow_avg:.1f} | Delta: {delta:+.1f}")


# ----------------------------
# Main
# ----------------------------

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Filter configs by available API keys
    available = [c for c in CONFIGS if get_env_key(c["provider"])]
    if not available:
        print("No provider API keys found. Need at least one of:")
        print("  API_KEY_OPENAI / OPENAI_API_KEY, DEEPSEEK_API_KEY, KIMI_API_KEY")
        sys.exit(1)

    # Allow filtering via CLI args
    if len(sys.argv) > 1:
        filter_arg = sys.argv[1].lower()
        if filter_arg in ("openai", "deepseek", "kimi"):
            available = [c for c in available if c["provider"] == filter_arg]
        elif filter_arg in ("shallow", "deep"):
            available = [c for c in available if c["depth"] == filter_arg]
        elif filter_arg.startswith("qa"):
            # Filter questions
            pass  # handled below

    question_filter = None
    for arg in sys.argv[1:]:
        if arg.upper().startswith("QA"):
            question_filter = arg.upper()

    questions = QUESTIONS
    if question_filter:
        questions = [q for q in QUESTIONS if q["id"] == question_filter]

    print(f"Test Search Quality — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Configs: {len(available)} | Questions: {len(questions)}")
    print(f"Total runs: {len(available) * len(questions)}")
    print()

    for c in available:
        print(f"  {c['provider']}/{c['model']} depth={c['depth']}")

    all_evals = []

    for qa in questions:
        for config in available:
            label = f"{config['provider']}/{config['model']} depth={config['depth']}"
            print(f"\n>>> {qa['id']} | {label}...")

            result = run_question(config, qa["question"])

            if result.get("timeout"):
                print(f"  TIMEOUT after {TIMEOUT_DEEP if config['depth'] == 'deep' else TIMEOUT_SHALLOW}s")
                ev = {
                    "qa_id": qa["id"], "provider": config["provider"],
                    "model": config["model"], "depth": config["depth"],
                    "timeout": True, "overall_score": 0,
                    "scores": {}, "issues": ["TIMEOUT"], "tool_info": {},
                    "num_citations": 0, "citation_numbers": [],
                    "has_auto_references": False, "terms_found": 0,
                    "terms_total": len(qa["key_terms"]), "response_length": 0,
                }
                all_evals.append(ev)
                save_output(qa["id"], config, result, ev)
                continue

            ev = evaluate(qa, config, result)
            all_evals.append(ev)
            save_output(qa["id"], config, result, ev)
            print_eval(ev)

    print_summary(all_evals)

    # Save overall results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = OUTPUT_DIR / f"summary_{ts}.json"
    summary_path.write_text(json.dumps(all_evals, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
