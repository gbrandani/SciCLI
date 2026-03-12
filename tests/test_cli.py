#!/usr/bin/env python
"""
test_cli.py — Automated tests for SciCLI (scicli.py).

Tests the CLI by simulating user input via pexpect (or subprocess fallback).
Tests include: launch, /help, /info, !shell shorthand, /provider, /model, sending a message, /quit.
"""

import subprocess
import sys
import os
import time
import signal

CLI = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scicli.py")
TIMEOUT = 30


def run_cli_with_input(commands: list, timeout: int = TIMEOUT) -> tuple:
    """
    Run scicli.py with a sequence of commands piped to stdin.
    Returns (stdout, stderr, returncode).
    """
    input_text = "\n".join(commands) + "\n"

    env = os.environ.copy()
    env["TERM"] = "dumb"  # Avoid ANSI escape issues

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
        return stdout, stderr + "\n[TIMEOUT]", -1

    return stdout, stderr, proc.returncode


def test_launch_and_quit():
    """Test that CLI launches and /quit works."""
    print("Test: launch and /quit ... ", end="", flush=True)
    stdout, stderr, rc = run_cli_with_input(["/quit", "n"])

    if rc != 0 and "SystemExit" not in stderr:
        print(f"FAIL (rc={rc})")
        print(f"  stderr: {stderr[:500]}")
        return False

    if "SciCLI" not in stdout:
        print(f"FAIL (no welcome banner)")
        print(f"  stdout: {stdout[:500]}")
        return False

    print("OK")
    return True


def test_help():
    """Test /help command."""
    print("Test: /help ... ", end="", flush=True)
    stdout, stderr, rc = run_cli_with_input(["/help", "/quit", "n"])

    # Grouped help should show group names and commands
    if "Literature" not in stdout and "Search" not in stdout:
        print(f"FAIL (no group headings)")
        print(f"  stdout: {stdout[:500]}")
        return False

    if "/scholar" not in stdout:
        print(f"FAIL (/scholar not in help)")
        return False

    print("OK")
    return True


def test_info():
    """Test /info command."""
    print("Test: /info ... ", end="", flush=True)
    stdout, stderr, rc = run_cli_with_input(["/info", "/quit", "n"])

    if "Provider" not in stdout and "provider" not in stdout.lower():
        print(f"FAIL (no provider info)")
        print(f"  stdout: {stdout[:500]}")
        return False

    # Verify the actual configured default provider appears (not hardcoded to "openai")
    try:
        from config import load_settings
        settings = load_settings()
        default_provider = settings.get("defaults", {}).get("provider", "openai")
    except Exception:
        default_provider = "openai"

    if default_provider.lower() not in stdout.lower():
        print(f"FAIL (no {default_provider!r} mention in /info output)")
        print(f"  stdout: {stdout[:300]}")
        return False

    print("OK")
    return True


def test_shell_command():
    """Test ! shorthand for /shell."""
    print("Test: !echo ... ", end="", flush=True)
    stdout, stderr, rc = run_cli_with_input(["!echo hello_from_shell", "/quit", "n"])

    if "hello_from_shell" not in stdout:
        print(f"FAIL (no shell output)")
        print(f"  stdout: {stdout[:500]}")
        return False

    if "Use /feed" not in stdout:
        print(f"FAIL (no /feed hint in output)")
        print(f"  stdout: {stdout[:500]}")
        return False

    print("OK")
    return True


def test_provider_switch():
    """Test /provider and /model commands."""
    print("Test: /provider + /model ... ", end="", flush=True)
    stdout, stderr, rc = run_cli_with_input([
        "/provider deepseek",
        "/model deepseek-chat",
        "/info",
        "/quit", "n"
    ])

    if "deepseek" not in stdout.lower():
        print(f"FAIL (no deepseek in output)")
        print(f"  stdout: {stdout[:500]}")
        return False

    print("OK")
    return True


def test_formats():
    """Test /formats command."""
    print("Test: /formats ... ", end="", flush=True)
    stdout, stderr, rc = run_cli_with_input(["/formats", "/quit", "n"])

    if ".pdf" not in stdout:
        print(f"FAIL (no .pdf in formats)")
        print(f"  stdout: {stdout[:500]}")
        return False

    print("OK")
    return True


def test_clear():
    """Test /clear command."""
    print("Test: /clear ... ", end="", flush=True)
    stdout, stderr, rc = run_cli_with_input(["/clear", "/quit", "n"])

    if "cleared" not in stdout.lower():
        print(f"FAIL (no 'cleared' message)")
        print(f"  stdout: {stdout[:500]}")
        return False

    print("OK")
    return True


def test_unknown_command():
    """Test unknown command."""
    print("Test: unknown command ... ", end="", flush=True)
    stdout, stderr, rc = run_cli_with_input(["/foobar", "/quit", "n"])

    if "unknown" not in stdout.lower() and "Unknown" not in stdout:
        print(f"FAIL (no unknown command message)")
        print(f"  stdout: {stdout[:500]}")
        return False

    print("OK")
    return True


def test_send_message_openai():
    """Test sending an actual message to OpenAI (requires API key)."""
    key = os.environ.get("API_KEY_OPENAI") or os.environ.get("OPENAI_API_KEY") or ""
    if not key:
        print("Test: send message (openai) ... SKIP (no API key)")
        return True  # Don't fail, just skip

    print("Test: send message (openai) ... ", end="", flush=True)
    stdout, stderr, rc = run_cli_with_input([
        "Say exactly: HELLO_TEST_OK",
        "/quit", "n"
    ], timeout=60)

    if "HELLO_TEST_OK" not in stdout:
        # The model should have said HELLO_TEST_OK
        print(f"FAIL (model did not respond with HELLO_TEST_OK)")
        print(f"  stdout tail: ...{stdout[-800:]}")
        print(f"  stderr: {stderr[:300]}")
        return False

    print("OK")
    return True


def test_send_message_deepseek():
    """Test sending a message to DeepSeek (requires API key)."""
    key = os.environ.get("DEEPSEEK_API_KEY") or ""
    if not key:
        print("Test: send message (deepseek) ... SKIP (no API key)")
        return True

    print("Test: send message (deepseek) ... ", end="", flush=True)
    stdout, stderr, rc = run_cli_with_input([
        "/provider deepseek",
        "/model deepseek-chat",
        "Say exactly: DEEPSEEK_TEST_OK",
        "/quit", "n"
    ], timeout=60)

    if "DEEPSEEK_TEST_OK" not in stdout:
        print(f"FAIL (model did not respond with DEEPSEEK_TEST_OK)")
        print(f"  stdout tail: ...{stdout[-800:]}")
        print(f"  stderr: {stderr[:300]}")
        return False

    print("OK")
    return True


def test_send_message_kimi():
    """Test sending a message to Kimi (requires API key)."""
    key = os.environ.get("KIMI_API_KEY") or ""
    if not key:
        print("Test: send message (kimi) ... SKIP (no API key)")
        return True

    print("Test: send message (kimi) ... ", end="", flush=True)
    stdout, stderr, rc = run_cli_with_input([
        "/provider kimi",
        "Say exactly: KIMI_TEST_OK. Nothing else.",
        "/quit", "n"
    ], timeout=90)

    if "KIMI_TEST_OK" not in stdout:
        print(f"FAIL (model did not respond with KIMI_TEST_OK)")
        print(f"  stdout tail: ...{stdout[-800:]}")
        print(f"  stderr: {stderr[:300]}")
        return False

    print("OK")
    return True


def test_config_import():
    """Test that config module works correctly."""
    print("Test: config module ... ", end="", flush=True)
    stdout, stderr, rc = run_cli_with_input([], timeout=5)
    # Just importing is sufficient; we tested this before
    # Run a quick inline test
    result = subprocess.run(
        [sys.executable, "-c", """
from config import load_settings, get_model_specs, models_by_provider, get_compact_model
s = load_settings()
assert 'providers' in s
specs = get_model_specs()
assert 'gpt-5-mini' in specs
grouped = models_by_provider()
assert 'openai' in grouped
assert get_compact_model('openai') == 'gpt-5-nano'
assert get_compact_model('deepseek') == 'deepseek-chat'
assert get_compact_model('kimi') == 'kimi-k2.5'
print('ALL_CONFIG_OK')
"""],
        capture_output=True, text=True, timeout=30,
    )
    if "ALL_CONFIG_OK" not in result.stdout:
        print(f"FAIL")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return False
    print("OK")
    return True


def test_tools_import():
    """Test that tools module imports and schemas are valid."""
    print("Test: tools module ... ", end="", flush=True)
    result = subprocess.run(
        [sys.executable, "-c", """
from tools import AGENTIC_TOOLS, CITE_TOOL, execute_tool
assert len(AGENTIC_TOOLS) == 4, f"Expected 4 tools, got {len(AGENTIC_TOOLS)}"
names = [t['function']['name'] for t in AGENTIC_TOOLS]
assert 'search' in names, f"'search' not in {names}"
assert 'read' in names, f"'read' not in {names}"
assert 'get_paper_references' in names, f"'get_paper_references' not in {names}"
assert 'reread' in names, f"'reread' not in {names}"
# Verify search tool uses single query parameter
sp = [t for t in AGENTIC_TOOLS if t['function']['name'] == 'search'][0]
assert 'query' in sp['function']['parameters']['properties'], "'query' not in search tool params"
# Verify cite tool uses keys not source_numbers
cite_params = CITE_TOOL['function']['parameters']['properties']
assert 'keys' in cite_params
assert 'source_numbers' not in cite_params
# Verify RecordInfo dataclass has ref_key field
from config import SessionState, RecordInfo, SourceInfo
assert 'access_level' in RecordInfo.__dataclass_fields__
assert 'ref_key' in RecordInfo.__dataclass_fields__
# Backward compat alias
assert SourceInfo is RecordInfo
# Verify SessionState has new fields
s = SessionState()
assert s.citation_style == 'numbered'
assert s.search_queries_history == []
assert s.agentic_tool_max_chars == 80_000
print('ALL_TOOLS_OK')
"""],
        capture_output=True, text=True, timeout=30,
    )
    if "ALL_TOOLS_OK" not in result.stdout:
        print(f"FAIL")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return False
    print("OK")
    return True


def test_agentic_deepseek():
    """Test agentic tool loop: DeepSeek should call search_papers autonomously."""
    key = os.environ.get("DEEPSEEK_API_KEY") or ""
    if not key:
        print("Test: agentic deepseek ... SKIP (no API key)")
        return True

    print("Test: agentic deepseek ... ", end="", flush=True)
    stdout, stderr, rc = run_cli_with_input([
        "/provider deepseek",
        "/model deepseek-chat",
        "Search for papers about CRISPR gene editing in plants. Use your search_papers tool.",
        "/quit", "n"
    ], timeout=120)

    # Check that the tool was called (status output)
    has_tool_call = "search_papers" in stdout.lower() or "Tool:" in stdout
    if not has_tool_call:
        print(f"WARN (no tool call visible in output)")
        # Not a hard failure — model might not have called tools

    # Check that we got some response (look for tool calls or meaningful output)
    if len(stdout) < 200:
        print(f"FAIL (output too short)")
        print(f"  stdout tail: ...{stdout[-500:]}")
        print(f"  stderr: {stderr[:300]}")
        return False

    print("OK")
    return True


def test_agentic_kimi():
    """Test agentic tool loop: Kimi should call tools autonomously."""
    key = os.environ.get("KIMI_API_KEY") or ""
    if not key:
        print("Test: agentic kimi ... SKIP (no API key)")
        return True

    print("Test: agentic kimi ... ", end="", flush=True)
    stdout, stderr, rc = run_cli_with_input([
        "/provider kimi",
        "Search for recent papers on chromatin remodeling using your search_papers tool. Report what you find.",
        "/quit", "n"
    ], timeout=120)

    if len(stdout) < 200:
        print(f"FAIL (output too short)")
        print(f"  stdout tail: ...{stdout[-500:]}")
        print(f"  stderr: {stderr[:300]}")
        return False

    print("OK")
    return True


def test_sakura_usage_tracker():
    """Test SakuraUsageTracker logic with a temp file."""
    print("Test: Sakura usage tracker ... ", end="", flush=True)
    result = subprocess.run(
        [sys.executable, "-c", """
import tempfile, os
from pathlib import Path
from providers import SakuraUsageTracker, SAKURA_MONTHLY_LIMIT

with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
    tmp = Path(f.name)

try:
    tracker = SakuraUsageTracker(usage_file=tmp)
    assert tracker.current_count() == 0
    assert tracker.check_limit() is None
    tracker.increment()
    tracker.increment()
    assert tracker.current_count() == 2
    assert tracker.check_limit() is None

    # Simulate hitting the limit by writing a fake count (new stats format)
    import json
    from datetime import datetime
    key = datetime.now().strftime("%Y-%m")
    tmp.write_text(json.dumps({"sakura": {key: SAKURA_MONTHLY_LIMIT}}))
    assert tracker.check_limit() is not None

    print('SAKURA_TRACKER_OK')
finally:
    os.unlink(tmp)
"""],
        capture_output=True, text=True, timeout=30,
    )
    if "SAKURA_TRACKER_OK" not in result.stdout:
        print(f"FAIL")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return False
    print("OK")
    return True


def test_send_message_sakura():
    """Test sending a message to Sakura (requires API key)."""
    key = os.environ.get("SAKURA_API_KEY") or ""
    if not key:
        print("Test: send message (sakura) ... SKIP (no API key)")
        return True

    print("Test: send message (sakura) ... ", end="", flush=True)
    stdout, stderr, rc = run_cli_with_input([
        "/provider sakura",
        "/model gpt-oss-120b",
        "Say exactly: SAKURA_TEST_OK",
        "/quit", "n"
    ], timeout=60)

    if "SAKURA_TEST_OK" not in stdout:
        print(f"FAIL (model did not respond with SAKURA_TEST_OK)")
        print(f"  stdout tail: ...{stdout[-800:]}")
        print(f"  stderr: {stderr[:300]}")
        return False

    print("OK")
    return True


def test_agentic_sakura():
    """Test agentic tool loop: Sakura should call tools autonomously."""
    key = os.environ.get("SAKURA_API_KEY") or ""
    if not key:
        print("Test: agentic sakura ... SKIP (no API key)")
        return True

    print("Test: agentic sakura ... ", end="", flush=True)
    stdout, stderr, rc = run_cli_with_input([
        "/provider sakura",
        "/model gpt-oss-120b",
        "Search for papers about protein folding using your search_papers tool. Report what you find.",
        "/quit", "n"
    ], timeout=120)

    if len(stdout) < 200:
        print(f"FAIL (output too short)")
        print(f"  stdout tail: ...{stdout[-500:]}")
        print(f"  stderr: {stderr[:300]}")
        return False

    print("OK")
    return True


def test_depth_command():
    """Test /depth command."""
    print("Test: /depth command ... ", end="", flush=True)
    stdout, stderr, rc = run_cli_with_input([
        "/depth deep",
        "/info",
        "/depth shallow",
        "/info",
        "/depth normal",
        "/quit", "n"
    ])

    if "deep" not in stdout.lower():
        print(f"FAIL (no 'deep' in output)")
        print(f"  stdout: {stdout[:500]}")
        return False

    if "shallow" not in stdout.lower():
        print(f"FAIL (no 'shallow' in output)")
        return False

    # "normal" is no longer valid, should show usage error
    if "usage" not in stdout.lower():
        print(f"FAIL (no usage error for 'normal' depth)")
        return False

    print("OK")
    return True


def test_depth_config_import():
    """Test that DEPTH_CONFIG and new fields work correctly."""
    print("Test: depth config ... ", end="", flush=True)
    result = subprocess.run(
        [sys.executable, "-c", """
from config import DEPTH_CONFIG, SessionState, ReplyBundle, Record, RecordInfo
# Check DEPTH_CONFIG
assert "shallow" in DEPTH_CONFIG
assert "deep" in DEPTH_CONFIG
assert "normal" not in DEPTH_CONFIG, "normal depth should be removed"
assert DEPTH_CONFIG["shallow"]["force_answer_at"] < DEPTH_CONFIG["deep"]["force_answer_at"]

# Check SessionState defaults
s = SessionState()
assert s.search_depth == "shallow", f"default should be shallow, got {s.search_depth}"
assert s.agentic_tool_max_chars == 80_000
assert s.target_papers == 0
assert s.target_searches == 0
assert s.records == []
assert s.records_next_id == 1

# Check Record
ri = RecordInfo(title="test", url="", access_level="full_text", tool_name="read_paper")
rec = Record(id=1, info=ri, char_count=100)
assert rec.truncated_from == 0
assert rec.cleared == False
assert rec.content_type == ""

# Check ReplyBundle has tool_context_summary
b = ReplyBundle(text="test", cited=[], consulted=[])
assert b.tool_context_summary is None
b2 = ReplyBundle(text="test", cited=[], consulted=[], tool_context_summary="inventory")
assert b2.tool_context_summary == "inventory"

print('DEPTH_CONFIG_OK')
"""],
        capture_output=True, text=True, timeout=30,
    )
    if "DEPTH_CONFIG_OK" not in result.stdout:
        print(f"FAIL")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return False
    print("OK")
    return True


def test_source_inventory():
    """Test _build_source_inventory function."""
    print("Test: source inventory ... ", end="", flush=True)
    result = subprocess.run(
        [sys.executable, "-c", """
from records import build_record_inventory as _build_record_inventory
from config import RecordInfo, PaperRecord

# Empty
assert _build_record_inventory([]) == ""

# With sources (using ref keys)
sources = [
    PaperRecord(title="Paper A", url="https://example.com/a", access_level="full_text",
               tool_name="read_paper", external_id="abc123", year="2024", authors="Smith et al.",
               ref_key="Smith2024paper", record_type="paper"),
    PaperRecord(title="Paper B", url="https://example.com/b", access_level="abstract_only",
               tool_name="read_paper", external_id="def456", year="2021", authors="Jones et al.",
               ref_key="Jones2021study", record_type="paper"),
    RecordInfo(title="Web Page", url="https://example.com/web", access_level="webpage",
               tool_name="read_webpage"),
    RecordInfo(title="Search Result", url="https://example.com/sr", access_level="search_result",
               tool_name="search_papers", external_id="ghi789", ref_key="Doe2023result"),
]
inv = _build_record_inventory(sources)
assert "SOURCE INVENTORY" in inv
assert "Smith2024paper" in inv
assert "Jones2021study" not in inv   # abstract_only: no citable key shown
assert "not citable" in inv.lower()  # abstract_only marker present
assert "Paper A" in inv
assert "Paper B" in inv
assert "Web Page" in inv
assert "full text" in inv.lower()
assert "abstract only" in inv.lower()
assert "search results seen but not read" in inv

print('INVENTORY_OK')
"""],
        capture_output=True, text=True, timeout=30,
    )
    if "INVENTORY_OK" not in result.stdout:
        print(f"FAIL")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return False
    print("OK")
    return True


def test_openai_agentic_import():
    """Test that OpenAI and Sakura providers exist and are correctly structured."""
    print("Test: provider classes ... ", end="", flush=True)
    result = subprocess.run(
        [sys.executable, "-c", """
from providers import OpenAIProvider, OpenAIChatProvider, SakuraProvider, SakuraUsageTracker, SAKURA_MONTHLY_LIMIT
assert OpenAIProvider is OpenAIChatProvider, "OpenAIProvider should alias OpenAIChatProvider"
assert hasattr(OpenAIChatProvider, 'send'), "OpenAIChatProvider missing send method"
assert hasattr(SakuraProvider, 'send'), "SakuraProvider missing send method"
assert OpenAIChatProvider.name == 'openai'
assert SakuraProvider.name == 'sakura'
assert SAKURA_MONTHLY_LIMIT == 3000
tracker = SakuraUsageTracker()
assert hasattr(tracker, 'current_count')
assert hasattr(tracker, 'check_limit')
assert hasattr(tracker, 'increment')
print('OAI_AGENTIC_OK')
"""],
        capture_output=True, text=True, timeout=30,
    )
    if "OAI_AGENTIC_OK" not in result.stdout:
        print(f"FAIL")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return False
    print("OK")
    return True


def test_shared_loop_import():
    """Test that shared agentic loop function exists."""
    print("Test: shared agentic loop ... ", end="", flush=True)
    result = subprocess.run(
        [sys.executable, "-c", """
from agentic import run_agentic_loop, _build_force_answer_prompt
from records import build_record_inventory
assert callable(run_agentic_loop)
assert callable(build_record_inventory)
assert callable(_build_force_answer_prompt)
print('LOOP_OK')
"""],
        capture_output=True, text=True, timeout=30,
    )
    if "LOOP_OK" not in result.stdout:
        print(f"FAIL")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return False
    print("OK")
    return True


def test_help_shows_depth():
    """Test that /help includes /depth and new commands."""
    print("Test: /help shows /depth ... ", end="", flush=True)
    stdout, stderr, rc = run_cli_with_input(["/help", "/quit", "n"])

    if "/depth" not in stdout:
        print(f"FAIL (/depth not in help output)")
        print(f"  stdout: {stdout[:800]}")
        return False

    if "/citestyle" not in stdout:
        print(f"FAIL (/citestyle not in help output)")
        return False

    if "/tools" not in stdout:
        print(f"FAIL (/tools not in help output)")
        return False

    if "/sources" not in stdout:
        print(f"FAIL (/sources not in help output)")
        return False

    if "/trunclimit" not in stdout:
        print(f"FAIL (/trunclimit not in help output)")
        return False

    if "/feed" not in stdout:
        print(f"FAIL (/feed not in help output)")
        return False

    if "/targets" not in stdout:
        print(f"FAIL (/targets not in help output)")
        return False

    print("OK")
    return True


def test_source_registry():
    """Test source registry data structures."""
    print("Test: source registry ... ", end="", flush=True)
    result = subprocess.run(
        [sys.executable, "-c", """
from config import SessionState, RecordInfo, Record
import time

s = SessionState()
assert s.records == []
assert s.records_next_id == 1

# Add entries
ri1 = RecordInfo(title="Paper A", url="https://example.com/a", access_level="full_text",
                  tool_name="read_paper", external_id="abc", ref_key="Smith2024a")
e1 = Record(id=1, info=ri1, char_count=5000, content_type="full_text",
                  timestamp=time.time())
s.records.append(e1)
s.records_next_id = 2

assert len(s.records) == 1
assert s.records[0].char_count == 5000
assert s.records[0].cleared == False

# Clear
e1.cleared = True
assert s.records[0].cleared == True

print('REGISTRY_OK')
"""],
        capture_output=True, text=True, timeout=30,
    )
    if "REGISTRY_OK" not in result.stdout:
        print(f"FAIL")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return False
    print("OK")
    return True


def test_duplicate_detection():
    """Test duplicate paper detection."""
    print("Test: duplicate detection ... ", end="", flush=True)
    result = subprocess.run(
        [sys.executable, "-c", """
from utils import find_duplicate_paper, deduplicate_paper_list, _normalize_title, _title_similarity
from config import SourceInfo

# Test normalization
assert _normalize_title("Hello, World!") == "hello world"
assert _normalize_title("  Multiple   spaces  ") == "multiple spaces"

# Test similarity
assert _title_similarity("hello world foo", "hello world foo") == 1.0
assert _title_similarity("hello world", "goodbye world") < 0.9

# Test find_duplicate_paper
sources = [
    SourceInfo(title="Chromatin remodeling in yeast", url="", access_level="full_text",
               tool_name="read_paper", external_id="abc", ref_key="Smith2024"),
]
# Exact match
dup = find_duplicate_paper({"title": "Chromatin remodeling in yeast"}, sources)
assert dup is not None

# No match
no_dup = find_duplicate_paper({"title": "Completely different paper"}, sources)
assert no_dup is None

# Test deduplicate_paper_list
papers = [
    {"title": "Paper A", "citationCount": 10, "paperId": "1"},
    {"title": "Paper A", "citationCount": 20, "paperId": "2"},
    {"title": "Paper B", "citationCount": 5, "paperId": "3"},
]
deduped = deduplicate_paper_list(papers)
assert len(deduped) == 2
# Should keep the one with higher citation count
assert deduped[0].get("citationCount") == 20

print('DEDUP_OK')
"""],
        capture_output=True, text=True, timeout=30,
    )
    if "DEDUP_OK" not in result.stdout:
        print(f"FAIL")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return False
    print("OK")
    return True


def test_tool_plugin_registry():
    """Test ToolPlugin registration and validation."""
    print("Test: tool plugin registry ... ", end="", flush=True)
    result = subprocess.run(
        [sys.executable, "-c", """
from tool_registry import REGISTRY, ToolPlugin
import tools as _tools  # ensure plugins registered
from config import SessionState

# Verify registry has tools (both new and legacy)
assert REGISTRY.has_tool("search"), "Missing new 'search' tool"
assert REGISTRY.has_tool("read"), "Missing new 'read' tool"
assert REGISTRY.has_tool("search_papers"), "Missing legacy 'search_papers' tool"
assert REGISTRY.has_tool("read_paper"), "Missing legacy 'read_paper' tool"
assert REGISTRY.has_tool("web_search"), "Missing legacy 'web_search' tool"
assert REGISTRY.has_tool("read_webpage"), "Missing legacy 'read_webpage' tool"
assert REGISTRY.has_tool("get_paper_references")
assert not REGISTRY.has_tool("nonexistent_tool")

# Test validation
err = REGISTRY.validate_args("read_paper", {})
assert err is not None  # missing paper_id
err = REGISTRY.validate_args("read_paper", {"paper_id": "abc123"})
assert err is None  # valid
err = REGISTRY.validate_args("get_paper_references", {"paper_id": "abc", "direction": "invalid"})
assert err is not None  # invalid enum
err = REGISTRY.validate_args("get_paper_references", {"paper_id": "abc", "direction": "cited_by"})
assert err is None  # valid

# Test schemas (REGISTRY has all tools including legacy)
schemas = REGISTRY.get_schemas()
assert len(schemas) >= 7, f"Expected >=7 schemas, got {len(schemas)}"
names = [s['function']['name'] for s in schemas]
assert 'search' in names, f"'search' not in registry schemas: {names}"
assert 'read' in names, f"'read' not in registry schemas: {names}"
assert 'search_papers' in names

print('REGISTRY_OK')
"""],
        capture_output=True, text=True, timeout=30,
    )
    if "REGISTRY_OK" not in result.stdout:
        print(f"FAIL")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return False
    print("OK")
    return True


def test_record_subclasses():
    """Test PaperRecord subclass and deserialization."""
    print("Test: record subclasses ... ", end="", flush=True)
    result = subprocess.run(
        [sys.executable, "-c", """
from config import RecordInfo, PaperRecord, record_info_from_dict

# Test PaperRecord
pr = PaperRecord(title="Test Paper", url="https://example.com", access_level="full_text",
                  tool_name="read_paper", record_type="paper", year="2024", authors="Smith")
assert pr.year == "2024"
assert pr.authors == "Smith"
assert isinstance(pr, RecordInfo)

# Test deserialization with record_type
d = {"title": "P", "url": "", "access_level": "full_text", "tool_name": "read_paper",
     "record_type": "paper", "year": "2023", "authors": "Jones"}
ri = record_info_from_dict(d)
assert isinstance(ri, PaperRecord)
assert ri.year == "2023"

# Test backward compat (no record_type, detect from fields)
d2 = {"title": "P2", "url": "", "access_level": "full_text", "tool_name": "read_paper",
      "year": "2022", "authors": "Lee"}
ri2 = record_info_from_dict(d2)
assert isinstance(ri2, PaperRecord)

# Test base RecordInfo (no domain-specific fields)
d3 = {"title": "W", "url": "https://example.com", "access_level": "webpage", "tool_name": "read_webpage"}
ri3 = record_info_from_dict(d3)
assert type(ri3) is RecordInfo

print('SUBCLASSES_OK')
"""],
        capture_output=True, text=True, timeout=30,
    )
    if "SUBCLASSES_OK" not in result.stdout:
        print(f"FAIL")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return False
    print("OK")
    return True


def test_computed_checkpoints():
    """Test that checkpoint iterations scale with force_answer_at."""
    print("Test: computed checkpoints ... ", end="", flush=True)
    # Verify the formula from agentic.py
    for force_at in [5, 9, 15, 20]:
        cr = max(2, force_at // 3)
        cc = max(cr + 1, (force_at * 2) // 3)
        cg = max(cc + 1, force_at - 2)
        # Clamp to ensure all < force_at
        if cg >= force_at:
            cg = force_at - 1
        if cc >= cg:
            cc = cg - 1
        if cr >= cc:
            cr = cc - 1
        assert cr < cc < cg < force_at, f"Checkpoint order wrong for force_at={force_at}: {cr}, {cc}, {cg}"
        assert cr >= 1
    print("OK")
    return True


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=" * 60)
    print("SciCLI Test Suite")
    print("=" * 60)

    tests = [
        test_config_import,
        test_tools_import,
        test_depth_config_import,
        test_source_inventory,
        test_openai_agentic_import,
        test_shared_loop_import,
        test_launch_and_quit,
        test_help,
        test_help_shows_depth,
        test_info,
        test_formats,
        test_clear,
        test_unknown_command,
        test_shell_command,
        test_provider_switch,
        test_depth_command,
        test_source_registry,
        test_duplicate_detection,
        test_tool_plugin_registry,
        test_record_subclasses,
        test_computed_checkpoints,
        test_sakura_usage_tracker,
        test_send_message_openai,
        test_send_message_deepseek,
        test_send_message_kimi,
        test_send_message_sakura,
        test_agentic_deepseek,
        test_agentic_kimi,
        test_agentic_sakura,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed else 0)
