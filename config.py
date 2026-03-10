"""
config.py — Settings, constants, and data structures for SciCLI.

Loads configuration from ~/.config/scicli/settings.json (fallback: ~/.config/chat_cli/settings.json for backward compat, then ./settings.json, then defaults).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Constants
# ----------------------------

APP_NAME = "SciCLI"

# Data directories are anchored to the directory containing this file so that
# scicli runs correctly regardless of the working directory (e.g. when invoked
# via a symlink from ~/bin or from a different project directory).
_APP_DIR = Path(__file__).resolve().parent

HISTORY_FILE = _APP_DIR / ".llmcli_history"
CONV_DIR     = _APP_DIR / "conversations"
UPLOAD_DIR   = _APP_DIR / "uploads"
STATS_FILE   = _APP_DIR / ".llmcli_stats.json"

DEFAULT_CONTEXT_MESSAGES = 10  # sliding window length
DEFAULT_MAX_FILE_BYTES = 20 * 1024 * 1024

# When to compact conversation (as fraction of context window used by input tokens)
COMPACT_AT_FRACTION = 0.85

# ----------------------------
# Command registry types
# ----------------------------

@dataclass
class CommandInfo:
    handler: str           # method name on LLMChatClient, e.g. "cmd_help"
    group: str             # key into COMMAND_GROUPS
    short_help: str        # one-line for /help table
    example: str           # e.g. "/read myfile.pdf"
    detailed_help: str     # multi-paragraph prose explaining methodology
    arg_spec: str = ""     # e.g. "<query>", "[on|off]", ""

COMMAND_GROUPS = {
    "search": "Literature & Web Search",
    "sources": "Source Management",
    "files": "File Operations",
    "settings": "Configuration",
    "conversation": "Conversation Management",
    "display": "Display & Output",
}

# Web browsing defaults
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_BRAVE_COUNT = 20
DEFAULT_MAX_OPEN_PAGES = 6
DEFAULT_WEB_CHAR_BUDGET_PER_PAGE = 12_000
DEFAULT_WEB_TOTAL_CHAR_BUDGET = 50_000
DEFAULT_BRAVE_WARN_THRESHOLD = 1000

# Semantic Scholar (Graph API)
S2_ENDPOINT_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_ENDPOINT_PAPER = "https://api.semanticscholar.org/graph/v1/paper/{}"
S2_ENDPOINT_CITATIONS = "https://api.semanticscholar.org/graph/v1/paper/{}/citations"
S2_ENDPOINT_REFERENCES = "https://api.semanticscholar.org/graph/v1/paper/{}/references"
DEFAULT_S2_LIMIT = 10
DEFAULT_S2_LIMIT_MAX = 100
DEFAULT_S2_CHAR_BUDGET_METADATA = 60_000
DEFAULT_S2_WARN_THRESHOLD = 2000

# PDF extraction heuristics
MIN_TEXT_CHARS_TO_ASSUME_NOT_SCANNED = 2000
DEFAULT_OCR_DPI_SCALE = 2.0

# Agentic tool limits
AGENTIC_TOOL_MAX_CHARS = 40_000  # truncate tool results to stay within context
AGENTIC_MAX_ITERATIONS = 18      # absolute max tool-call loop iterations (deep mode)

# Search depth configuration
DEPTH_CONFIG: Dict[str, Dict[str, Any]] = {
    "shallow": {
        "force_answer_at": 5,
        "max_iterations": 8,
        "use_think_phase": False,
        "prompt": "Quick search: 1-2 queries, read 2-3 papers.",
        "default_papers": 3,
        "default_searches": 2,
    },
    "deep": {
        "force_answer_at": 14,
        "max_iterations": 18,
        "use_think_phase": True,
        "prompt": (
            "Thorough investigation: use 8-10 different query formulations to cover the topic broadly. "
            "Read 8-12 papers (prioritize reviews and highly-cited works). "
            "After initial reading, use get_paper_references to follow citation trails from your best finds. "
            "Identify gaps and do targeted follow-up searches. "
            "Also search the web for complementary non-academic sources when relevant."
        ),
        "default_papers": 8,
        "default_searches": 8,
    },
}


# ----------------------------
# Default settings (used when no JSON file found)
# ----------------------------

DEFAULT_SETTINGS: Dict[str, Any] = {
    "providers": {
        "openai": {
            "api_key_env": "API_KEY_OPENAI",
            "api_key_env_alt": "OPENAI_API_KEY",
            "compact_model": "gpt-5-nano",
            "models": {
                "gpt-5-mini": {"context": 400_000, "max_output": 128_000},
                "gpt-5-nano": {"context": 400_000, "max_output": 128_000},
                "gpt-5.2": {"context": 400_000, "max_output": 128_000},
                "gpt-5.2-pro": {"context": 400_000, "max_output": 128_000},
            },
        },
        "sakura": {
            "base_url": "https://api.ai.sakura.ad.jp/v1",
            "api_key_env": "SAKURA_API_KEY",
            "compact_model": "gpt-oss-120b",
            "models": {
                "gpt-oss-120b": {"context": 32_000, "max_output": 8_000, "supports_tools": True},
                "Qwen3-Coder-480B-A35B-Instruct-FP8": {"context": 32_000, "max_output": 8_000},
                "llm-jp-3.1-8x13b-instruct4": {"context": 32_000, "max_output": 4_000},
            },
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
            "compact_model": "deepseek-chat",
            "models": {
                "deepseek-chat": {"context": 128_000, "max_output": 8_000, "supports_tools": True},
                "deepseek-reasoner": {"context": 128_000, "max_output": 32_000},
            },
        },
        "kimi": {
            "base_url": "https://api.moonshot.ai/v1",
            "api_key_env": "KIMI_API_KEY",
            "compact_model": "kimi-k2.5",
            "models": {
                "kimi-k2.5": {"context": 256_000, "max_output": 16_000, "supports_tools": True},
            },
        },
    },
    "defaults": {
        "provider": "sakura",
        "model": "gpt-oss-120b",
    },
    "ui": {
        "codecolor": True,
        "auto_ocr": True,
        "theme": "nord",       # theme file name in themes/ directory (no extension)
        "colors": {},          # per-key overrides applied on top of the theme file
    },
    "prompts": {
        "system_preamble": "",            # prepended to system prompt
        "search_strategy_override": "",   # replaces SEARCH STRATEGY section if non-empty
        "synthesis_instructions": "",     # replaces SYNTHESIS INSTRUCTIONS if non-empty
    },
}


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class SessionState:
    provider: str = "openai"
    model: str = "gpt-5-mini"
    reasoning_effort: str = "auto"  # OpenAI: auto|none|low|medium|high|xhigh
    context_messages: int = DEFAULT_CONTEXT_MESSAGES
    auto_compact: bool = True

    # display / UX
    codecolor: bool = True   # True = Rich Markdown rendering; False = raw text fallback
    theme: str = "academic"  # theme name; must match a file in themes/

    # Brave
    brave_warn_threshold: int = DEFAULT_BRAVE_WARN_THRESHOLD
    brave_count: int = DEFAULT_BRAVE_COUNT  # results per /web search (Brave API max: 20/call)
    max_open_pages: int = DEFAULT_MAX_OPEN_PAGES

    # Display / verbose
    verbose: bool = False  # show full tool context (extra_snippets, LLM contexts) in output

    # Semantic Scholar
    s2_page_size: int = DEFAULT_S2_LIMIT

    # Agentic tool limits (configurable per-session)
    agentic_tool_max_chars: int = 80_000  # truncate tool results; 0 = no truncation

    # Search depth and pipeline
    search_depth: str = "shallow"  # "shallow" or "deep"
    research_pipeline: bool = True  # Enable 3-phase Think→Search→Synthesize pipeline
    target_papers: int = 0     # 0 = use depth default; soft hint in system prompt
    target_searches: int = 0   # 0 = use depth default; soft hint in system prompt
    force_answer_at: int = 0   # 0 = use depth default; hard limit
    max_iterations: int = 0    # 0 = use depth default; hard limit
    use_think_phase: int = -1  # -1 = use depth default; 0 = off; 1 = on

    # Citation display
    citation_style: str = "numbered"  # "numbered" or "authoryear"
    tools_enabled: bool = True

    # Search behaviour
    search_mode: str = "auto"   # "auto" (router call) | "on" (always) | "off" (never)
    domain_filter: str = "web"  # "web" | "academic" (Brave Goggle re-ranking)

    # Search query history (for TF-IDF ranking)
    search_queries_history: List[str] = field(default_factory=list)

    # Record registry
    records: List['Record'] = field(default_factory=list)
    records_next_id: int = 1

    # PDF/OCR
    auto_ocr_on_scans: bool = True
    ocr_scale: float = DEFAULT_OCR_DPI_SCALE

    # File registry
    file_registry: List['FileEntry'] = field(default_factory=list)
    file_registry_next_id: int = 1

    # Pinned records (Group 1 — always in context, never compacted)
    pinned_records: List['PinnedRecord'] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Runtime-only registry for reread tool: ref_key → {url, external_id, local_path, title}
        # Not a dataclass field so it is never serialized to disk.
        object.__setattr__(self, 'reread_registry', {})
        # Runtime-only URL → llm_contexts cache (Brave extra_snippets).
        # Populated by _tool_search; consumed by _tool_read to attach contexts to PaperRecord.
        object.__setattr__(self, 'search_contexts', {})
        # Runtime-only session-global snap key counter.
        # Incremented by _tool_search so snap keys are unique across multiple search calls.
        object.__setattr__(self, 'search_snap_counter', 0)
        # Runtime-only registry for /snippets command: list of batch dicts
        # {query, record_id, snippets: [RecordInfo, ...]}. Never serialized.
        object.__setattr__(self, 'search_snap_registry', [])


@dataclass
class AppStats:
    """Persistent usage statistics saved to .llmcli_stats.json.

    counts stores monthly totals per service:
      {"brave": {"2026-03": 42}, "sakura": {"2026-03": 150}, ...}
    Services: brave, s2, openai, deepseek, kimi, sakura.
    """
    counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    last_updated_utc: str = ""

    def monthly(self, service: str, month: str = "") -> int:
        """Monthly count for a service (defaults to current month)."""
        if not month:
            import datetime as _dt
            month = _dt.datetime.now().strftime("%Y-%m")
        return self.counts.get(service, {}).get(month, 0)

    def total(self, service: str) -> int:
        """All-time total for a service (sum of all monthly values)."""
        return sum(self.counts.get(service, {}).values())


@dataclass
class FileEntry:
    id: int
    path: str                  # absolute resolved path
    filename: str              # basename
    size: int                  # bytes at time of read
    mtime: float               # os.path.getmtime() at time of read
    char_count: int            # chars fed to model
    content_hash: str          # hashlib.md5 for change detection
    last_read_timestamp: float # time.time()
    truncated_from: int = 0


@dataclass
class PinnedRecord:
    """A source pinned to always stay in context (Group 1 — never compacted)."""
    ref_key: str           # citation key (BibTeX key or snap/web key)
    title: str             # display title
    content: str           # full text or abstract — stored WITHOUT truncation
    source_type: str = ""  # "search", "webpage", "paper", "file"
    pinned_at: float = 0.0 # time.time()
    note: str = ""         # user-supplied annotation
    local_path: str = ""   # for Type D (local files) — enables mtime change detection
    loaded_mtime: float = 0.0  # os.path.getmtime() when last loaded
    # Rich metadata for model context (populated from RecordInfo at pin time)
    authors: str = ""
    year: str = ""
    venue: str = ""
    url: str = ""
    access_level: str = ""


@dataclass
class RecordInfo:
    """Base record metadata. All tools produce at least these fields."""
    title: str
    url: str
    access_level: str      # "full_text", "abstract_only", "metadata_only", "failed", "webpage", "snippet", "search_result"
    tool_name: str         # which tool produced this
    ref_key: str = ""      # citation/reference key (bibtex key for papers, accession for sequences)
    external_id: str = ""  # external DB identifier (S2 paper_id, NCBI accession, UniProt ID)
    record_type: str = ""  # discriminator: "paper", "sequence", "webpage", ""
    source_type: str = ""  # semantic group: "search", "webpage", "paper", "file"
    local_path: str = ""   # local file path if content was saved (PDF); empty otherwise
    llm_contexts: List[str] = field(default_factory=list)  # Brave extra_snippets for this URL


@dataclass
class PaperRecord(RecordInfo):
    """Extended metadata for academic papers."""
    year: str = ""
    authors: str = ""
    venue: str = ""
    abstract: str = ""     # populated from S2; empty for HTML-sourced or local-file papers


def record_info_from_dict(d: dict) -> RecordInfo:
    """Reconstruct the right RecordInfo subclass from a serialized dict."""
    rt = d.get("record_type", "")

    if rt == "paper":
        return PaperRecord(**{k: v for k, v in d.items() if k in PaperRecord.__dataclass_fields__})
    elif rt:
        return RecordInfo(**{k: v for k, v in d.items() if k in RecordInfo.__dataclass_fields__})

    # Backward compat: no record_type — detect from fields
    if d.get("year") or d.get("authors") or d.get("venue"):
        d_copy = dict(d)
        d_copy["record_type"] = "paper"
        return PaperRecord(**{k: v for k, v in d_copy.items() if k in PaperRecord.__dataclass_fields__})
    else:
        return RecordInfo(**{k: v for k, v in d.items() if k in RecordInfo.__dataclass_fields__})


@dataclass
class Record:
    id: int                    # auto-increment
    info: RecordInfo           # record metadata
    char_count: int            # chars actually fed into model
    truncated_from: int = 0    # original char count if truncated (0 = not truncated)
    content_type: str = ""     # "full_text", "abstract_only", "truncated_full_text",
                               # "compacted_full_text", "search_batch", "webpage",
                               # "web_search", "citations"
    timestamp: float = 0.0     # time.time() when fed
    cleared: bool = False      # soft-delete; excluded from future context
    summary: str = ""          # user-requested summary replacement
    compacted: bool = False           # whether this was compacted
    originating_question: str = ""    # user question that triggered this search


@dataclass
class ReplyBundle:
    text: str
    cited: List[Tuple[str, str]]       # (title, url)
    consulted: List[Tuple[str, str]]   # (title, url)
    source_details: Optional[List[RecordInfo]] = None
    tool_context_summary: Optional[str] = None


# ----------------------------
# Settings loader
# ----------------------------

def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base (overlay wins for leaf values)."""
    result = dict(base)
    for k, v in overlay.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


_settings_cache: Optional[Tuple[float, Dict[str, Any]]] = None  # (mtime, settings)
_settings_cache_path: Optional[Path] = None


def load_settings(force_reload: bool = False) -> Dict[str, Any]:
    """
    Load settings from JSON file, merged over defaults.
    Search order: ~/.config/chat_cli/settings.json -> ./settings.json -> defaults only.
    Caches result and invalidates when file mtime changes.
    """
    global _settings_cache, _settings_cache_path

    if _settings_cache is not None and not force_reload:
        cached_mtime, cached_settings = _settings_cache
        # Check if the file has been modified
        if _settings_cache_path is not None:
            try:
                current_mtime = _settings_cache_path.stat().st_mtime
                if current_mtime == cached_mtime:
                    return cached_settings
            except Exception:
                return cached_settings
        else:
            return cached_settings

    candidates = [
        Path.home() / ".config" / "scicli" / "settings.json",
        Path.home() / ".config" / "chat_cli" / "settings.json",  # backward compat
        Path("settings.json"),
    ]

    settings = dict(DEFAULT_SETTINGS)
    found_path: Optional[Path] = None
    found_mtime: float = 0.0
    for path in candidates:
        try:
            if path.exists():
                user = json.loads(path.read_text(encoding="utf-8"))
                settings = _deep_merge(settings, user)
                found_path = path
                found_mtime = path.stat().st_mtime
                break
        except Exception:
            pass

    _settings_cache = (found_mtime, settings)
    _settings_cache_path = found_path
    return settings


def get_model_specs() -> Dict[str, Dict[str, Any]]:
    """Build a flat MODEL_SPECS dict from settings."""
    settings = load_settings()
    specs: Dict[str, Dict[str, Any]] = {}
    for prov_name, prov_cfg in settings.get("providers", {}).items():
        for model_name, model_cfg in prov_cfg.get("models", {}).items():
            specs[model_name] = dict(model_cfg)
    return specs


def models_by_provider() -> Dict[str, List[str]]:
    """Group known model names by provider."""
    settings = load_settings()
    out: Dict[str, List[str]] = {}
    for prov_name, prov_cfg in settings.get("providers", {}).items():
        out[prov_name] = list(prov_cfg.get("models", {}).keys())
    return out


def get_provider_config(provider: str) -> Dict[str, Any]:
    """Get provider-specific config (api_key_env, base_url, models, compact_model, etc.)."""
    settings = load_settings()
    return settings.get("providers", {}).get(provider, {})


def get_api_key(provider: str) -> str:
    """Resolve API key from environment using provider config."""
    cfg = get_provider_config(provider)
    key_env = cfg.get("api_key_env", "")
    key = (os.environ.get(key_env) or "").strip() if key_env else ""
    if not key:
        alt_env = cfg.get("api_key_env_alt", "")
        if alt_env:
            key = (os.environ.get(alt_env) or "").strip()
    return key


def get_compact_model(provider: str) -> str:
    """Get the compact/summarization model for a provider."""
    cfg = get_provider_config(provider)
    return cfg.get("compact_model", "gpt-5-nano")


_THEME_HARD_DEFAULTS: Dict[str, str] = {
    "command": "cyan",
    "success": "green",
    "error": "red",
    "warning": "yellow",
    "info": "dim",
    "accent": "magenta",
    "heading": "bold",
    "tool_call": "dim cyan",
    "diff_add": "green",
    "diff_del": "red",
    "diff_hdr": "cyan",
    "code_theme": "monokai",
}

_theme_cache: Dict[str, Dict[str, str]] = {}

_THEMES_DIR = Path(__file__).parent / "themes"


def get_theme(theme_name: Optional[str] = None) -> Dict[str, str]:
    """Return merged color theme for the given theme name.

    Resolution order:
      1. Hard-coded defaults (fallback)
      2. themes/{theme_name}.theme.json  (built-in or user-placed)
      3. ~/.config/scicli/themes/{theme_name}.theme.json  (user override)
      4. settings.json ui.colors  (per-key overrides)
    """
    settings = load_settings()
    if theme_name is None:
        theme_name = settings.get("ui", {}).get("theme", "academic")

    if theme_name in _theme_cache:
        return _theme_cache[theme_name]

    result = dict(_THEME_HARD_DEFAULTS)

    # Load theme file (local themes/ dir, then user config dir)
    candidates = [
        _THEMES_DIR / f"{theme_name}.theme.json",
        Path.home() / ".config" / "scicli" / "themes" / f"{theme_name}.theme.json",
        Path.home() / ".config" / "chat_cli" / "themes" / f"{theme_name}.theme.json",  # backward compat
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                file_colors = json.loads(candidate.read_text())
                result.update({k: v for k, v in file_colors.items() if not k.startswith("_")})
            except Exception:
                pass
            break

    # Apply per-key overrides from settings.json ui.colors
    user_colors = settings.get("ui", {}).get("colors", {})
    result.update(user_colors)

    _theme_cache[theme_name] = result
    return result


# Backward-compatible aliases
SourceInfo = RecordInfo
SourceEntry = Record
