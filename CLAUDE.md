# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SciCLI** is a terminal-based multi-provider LLM client for scientific research workflows. It runs a three-phase research pipeline (think → agentic search → synthesize) and tracks all sources with full citation metadata for reproducibility.

## Running the Application

```bash
python scicli.py
```

## Running Tests

```bash
python tests/test_cli.py                          # Main test suite
python tests/test_citation_debug.py               # Full citation pipeline diagnostic (uses real API)
python tests/test_citation_debug.py --kimi        # Same with Kimi provider
python tests/test_citation_debug.py --depth deep  # Same with deep mode
python tests/test_search_quality.py               # Search quality evaluation
```

No build step, Makefile, or package manager. The project is a standalone set of Python scripts.

## Repository Layout

```
scicli/
├── scicli.py              # entry point
├── agentic.py, providers.py, tools.py, ...  # core modules (flat)
├── tests/                 # test scripts
├── themes/                # color theme JSON files
├── docs/                  # documentation.md
├── benchmark/             # evaluation datasets and LAB-Bench-LitQA2
├── conversations/         # runtime: saved sessions (git-ignored)
├── uploads/               # runtime: downloaded PDFs (git-ignored)
└── .llmcli_stats.json     # runtime: API usage counters
```

Data directories (`conversations/`, `uploads/`, `.llmcli_stats.json`, `.llmcli_history`) are anchored to the directory of `config.py` via `_APP_DIR = Path(__file__).resolve().parent`, so `scicli` can be invoked from any working directory and data always lands in the repo directory.

## Key Dependencies

Core: `openai`, `requests`, `beautifulsoup4`, `rich`, `prompt_toolkit`
Recommended: `pymupdf`, `markdownify`, `tiktoken`
Optional: `python-docx`, `pytesseract`, `pillow`

## Architecture

### Module Responsibilities

| File | Responsibility |
|------|---------------|
| `scicli.py` | Main UI loop, 40+ command dispatcher, `LLMChatClient` class |
| `config.py` | `SessionState`, `RecordInfo`, `PaperRecord`, config loading |
| `agentic.py` | Three-phase research pipeline, agentic tool loop |
| `providers.py` | Provider classes (OpenAI, DeepSeek, Kimi, Sakura) wrapping `agentic.py` |
| `tools.py` | Agentic tool schemas and implementations; `AGENTIC_TOOLS` list defined here explicitly as `[search, read, get_paper_references, reread]`; internal `_tool_*` helpers also registered in REGISTRY for inventory formatting but not exposed to the model |
| `tool_registry.py` | `ToolPlugin` dataclass, `ToolRegistry` singleton |
| `records.py` | Citation rendering, BibTeX key generation, reference list building |
| `utils.py` | API wrappers (Semantic Scholar, Brave), PDF/HTML/file I/O, token counting |
| `compaction.py` | Auto context compaction (TF-IDF → model summarization → emergency synthesis) |

### Research Pipeline (agentic.py)

1. **Think phase** (deep mode only): Separate API call, no tools. Model writes initial assessment and search plan.
2. **Search phase**: Agentic loop (up to 18 iterations). A `SEARCH GUIDANCE` system message is injected at loop start instructing the model to pass questions verbatim to `search()` and synthesize from snippets before calling `read()`. Checkpoints inject further guidance at iterations 3, 7, 10.
3. **Synthesize phase** (triggered at `force_answer_at` iteration): Tools removed; SOURCE INVENTORY with BibTeX keys is injected; model writes final answer with inline citations.

### Agentic Tools (`tools.py`)

Four tools exposed via `AGENTIC_TOOLS` (defined explicitly, not rebuilt from registry):
- **`search(query)`**: Brave-only. Returns title/URL/snippet/llm_contexts/is_academic per result. Count is always `state.brave_count` (user-configured, default 20). `_tool_search_papers`, `_tool_web_search` etc. are kept as internal helpers and remain registered in REGISTRY for inventory formatting but are NOT in AGENTIC_TOOLS.
- **`read(url)`**: Smart reader. Academic URLs → `_url_to_s2_identifier` (arXiv/DOI/PubMed) → S2 paper fetch (PDF + BibTeX) → title-lookup fallback → plain webpage fetch. Returns `paper_id` for citation graph traversal. Downloaded PDFs are stored locally and their path tracked in `PaperRecord.local_path`.
- **`get_paper_references(paper_id, direction, limit)`**: S2 citation graph traversal.
- **`reread(ref_key)`**: Re-reads a paper already accessed this session. Uses fallback chain: local PDF (`PaperRecord.local_path`) → S2 re-fetch by `external_id` → original URL. Enabled by `state.reread_registry` (a non-serialized `SessionState` attribute, dict keyed by `ref_key`), populated by `agentic.py` after every full-text/abstract access.

`_register_search` also populates `state.search_snap_registry` (runtime-only, not serialized): a list of `{query, record_id, snippets: [RecordInfo, ...]}` dicts, one per search call. This powers the `/snippets` command without creating individual `Record` entries in `state.records`.

Depth config in `config.py`: shallow = 5 force-answer/8 max; deep = 14 force-answer/18 max + think phase.

### Tool Plugin System (tool_registry.py)

Each tool is a `ToolPlugin` dataclass:
- `name`, `schema` (OpenAI function calling JSON), `execute(args, state)`, `register_record(...)`, `inventory_category`, `inventory_formatter`

`ToolRegistry` is a singleton. Tools in `tools.py` self-register at import time. New tools can be added without modifying core files.

### Source Tracking & Citations

Every source accessed during research is tracked as a `RecordInfo` (subclass: `PaperRecord`) with `ref_key` in BibTeX format `{FirstAuthorSurname}{Year}{FirstWordOfTitle}` (e.g., `Zhang2024explicit`).

Model output uses inline BibTeX keys: `[Zhang2024explicit]`. Post-processing in `records.py` replaces these with numbered `[1]` citations plus a reference list. The SOURCE INVENTORY injected at synthesize phase lists all available ref keys.

### Context Auto-Compaction

Two separate mechanisms:

**Interactive chat** (`compaction.py`, triggered at 85% of context via `COMPACT_AT_FRACTION`): `compact_conversation()` model-summarizes the full conversation history (200–400 words), then rebuilds the message list as `[system_msg, summary_assistant_msg, source_inventory_system_msg]`, preserving all ref keys so the model can still cite prior sources.

**Agentic pipeline** (`agentic.py:738`): At 72% of context characters used in the tool loop (hardcoded, no named constant), `force_answer_at` is advanced to the next iteration, triggering early synthesis with whatever sources have been collected so far. After this, the `reread_registry` is rebuilt from the now-deduplicated `all_records` to stay consistent with the SOURCE INVENTORY keys.

### Response Rendering

`render_assistant()` in `utils.py` renders model responses using Rich's `Markdown` class (bold, headings, syntax-highlighted code). This is the default (`codecolor=True`). `/codecolor off` switches to raw text mode (terminal copy-safe fallback). Code in formatted mode must be extracted with `/copy N` or `/save N`, which operate on the raw `last_assistant_text`.

### Color Themes

Themes live in `themes/*.theme.json`. Three built-ins: `academic` (default, monokai code), `nord` (github-dark code), `dracula` (dracula code). Switch with `/theme <name>` (session only) or `ui.theme` in settings.json (persistent). Custom themes can be placed in `themes/` or `~/.config/scicli/themes/`. `get_theme(name)` in `config.py` handles loading with hard-coded fallback. Theme keys: `command`, `success`, `error`, `warning`, `info`, `accent`, `heading`, `tool_call`, `diff_add`, `diff_del`, `diff_hdr`, `code_theme`.

### Configuration

Loaded from `~/.config/scicli/settings.json` (or `./settings.json`, which takes precedence). Key sections: `defaults`, `providers` (API keys), `prompts` (system preamble, synthesis instructions override), `ui` (`theme`, `codecolor`, `auto_ocr`, `colors` for per-key overrides).

### Supported Providers

| Provider | Class | Models | Notes |
|----------|-------|--------|-------|
| OpenAI | `OpenAIChatProvider` | gpt-5.2, gpt-5-mini, gpt-5-nano | Full tools |
| DeepSeek | `DeepSeekProvider` | deepseek-chat, deepseek-reasoner | Reasoner: no tools |
| Kimi (Moonshot) | `KimiProvider` | kimi-k2.5 | Full tools |
| Sakura Internet | `SakuraProvider` | gpt-oss-120b (default), Qwen3-Coder-480B-A35B-Instruct-FP8, llm-jp-3.1-8x13b-instruct4 | Full tools; 3,000 req/month free tier enforced by `SakuraUsageTracker` in `providers.py` |

Providers are `OpenAIChatProvider`, `DeepSeekProvider`, `KimiProvider`, `SakuraProvider` — all subclass `ProviderBase` and call `run_research_pipeline()`. `SakuraUsageTracker` (in `providers.py`) manages the monthly Sakura request counter and blocks calls once the limit is reached; it reads/writes `.llmcli_stats.json` under the `"sakura"` key.

### Usage Stats (`.llmcli_stats.json`)

All API usage is consolidated in `.llmcli_stats.json` (path from `config.STATS_FILE`). Structure: `{service: {"YYYY-MM": count, ...}, "last_updated_utc": "..."}` where service is one of `brave`, `sakura`, `openai`, `deepseek`, `kimi`.

- `increment_stat(service)` in `utils.py` atomically increments the current-month counter for any service.
- Brave is incremented in `brave_search()` in `utils.py`.
- LLM providers (non-Sakura) are incremented in `scicli.py` after each `provider.send()` call.
- `AppStats` in `config.py` wraps the counts dict with `monthly(service)` and `total(service)` helpers.
- `LLMChatClient._session_counts` (dict) tracks per-session counts separately (not persisted); displayed in `/info` via `_usage_info()`.

### External APIs

| Service | Purpose | Key |
|---------|---------|-----|
| Brave Search | Primary discovery for all queries (academic + general) | Required `BRAVE_API_KEY` |
| Semantic Scholar | Paper full-text + BibTeX metadata (via `read()` tool, not at search time) | Optional `SEMANTIC_SCHOLAR_API_KEY` |
| Sakura AI Engine | LLM inference (`https://api.ai.sakura.ad.jp/v1`) | Required `SAKURA_API_KEY` (format: `TOKEN:SECRET`) |
