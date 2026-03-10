# SciCLI — LLM Interface for Reproducible Scientific Workflows

A terminal-based multi-provider LLM client designed for **reproducible, transparent scientific workflows** that involve literature search and structured synthesis of information from external sources.

---

## What This Is

SciCLI is **an interface for LLMs to engage with the scientific literature in a way that is transparent, reproducible, and extensible.**

The key design principles are:

- **Reproducibility**: every action — which papers were read, which tools were called, which citations were generated — is tracked with full metadata. Sessions can be saved and reloaded exactly as they were.
- **Transparency**: the model's search strategy, tool calls, and source inventory are all visible. Citations in answers are backed by real Semantic Scholar BibTeX entries, not hallucinated.
- **Dual control**: most actions can be initiated by **either the model or the user**. You can let the model search autonomously, or you can manually run `/scholar`, `/read`, `/cite`, and feed the results to the model yourself.
- **Controlled tool use**: the set of tools available to the model is explicit and configurable. You know exactly what external services the model can reach.
- **Domain extensibility**: new tools (databases, APIs, file formats) can be added as self-contained plugins without touching the core loop.

### Primary use cases

- Asking research questions that require real literature search and citation
- Systematic exploration of a topic across dozens of papers with a synthesised, cited answer
- Building reproducible question-answering pipelines over the scientific literature
- Comparing how different LLMs reason about the same evidence base

---

## Features at a Glance

| Feature | Details |
|---------|---------|
| **LLM providers** | OpenAI, DeepSeek, Kimi, Sakura Internet |
| **Literature search** | Semantic Scholar full-text retrieval with real BibTeX keys |
| **Web search** | Brave Search API with HTML→Markdown conversion |
| **Research pipeline** | Think → Search → Synthesize with auto-compaction |
| **Citations** | Numbered `[1]` or author-year `[Smith, 2024]`, verified against source inventory |
| **File ingestion** | PDF (PyMuPDF, OCR fallback), DOCX, text, Markdown, code files |
| **Session management** | Save/load conversations with full source metadata |
| **Extensible** | Plugin architecture for new tools and providers |

---

## Installation

### Requirements

Python 3.9 or later.

```bash
# Core (required)
pip install openai requests beautifulsoup4 rich prompt_toolkit

# Recommended
pip install pymupdf markdownify tiktoken

# Optional
pip install python-docx     # DOCX support
pip install pytesseract pillow  # OCR for scanned PDFs
```

### External API keys

| Service | Environment variable | Required for |
|---------|---------------------|--------------|
| OpenAI | `API_KEY_OPENAI` or `OPENAI_API_KEY` | OpenAI models |
| DeepSeek | `DEEPSEEK_API_KEY` | DeepSeek models |
| Kimi (Moonshot AI) | `KIMI_API_KEY` | Kimi models |
| Sakura Internet | `SAKURA_API_KEY` | Sakura models (format: `TOKEN:SECRET`) |
| Brave Search | `BRAVE_API_KEY` | `/web` command and agentic web search |
| Semantic Scholar | `SEMANTIC_SCHOLAR_API_KEY` | Optional — increases S2 rate limits |

API keys can be set as environment variables or in `settings.json` (see [Configuration](#configuration)).

### Clone and run

```bash
git clone <https://github.com/gbrandani/SciCLI> scicli
cd scicli
python scicli.py
```

### Run from anywhere

Data directories (`conversations/`, `uploads/`, `.llmcli_stats.json`) are always stored inside the `scicli/` repo directory regardless of where you invoke the script. This means you can call it from any working directory:

```bash
# Direct path
python /path/to/scicli/scicli.py

# Or add a symlink to your PATH (Linux/macOS)
chmod +x /path/to/scicli/scicli.py
ln -s /path/to/scicli/scicli.py ~/.local/bin/scicli
# Then just:
scicli
```

---

## Configuration

Configuration lives at `~/.config/scicli/settings.json` (user-global) or `./settings.json` in the current working directory (local project override, takes precedence).

A minimal configuration for DeepSeek:

```json
{
  "defaults": {
    "provider": "deepseek",
    "model": "deepseek-chat"
  }
}
```

A full configuration showing all options:

```json
{
  "defaults": {
    "provider": "deepseek",
    "model": "deepseek-chat"
  },
  "providers": {
    "openai": {
      "api_key_env": "API_KEY_OPENAI",
      "compact_model": "gpt-5-nano",
      "models": {
        "gpt-5.2":     { "context": 400000, "max_output": 128000 },
        "gpt-5-mini":  { "context": 400000, "max_output": 128000 },
        "gpt-5-nano":  { "context": 400000, "max_output": 128000 }
      }
    },
    "deepseek": {
      "api_key_env": "DEEPSEEK_API_KEY",
      "base_url": "https://api.deepseek.com",
      "compact_model": "deepseek-chat",
      "models": {
        "deepseek-chat":     { "context": 128000, "max_output": 8000 },
        "deepseek-reasoner": { "context": 128000, "max_output": 32000 }
      }
    },
    "kimi": {
      "api_key_env": "KIMI_API_KEY",
      "base_url": "https://api.moonshot.ai/v1",
      "compact_model": "kimi-k2.5",
      "models": {
        "kimi-k2.5": { "context": 256000, "max_output": 16000 }
      }
    },
    "sakura": {
      "api_key_env": "SAKURA_API_KEY",
      "base_url": "https://api.ai.sakura.ad.jp/v1",
      "models": {
        "gpt-oss-120b": { "context": 32000, "max_output": 8000 }
      }
    }
  },
  "prompts": {
    "system_preamble": "You are an expert in structural biology.",
    "search_strategy_override": "",
    "synthesis_instructions": ""
  },
  "ui": {
    "codecolor": false,
    "auto_ocr": true
  }
}
```

#### Prompt overrides

Three sections of the system/force-answer prompt can be replaced per-project:

- **`system_preamble`** — prepended to the system prompt; use for domain expertise context.
- **`search_strategy_override`** — replaces the built-in SEARCH STRATEGY section entirely.
- **`synthesis_instructions`** — replaces the SYNTHESIS INSTRUCTIONS in the force-answer prompt.

Run `/prompts` to inspect active overrides, or `/prompts dump` to write the full prompt to `prompts_dump.json`.

---

## Quick Start

```
$ python scicli.py

> What are the main mechanisms of CRISPR-Cas9 off-target cleavage?
  [model searches literature autonomously, then synthesises a cited answer]

> /depth deep
  [switch to thorough search: 8-12 papers, think phase enabled]

> /provider deepseek
> /model deepseek-chat
  [switch provider mid-session]

> /scholar CRISPR off-target mechanisms
> /sread 1 3 5
> How do the papers above compare in their conclusions?
  [manual paper selection, then model synthesises]
```

Press **Alt+Enter** (or **Esc then Enter**) for multiline input. Regular Enter submits.

---

## Research Pipeline

When the research pipeline is enabled (default: on), every question goes through up to three phases:

### Phase 1 — Think (deep search only)

A separate API call with no tools. The model writes:
1. An initial assessment from its training knowledge
2. An explicit search plan (queries to run, paper types to prioritise, known gaps)

If the model determines no search is needed (`SEARCH: NONE`), it answers directly.

### Phase 2 — Search (agentic tool loop)

The model autonomously calls tools to gather evidence:

| Tool | What it does |
|------|-------------|
| `search(query, count)` | Brave Search; returns title/URL/snippet per result for both academic and general web |
| `read(url)` | Smart reader: academic URLs → Semantic Scholar full text + BibTeX; general URLs → HTML→Markdown |
| `get_paper_references(paper_id, direction, limit)` | Fetches citing or referenced papers from Semantic Scholar citation graph |
| `reread(ref_key)` | Re-reads a paper already accessed this session (from local PDF, S2, or original URL) |

The loop runs up to 8 iterations (shallow) or 18 iterations (deep). At set checkpoints, the CLI injects guidance messages prompting the model to read papers, traverse citation trails, and fill gaps.

Duplicate detection prevents re-reading the same paper (>85% title similarity threshold).

### Phase 3 — Synthesise

At a configured iteration (`force_answer_at`: 5 for shallow, 14 for deep), all search tools are removed and the model receives a **force-answer prompt** containing:

- A **SOURCE INVENTORY** listing every paper read (full text or abstract) and every web page fetched, with BibTeX keys
- The think-phase draft (deep mode only)
- Synthesis instructions (structure, cross-referencing, critical evaluation)
- Citation instructions: use `[@BibTeXKey]` inline; the system builds the reference list automatically

The model writes its answer using `[@BibTeXKey]` inline citations. Post-processing replaces these with numbered `[1]`, `[2]` or author-year labels, and appends a formatted reference list with full metadata and access level notes.

### Search depth presets

```
/depth shallow   # default: 5 force-answer, 8 max iterations, no think phase
/depth deep      # 14 force-answer, 18 max iterations, think phase enabled
```

Fine-grained control:

```
/targets papers=10 searches=6 force=12 max=20 think=1
```

---

## Citation System

Citations are backed by real Semantic Scholar metadata and are only rendered for sources the model actually accessed in the session.

**BibTeX key format**: `AuthorYearWord` — first author surname + year + first word of title (lowercased, alphanumeric only). Example: `Zhang2024explicit`, `Moller2019the`.

**Display styles** (toggle with `/citestyle`):
- `numbered` (default): `[1]`, `[2]`, …
- `authoryear`: `[Moller, ACS Cent. Sci., 2019]`

**Reference list** format:
```
[1] Xinqiang Ding, Xingcheng Lin, Bin Zhang. (2021). "Stability and folding
    pathways of tetra-nucleosome from six-dimensional free energy surface".
    Nature Communications. https://... (full text)
[2] Joshua Moller et al. (2019). "The Free Energy Landscape of Internucleosome
    Interactions...". ACS Central Science. https://... (abstract only)
```

Access level notes: `(full text)`, `(abstract only)`, `(webpage)`. Only `full_text` and `webpage` sources are included in the rendered reference list; `abstract_only` and `snippet` sources are shown in the SOURCE INVENTORY as non-citable.

---

## Manual Literature Search

You can run any part of the pipeline yourself without asking a question:

```
/scholar CRISPR off-target                # search Semantic Scholar, display results
/scholar_more 2                           # fetch next 2 pages of results
/sread 1 3-5                              # read full text of results 1, 3, 4, 5
/sdownload 2                              # download open-access PDF of result 2
/sbib 1-5                                 # generate BibTeX for results 1–5
/cite abc123def456                        # fetch citation graph for a paper ID
/query off-target CRISPR efficiency       # silent search, stash results for later
```

Then ask the model to synthesise from the sources you selected.

---

## Web Search

```
/web mitochondrial fission mechanisms     # Brave Search, stores results
/read 1 3                                 # read web result 1 and 3 into context
```

During autonomous search the model can also call `search()` and `read()` for web URLs directly.

Web page content is always wrapped in `[UNTRUSTED_CONTENT_START]` / `[UNTRUSTED_CONTENT_END]` markers, and the model is instructed to treat it purely as data (prompt injection prevention).

---

## File Ingestion

```
/read paper.pdf                           # read PDF (PyMuPDF, OCR fallback)
/read report.docx                         # read Word document
/read notes.md                            # read any text file
/read https://example.com/paper.html      # fetch and read URL
/read *.py                                # glob: read all Python files
/reread                                   # re-read all modified tracked files
/reread ID                                # re-read source with index ID
```

Supported formats: PDF, DOCX, plain text, Markdown, code files, HTML URLs.

---

## Providers and Models

### OpenAI

Full agentic tool support. Recommended for most research tasks.

```
/provider openai
/model gpt-5.2
/model gpt-5-mini        # faster, lower cost
```

### DeepSeek

Full agentic tool support on `deepseek-chat`. `deepseek-reasoner` is supported but without tools (long-form reasoning mode).

```
/provider deepseek
/model deepseek-chat       # tools supported
/model deepseek-reasoner   # no tools, extended reasoning
```

### Kimi (Moonshot AI)

Full agentic tool support. Uses only custom tools (Kimi's built-in web search is intentionally disabled for consistent, comparable behaviour across providers).

```
/provider kimi
/model kimi-k2.5
```

### Sakura Internet

Full agentic tool support. 3,000 requests/month free tier; the CLI tracks usage and blocks calls once the limit is reached.

```
/provider sakura
/model gpt-oss-120b
```

### Switching providers mid-session

```
/provider deepseek         # switch provider
/model deepseek-chat       # switch model
```

All source records, conversation history, and session state are preserved across provider switches.

---

## Context Management

### Automatic compaction

**During the agentic research loop**: if the accumulated context reaches 72% of the model's context window, `force_answer_at` is advanced to the next iteration, triggering early synthesis with the sources collected so far.

**During interactive conversation**: when the conversation history exceeds 85% of the context window, the CLI calls a compact model to summarise the conversation into 200–400 words while preserving all source ref_keys (so you can still cite prior papers after compaction).

### Manual compaction

```
/compact                           # compact current conversation
/compact focus on protein folding  # compact with topic bias
```

### Conversation save/load

```
/save my_research.json             # save full session including source metadata
/load my_research.json             # restore session exactly
/save 3 snippet.py                 # save code snippet from last reply
```

Saved sessions include all source records, BibTeX keys, conversation history, and settings — enabling exact reproduction of a research workflow.

---

## Session Control

```
/info                    # show current settings, API usage, source count
/clear                   # clear all messages and reset source registry
/sources                 # list all sources accessed this session
/sources 1 3 5           # show details for sources 1, 3, 5
/sources dump 1-5        # dump source content to terminal
/sources clear           # remove all sources
```

---

## Display and Output

```
/citestyle numbered       # [1], [2], ...  (default)
/citestyle authoryear     # [Smith, Nature, 2024]
/codecolor on             # syntax-highlighted code blocks
/copy                     # copy last reply to clipboard
/copy 3                   # copy code snippet 3 to clipboard
/run pytest tests/        # run shell command, feed output to model
```

---

## Pipeline and Tool Toggles

```
/pipeline on|off          # enable/disable 3-phase research pipeline
/tools on|off             # enable/disable all agentic tools
/effort auto|low|medium|high|xhigh   # reasoning effort (where supported)
/autocite on|off          # auto-fetch citation graph after full-text reads
/trunclimit 40000         # max chars per tool result (default: 40,000)
/trunclimit none          # disable truncation
```

---

## Source Tracking Architecture

Every external resource accessed — whether by the model autonomously or by the user manually — is recorded in the session's **source registry** with full metadata:

- Source type: `full_text`, `abstract_only`, `webpage`, `snippet`
- BibTeX key (papers), URL (web pages)
- Character count, truncation flag, timestamp
- Which question originated the access

This registry drives the SOURCE INVENTORY shown to the model during synthesis, ensuring citations refer only to sources that were actually consulted.

---

## Architecture Overview

```
scicli.py          Main UI, command dispatcher, input handling
config.py          Settings loader, SessionState, RecordInfo/PaperRecord, DEPTH_CONFIG
providers.py       Provider classes (OpenAI/DeepSeek/Kimi/Sakura), all wrap run_research_pipeline()
agentic.py         Think phase, agentic tool loop, force-answer prompt, context limit handling
tools.py           Tool schemas + implementations + plugin registration; AGENTIC_TOOLS defined here
tool_registry.py   ToolPlugin dataclass, ToolRegistry singleton, argument validation
compaction.py      Conversation-level model-based summarisation (interactive mode)
records.py         Citation rendering, BibTeX key mapping, reference list generation
utils.py           File I/O, web fetching, S2/Brave API wrappers, BibTeX utilities
```

### Adding a new tool

Each tool is a self-contained `ToolPlugin`:

```python
from tool_registry import REGISTRY, ToolPlugin

REGISTRY.register(ToolPlugin(
    name="my_tool",
    schema={ ... },                     # OpenAI function-calling schema
    execute=_tool_my_tool,              # (args, state) -> {"result": ..., "source_infos": [...]}
    register_record=_register_my_tool, # updates state.records
    inventory_category="web",          # where it appears in SOURCE INVENTORY
    inventory_formatter=lambda s: f"[{s.ref_key}] {s.title}",
))
```

The tool is immediately available to the model and to the user via command.

---

## Testing

```bash
python tests/test_cli.py                          # unit and integration tests
python tests/test_citation_debug.py              # full citation pipeline diagnostic (uses real API)
python tests/test_citation_debug.py --kimi       # same with Kimi provider
python tests/test_citation_debug.py --depth deep # deep search diagnostic
```

The citation debug script traces the entire pipeline — think phase, each tool call with args and result, the SOURCE INVENTORY shown to the model, and the `apply_references` key-matching analysis — making it easy to diagnose citation issues.

---

## Known Limitations

- **Paper access**: full-text retrieval depends on open-access availability. Many papers return abstract only. Access to paywalled PDFs requires institutional access or a configured PDF proxy.
- **Rate limits**: Semantic Scholar has a public rate limit (~1 req/s without API key, ~10 req/s with). Heavy deep searches may slow down due to S2 throttling.
- **Context limits**: very long papers may be truncated (default: 40,000 chars per tool result). Increase with `/trunclimit`.
- **Sakura**: 3,000 requests/month free tier. The CLI blocks further calls once the monthly limit is reached.

---

## License

See `LICENSE` file.
