# SciCLI Documentation

## How Agentic Search Works

This section explains the full lifecycle of a research question, from input to final cited answer. The system uses OpenAI-compatible function calling (tool use) to let the model autonomously search, read, and cite literature.

### Two modes of operation

1. **Manual mode**: You search with `/scholar`, browse results, read papers with `/sread`, and ask the model questions about what you've read. The model sees the ingested text as conversation messages but does not call any tools itself.

2. **Agentic mode**: You ask a research question directly (e.g. "What is the mechanism of chromatin remodellers?"). The model autonomously calls tools — searching Semantic Scholar, reading papers, browsing the web — and then synthesizes an answer with citations. This is the default when tools are enabled (`/tools on`).

The rest of this section describes agentic mode in detail.

### What the model always sees

On every API call, the model receives:

1. **The system prompt** — built by `build_system_prompt()`. This contains:
   - The current date (for time awareness)
   - **SEARCH STRATEGY** instructions: how to use tools, when to search papers vs web, how many queries and papers to target
   - **KEYWORD STRATEGY** guidance: vary terminology, use synonyms, try broader/narrower terms
   - **SEARCH DEPTH OVERRIDE**: depth-specific instruction (e.g. "Quick search: 1-2 queries, read 2-3 papers" for shallow)
   - **TARGET** numbers: if `/targets papers=8` is set, this appears explicitly
   - **CONTENT SAFETY** warning about untrusted web content
   - **RESPONSE FORMAT** instructions: how to cite, how to structure answers

2. **The conversation messages** — your previous questions and the model's previous answers.

3. **The tool definitions** — JSON schemas for `search_papers`, `read_paper`, `web_search`, `read_webpage`, and `get_paper_references`, passed as the `tools` parameter of the API call. The model sees the name, description, and parameter schema of each tool.

### The research pipeline: Think → Search → Synthesize

When the pipeline is enabled (`/pipeline on`, which is the default), a research question goes through three phases. The pipeline can be toggled off, in which case only the Search and Synthesize phases run.

#### Phase 1: Think (deep mode only)

In deep mode (`/depth deep`), before any search happens, the CLI makes a **separate, isolated API call** to the model with a special prompt. This call has no tools — the model cannot search during this phase. The prompt asks the model to:

1. Write a brief **initial assessment** from its training knowledge, being honest about uncertainties.
2. Propose a **search plan**: 2-3 specific queries, what types of papers to prioritize, what aspects need investigation.

If the model responds with "SEARCH: NONE" (meaning it can answer confidently without literature), the pipeline stops and returns that answer directly.

The CLI displays this draft in dim styling (labeled "Initial assessment:") so you can see the model's starting point. The dim rendering is done by the CLI code — the model itself doesn't know its output will look different from the final answer.

In shallow mode, this phase is skipped entirely.

#### Phase 2: Search (agentic tool loop)

The CLI starts the main agentic loop. If a think draft was produced, it is injected back into the conversation as a system message saying: *"RESEARCH CONTEXT — Your initial assessment and search plan: [draft]. Now execute your search plan. Use your tools to find evidence."*

The loop then proceeds as follows:

```
for each iteration (up to max_iterations):
    1. Send all accumulated messages + tool definitions to the model via the API
    2. If the model returns text (no tool calls) → done, return the answer
    3. If the model returns tool calls → execute each one:
       - search_papers → calls Semantic Scholar API, returns JSON with titles, abstracts, citation counts
       - read_paper → fetches full text (tries HTML, then PDF, then arXiv, falls back to abstract)
       - web_search → calls Brave Search API, returns titles/URLs/snippets
       - read_webpage → fetches URL, converts HTML to Markdown
       - get_paper_references → fetches citation graph from Semantic Scholar
    4. Append tool results as 'tool' role messages
    5. Loop back to step 1
```

**How paper selection works:** After a `search_papers` call, the model receives the full result JSON — every paper's title, authors, year, abstract, citation count, and open-access status. On its next API call, the model sees all of this and decides which papers to read by calling `read_paper` with specific `paper_id` values. The model can call multiple `read_paper` in parallel (e.g. 5 at once in a single response). The CLI does not select papers — it is entirely the model's autonomous decision, guided only by the system prompt's soft instructions ("prioritize reviews first, then highly cited empirical studies").

**Why the model sometimes skips reading papers:** The model may judge that the abstracts are sufficient, or it may misjudge which papers are relevant. It may also hit an iteration limit before getting to the reading phase. The deep mode checkpoints try to mitigate this with nudges injected at specific iterations:
- Iteration 3: *"You've done initial searches. Now read the most promising papers (5-8 in parallel)."*
- Iteration 7: *"Use get_paper_references on your best finds to explore citation trails."*
- Iteration 10: *"Identify gaps in your coverage. Do targeted follow-up searches."*

These checkpoints are system messages added to the conversation — the model sees them as instructions.

**Auto citation graph:** When a paper is read at full text and `/autocite on` is set (the default), the CLI automatically fetches the citation graph (both citing and referenced papers) from Semantic Scholar and injects a ranked list into the conversation as a system message. The model can then use `read_paper` on promising items from this list.

**Duplicate detection:** Before reading a paper, the tool checks if a paper with a very similar title (>85% word overlap) has already been read at full text. If so, it returns a "duplicate, already read" message instead of re-fetching.

#### Phase 3: Synthesize (force-answer)

At a configurable iteration (default: iteration 5 for shallow, 14 for deep), the CLI forces the model to stop searching and write its final answer. It does this by injecting a **force-answer prompt** as a user message. This prompt contains:

1. **SOURCE INVENTORY**: A structured list of everything the model has accessed, organized as:
   - Papers read (full text) — with `[BibTeXKey]`, authors, year, title, paper ID
   - Papers consulted (abstract only) — same format
   - Web pages read — with `[BibTeXKey]`, title, URL
   - Count of additional search results seen but not read

2. **The think draft** (if Phase 1 produced one): *"Compare your initial assessment with what you found in the literature."*

3. **SYNTHESIS INSTRUCTIONS**: Write like a scientist. Organize by theme, not by paper. Cross-reference multiple sources. Distinguish consensus from emerging evidence. Note methodological limitations.

4. **CITATION INSTRUCTIONS**: Call the `cite` tool with BibTeX keys, then use `[BibTeXKey]` inline.

During synthesis, the **only tool available** is `cite`. The search tools are removed. The model calls `cite(keys=["Smith2024explicit", "Lin2023chromatin"])` to register which sources it references, then writes its answer using `[BibTeXKey]` inline.

#### Post-processing: Citation rendering

After the model's answer is complete, the CLI processes it:

1. **Verified references**: The `cite` tool calls recorded which BibTeX keys the model explicitly registered. The CLI replaces each `[BibTeXKey]` in the text with a display format: either numbered (`[1]`, `[2]`, ...) or author-year (`[Smith, Nature, 2024]`), depending on `/citestyle`.

2. **Reference list**: The CLI strips any reference section the model may have generated and builds a canonical one from the verified citations, with full metadata (authors, year, title, venue, URL, access level).

3. **Source display**: The CLI shows a categorized source list: "Papers read (full text)", "Papers consulted (abstract only)", "Web pages read", and a count of search results.

### Context management during the loop

The agentic loop can accumulate large amounts of text (full papers, search results, citation graphs). The CLI monitors context usage and compacts when necessary:

**Auto-compaction during the loop** (triggered when context exceeds 70% of the model's limit):
- **Tier 1**: Re-ranks search batch results using TF-IDF against the user's query and search history. Keeps only the top 10 most relevant results per batch (no model call needed).
- **Tier 2**: Summarizes full paper texts using a smaller "compact model" (e.g. `gpt-5-nano`), oldest papers first. The summary replaces the full text in the conversation, so subsequent model calls see the summary instead.
- **Emergency**: If still over 85% after compaction, forces early synthesis on the next iteration.

**Manual compaction** (`/compact [guidance]`): Same multi-tier strategy applied to the main conversation. Optionally biased toward preserving content related to a specific topic.

### Manual vs. agentic source tracking

Sources are tracked in the **source registry** (`SessionState.source_registry`), which is a list of `SourceEntry` objects. Sources are registered by:
- **Agentic tools**: Each `read_paper`, `read_webpage`, `search_papers`, `web_search`, and `get_paper_references` call automatically registers its results.
- **Manual commands**: `/sread` registers papers with full S2 metadata. `/read` registers URLs (and attempts S2 metadata lookup for academic URLs). `/query` and `/cite` register their results.

The source registry is what powers `/sources` (list/clear/dump) and `/info` (source counts). It is separate from the conversation messages — clearing a source from the registry also removes its corresponding messages from the conversation.

### Safety: untrusted content

Content from external web pages is wrapped with `[UNTRUSTED_CONTENT_START]` and `[UNTRUSTED_CONTENT_END]` markers. The system prompt instructs the model to treat content within these markers as data only, ignoring any embedded instructions. This applies to:
- Agentic `read_webpage` tool results
- `/read` of URLs and web search result indices

It does **not** apply to: local files (trusted), Semantic Scholar API results (trusted academic API).

### Configurable prompts

The system prompt can be partially overridden via `settings.json`:
- `system_preamble`: Prepended to the system prompt (e.g. domain expertise instructions)
- `search_strategy_override`: Replaces the entire SEARCH STRATEGY section
- `synthesis_instructions`: Replaces the SYNTHESIS INSTRUCTIONS in the force-answer prompt

Use `/prompts` to inspect active overrides and `/prompts dump` to export the full built system prompt.

---

## Literature & Web Search

### /web <query>

Brave web search (stores results for /read).

**Example:** `/web "CRISPR plants"`

Searches the web using the Brave Search API. Results are displayed in a table and stored so you can use /read <indices> to fetch and ingest specific pages.

### /scholar <query>

Semantic Scholar search.

**Example:** `/scholar "nucleosome"`

Searches Semantic Scholar for academic papers. Results include title, authors, year, citation count, open-access status, and PDF availability. Results are stored for /sread, /sdownload, and /sbib.

### /scholar_more [n]

Fetch next page of last /scholar query.

**Example:** `/scholar_more 20`

Fetches the next page of results from the last /scholar query. Optionally specify how many results to fetch (default: current page size). Results are appended to the existing result set, so /sread indices continue from where the previous page left off.

### /sread <idxs>

Read S2 paper full text (HTML/PDF).

**Example:** `/sread 1,3,5`

Reads full text of papers from the last /scholar results. Tries the S2 URL (HTML), then open-access PDF, then arXiv. Falls back to abstract if nothing else works. Registers as a proper source with metadata.

### /sdownload <idxs>

Download open-access PDFs.

**Example:** `/sdownload 1-5`

Downloads open-access PDFs from /scholar results to the uploads/ directory. Only works for papers that have an open-access PDF URL in Semantic Scholar. Accepts comma-separated indices or ranges (e.g. 1-5). Files are saved with sanitized filenames derived from the paper title.

### /sbib <idxs>

Generate BibTeX for S2 results.

**Example:** `/sbib 1-3`

Generates BibTeX entries for selected /scholar results and saves to a .bib file. BibTeX keys are auto-generated from first author surname + year + first content word of the title. Accepts comma-separated indices or ranges (e.g. 1-3).

### /query <text>

Search S2 and stash results (no model response).

**Example:** `/query "chromatin remodeling"`

Searches Semantic Scholar directly (non-agentic) and stashes results as a system message without triggering a model response. Results are also available for /sread.

### /cite <paper_id>

Fetch citation graph for a paper.

**Example:** `/cite abc123`

Fetches both citing and referenced papers for the given S2 paper ID and stashes the ranked results as a system message.

### /ncbi <query>

Search NCBI Entrez databases.

**Example:** `/ncbi TP53[GENE] AND human[ORGN]`

Search NCBI protein/nucleotide/gene databases using Entrez query syntax.
Field tags: [ORGN], [GENE], [PROT], [ACCN]. Boolean: AND, OR, NOT.
Results are displayed in a table and stored for /seqread.

### /uniprot <query>

Search UniProt protein database.

**Example:** `/uniprot gene_exact:TP53 AND organism_id:9606`

Search UniProt using its query syntax.
Fields: gene_exact:, organism_id:, ec:, keyword:, family:.
Results are displayed in a table and stored for /seqread.

## Source Management

### /sources [clear <spec>|dump <spec> [file]|<ids>]

Inspect/manage source registry.

**Example:** `/sources clear 1-3`

With no args: list all registered sources with metadata.

/sources <ids>: show detail for specific source IDs.
/sources clear <ids|all>: hard-delete sources and remove their messages from the conversation.
/sources dump <ids> [file]: dump source metadata to stdout or file.

### /seqread <accession>

Fetch a sequence by accession.

**Example:** `/seqread P0DTC2`

Fetch a sequence record from NCBI or UniProt by accession/ID.
Auto-detects database from accession format.
The sequence and annotations are ingested into the conversation.

## File Operations

### /read <path|url|idxs>

Read file, URL, or /web result indices into conversation.

**Example:** `/read paper.pdf`

The /read command ingests content from three source types:

1. Local files: Supports PDF (via PyMuPDF with OCR fallback), DOCX, and plain text formats (.txt, .md, .py, .json, .csv, etc.). PDFs are extracted using PyMuPDF; if the result looks like a scan (< 2000 chars), automatic OCR is attempted using pytesseract.

2. URLs: Fetches the page, detects PDF vs HTML. HTML pages are converted to Markdown using markdownify. Content is wrapped with safety markers since web content is untrusted.

3. Web result indices (e.g. /read 1-3): Reads pages from the last /web search results.

Glob patterns (e.g. /read *.py) are expanded and each file is read.

URL content is truncated to 12,000 characters per page; local files are truncated to 80,000 characters. Content is sent to the model with instructions to confirm reading and list key points. Local files are tracked in the file registry (see /files). Papers are registered as sources when metadata can be determined.

### /formats

Show supported file formats.

**Example:** `/formats`

Displays which file formats are supported and which optional libraries are installed. Shows status for: PyMuPDF (PDF extraction), PyPDF2 (fallback PDF), python-docx (DOCX), pytesseract + Pillow (OCR), markdownify (HTML→Markdown), and tiktoken (token counting).

### /files

List tracked local files with modification status.

**Example:** `/files`

Shows a table of all local files read during this session. Columns include file ID, name, character count, and status (ok / MODIFIED / MISSING). Modified status is detected by comparing current mtime to stored mtime.

### /reread [id]

Re-read modified tracked files.

**Example:** `/reread 2`

/reread with no args re-reads all tracked files whose mtime has changed.
/reread <id> re-reads a specific tracked file by its ID.

### /run [cmd]

Run shell command and feed output to model.

**Example:** `/run ls -la`

/run <cmd> runs a shell command and feeds its output to the model for analysis.
/run with no args feeds the last !cmd output to the model.

## Configuration

### /provider <name>

Switch LLM provider.

**Example:** `/provider deepseek`

Switch to a different LLM provider. Available: openai, openai_responses, deepseek, kimi. Auto-switches model if current one doesn't belong to the new provider.

### /model <name>

Switch model.

**Example:** `/model gpt-5.2`

Switch to a different model within the current or any provider. If the model belongs to a different provider, the provider is auto-switched. Available models and their context/output limits are defined in settings.json or the built-in defaults (see /info for the current model's specs).

### /effort <level>

Set reasoning effort.

**Example:** `/effort high`

Set reasoning effort: auto|none|low|medium|high|xhigh. This controls how much computation the model spends on its response. Higher effort may produce more thorough answers but uses more tokens. 'auto' lets the model decide. Only supported by some providers (OpenAI reasoning models).

### /depth shallow|deep

Set search depth preset.

**Example:** `/depth deep`

Sets the search depth preset which controls iteration limits and search strategy. 'shallow' does 1-2 searches and reads 2-3 papers. 'deep' does 8-10 queries and reads 8-12 papers with citation trail following.

### /targets [key=val ...]

Override search parameters.

**Example:** `/targets papers=8 searches=8`

Fine-tune search loop parameters. Keys: papers (soft target), searches (soft target), force (hard: force answer at iteration N), max (hard: max iterations), think (on|off|default).

With no args, shows current values and defaults.

### /pipeline on|off

Toggle Think→Search→Synthesize pipeline.

**Example:** `/pipeline off`

Enables or disables the 3-phase research pipeline: 1) Think (draft from knowledge), 2) Search (agentic tool loop), 3) Synthesize (force-answer with inventory).

### /tools on|off

Toggle agentic tool use.

**Example:** `/tools off`

Enable or disable agentic tool calling. When on (default), the model can autonomously call search_papers, read_paper, web_search, read_webpage, and get_paper_references during the agentic loop. When off, the model answers from its training knowledge and any manually ingested sources only. The system prompt is adjusted accordingly to avoid hallucinated tool-call text.

### /oaiweb on|off

Toggle OpenAI built-in web search.

**Example:** `/oaiweb on`

Enable or disable OpenAI's native web search tool. Only available with the openai_responses provider (Responses API). When on, OpenAI's built-in web_search_preview tool is added to API calls, letting the model search the web natively. This is separate from Brave web search (/web) and the agentic web_search tool, which use the Brave API.

### /autocite on|off

Auto citation graph after full-text read.

**Example:** `/autocite off`

When on (default), every time the agentic loop reads a paper at full text, the CLI automatically fetches both citing papers and referenced papers from Semantic Scholar, ranks them by TF-IDF relevance to the search queries, and injects the top 30 into the conversation as a system message. The model can then use read_paper on promising items from this list. This enables citation trail following without the model needing to call get_paper_references explicitly.

### /trunclimit <N|none>

Set tool result truncation limit.

**Example:** `/trunclimit 60000`

Set the character limit for truncating agentic tool results (default: 40,000 characters). When a tool returns more text than this limit, it is truncated to stay within the model's context window. Use 'none' to disable truncation (useful for very long papers, but risks exceeding context). This applies to all agentic tool results: paper full texts, web pages, search results, and citation graphs.

### /prompts [dump]

Show/dump active prompt overrides.

**Example:** `/prompts dump`

/prompts shows active prompt overrides and system prompt length.
/prompts dump writes the current built system prompt to prompts_dump.json.

## Conversation Management

### /compact [guidance]

Compact conversation to save context.

**Example:** `/compact "focus on X"`

Multi-tier compaction: 1) Re-ranks search batches via TF-IDF, keeping top results. 2) Summarizes full-text references using the compact model. 3) If still large, summarizes older conversation messages. Optional guidance text biases what to preserve.

### /save [name.json | N [file]]

Save conversation or snippet.

**Example:** `/save 1 code.py`

/save or /save name.json saves the full conversation.
/save 0 [file] saves the last full assistant reply.
/save N [file] saves code snippet #N. Shows a diff and asks confirmation when overwriting an existing file.

### /load <name.json>

Load conversation.

**Example:** `/load myresearch.json`

Load a previously saved conversation from the conversations/ directory. Restores all messages, source registry, file registry, and session state. The current conversation is replaced entirely.

### /clear

Clear conversation.

**Example:** `/clear`

Clears all messages, resets the system prompt, and clears the source and file registries. Starts a fresh conversation while keeping provider, model, and display settings unchanged.

### /quit

Exit (prompts to save).

**Example:** `/quit`

Exit the CLI. Prompts to save conversation if there are messages.

## Display & Output

### /help [group|command]

Show help (grouped overview, per-group, or per-command).

**Example:** `/help search`

/help with no arguments shows a grouped summary table of all commands.

/help <group> shows detailed documentation for every command in that group (e.g. /help search, /help files).

/help <command> shows detailed help for a specific command (e.g. /help /read).

### /info

Show current settings, stats, and session state.

**Example:** `/info`

Displays provider, model, context usage, source registry summary, search depth configuration, display settings, and API usage counters.

### /citestyle numbered|authoryear

Toggle citation display style.

**Example:** `/citestyle authoryear`

Switch between [1] numbered and [Author, Venue, Year] citation styles. This controls how inline citations appear in the model's final synthesized answer. The underlying BibTeX keys are the same either way; only the display format changes. The reference list at the end of the answer is also formatted accordingly.

### /copy [N]

Copy last reply or snippet to clipboard.

**Example:** `/copy 1`

/copy copies the full last assistant reply to the clipboard.
/copy N copies code snippet #N to the clipboard.
Uses pbcopy (macOS) or xclip (Linux).

### /codecolor on|off

Toggle code highlighting.

**Example:** `/codecolor on`

Enable or disable syntax-highlighted code blocks. When on, code blocks are rendered with Rich syntax highlighting (colors, bold) for readability. The plain-text version is always available for copy-paste. When off, only the plain-text version is shown.

### /autoocr on|off

Toggle automatic OCR for scanned PDFs.

**Example:** `/autoocr off`

When on (default), PDFs that appear to be scans (< 2000 chars extracted by PyMuPDF) automatically trigger OCR using pytesseract. Each page is rendered as an image and OCR'd. Requires pytesseract and Pillow to be installed. When off, scanned PDFs return whatever minimal text PyMuPDF can extract.

### /docs [question]

Ask questions about the CLI documentation.

**Example:** `/docs how does compaction work?`

/docs (first time) generates documentation.md and initializes a docs conversation.
/docs (subsequent) shows docs conversation status.
/docs <question> sends a question to the docs conversation.
The docs conversation is separate from the main conversation — no enter/exit needed.
