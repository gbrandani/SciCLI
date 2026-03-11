"""
agentic.py — Agentic tool loop and research pipeline for Chat CLI.

Contains:
- Think phase (Phase 1)
- Research pipeline wrapper (Think → Search → Synthesize)
- Agentic tool loop with auto-compaction, citation graph, force-answer
"""

from __future__ import annotations

import json
import re
import time as _time
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import (
    SessionState, ReplyBundle, RecordInfo, Record,
    DEPTH_CONFIG, get_model_specs,
)
from utils import dedup_pairs, rank_papers_by_relevance, s2_citations, s2_references, _author_list
from tools import AGENTIC_TOOLS, execute_tool
from tool_registry import REGISTRY
from records import apply_references, build_record_inventory, get_ordered_sources, build_pinned_system_msg


def _clean_dsml_artifacts(text: str) -> str:
    """Remove DeepSeek's hallucinated tool-call XML tags from output."""
    text = re.sub(r"<｜DSML｜[^>]*>", "", text)
    text = re.sub(r"</｜DSML｜[^>]*>", "", text)
    return text.strip()


# ----------------------------
# Force-answer & think prompts
# ----------------------------

def _build_force_answer_prompt(
    source_details: List[RecordInfo],
    think_draft: Optional[str] = None,
) -> str:
    """Build the force-answer prompt with source inventory and constitutional synthesis instructions."""
    inventory = build_record_inventory(source_details)

    parts = [
        "IMPORTANT: Your search tools are now unavailable. Write your final answer NOW.",
        "",
    ]

    if think_draft:
        parts.extend([
            "YOUR INITIAL ASSESSMENT (from before searching):",
            think_draft,
            "",
            "Compare your initial assessment above with what you found in the literature.",
            "Note where your prior knowledge was confirmed, where it was wrong or outdated,",
            "and what new information you discovered.",
            "",
        ])

    if inventory:
        parts.append(inventory)
        parts.append("")

    parts.extend([
        "SYNTHESIS INSTRUCTIONS:",
        "You are a scientist writing a careful, evidence-based answer. Think about this the way",
        "a researcher would: not just collecting facts, but critically evaluating sources,",
        "recognizing limitations, and distinguishing strong evidence from preliminary findings.",
        "",
        "CITATION INSTRUCTIONS:",
        "- Cite sources using Pandoc citation format: [@key]",
        "  (This is the standard Pandoc/Quarto/R-Markdown inline citation syntax.)",
        "  The SOURCE INVENTORY lists each key as [Key]; cite it as [@Key] — add the @ prefix.",
        "  Example: \"This was shown by [@Zhang2024explicit] and confirmed by [@Lin2023nucleosome].\"",
        "  Multiple sources: [@Zhang2024explicit; @Lin2023nucleosome]",
        "- ONLY use keys listed in the SOURCE INVENTORY. Do NOT invent keys.",
        "- Papers marked '[abstract only — not citable]' cannot be cited.",
        "- Search snippets (shown with [snap*] keys) ARE citable but contain only excerpt text.",
        "  Prefer citing full-text papers when available; use snap* keys only for facts not in any paper.",
        "- If a source is not in the inventory, acknowledge the limitation; do not cite it.",
        "- The system auto-generates a reference list from your [@key] markers.",
        "  Do NOT write a References section yourself — it will be removed and replaced.",
        "",
        "WRITING STYLE:",
        "1. Structure: Overview → Key Findings → Synthesis → Open Questions.",
        "2. Do NOT just list findings paper by paper. Instead, organize by scientific theme.",
        "   For each major finding, explicitly state which sources agree, which disagree,",
        "   and how strong the evidence is. Example: 'This is supported by three independent",
        "   studies [@study1; @study2; @study3] using different force fields.'",
        "3. Cross-reference: when multiple sources address the same topic, compare them.",
        "   Agreements strengthen confidence; contradictions are informative.",
        "4. For abstract-only sources, note you had limited information.",
        "5. Distinguish: well-established consensus vs. emerging evidence vs. single-study findings.",
        "6. Discuss methodological considerations where relevant.",
        "7. End with what remains uncertain or needs more research.",
        "8. Do NOT offer follow-up options or ask what the user wants next.",
        "   End with your scientific conclusion.",
    ])

    return "\n".join(parts)


def _build_think_prompt(question: str, search_depth: str = "shallow") -> str:
    """Build the prompt for Phase 1 (Think) of the research pipeline."""
    depth_guidance = ""
    answer_strategy_guidance = ""
    if search_depth == "deep":
        depth_guidance = (
            "\nThis is a DEEP search. Plan a comprehensive investigation strategy with "
            "3-4 different query angles, aiming to read 8-12 papers including reviews."
        )
        answer_strategy_guidance = (
            "ANSWER_FROM_SNIPPETS: NO — Read the key papers in full before synthesizing. "
            "This mode is for thorough scientific analysis."
        )
    elif search_depth == "shallow":
        depth_guidance = (
            "\nThis is a QUICK search. Plan a focused strategy with 1 query and 2-3 key papers."
        )
        answer_strategy_guidance = (
            "ANSWER_FROM_SNIPPETS: Decide YES or NO.\n"
            "  YES — if the question is factual, definitional, or a quick reference lookup where "
            "search snippets alone will suffice. After searching, answer directly from snippets "
            "and end with: \"This answer is based on search snippets. Ask me to read the full papers for deeper analysis.\"\n"
            "  NO — if the question requires careful scientific analysis, comparing methods, or "
            "evaluating contested evidence. Read at least 2-3 papers in full."
        )

    return f"""You are a research scientist. Before searching the literature, think carefully about this question:

"{question}"

Do THREE things:

1. INITIAL ASSESSMENT: Write a brief answer based purely on your existing knowledge.
   Be honest about what you know well vs. what you're uncertain about.
   If you genuinely don't know, say so — that's valuable information.

2. SEARCH PLAN: Based on your knowledge (and gaps in it), propose a focused search strategy:
   - 2-3 specific search queries to use with search() (natural language questions or keyword phrases both work)
   - What types of papers to prioritize (reviews, recent empirical studies, etc.)
   - What aspects need the most investigation
   - Key URLs or papers to read in full using read(url)
{depth_guidance}

3. ANSWER STRATEGY:
   {answer_strategy_guidance}

If this question does NOT require literature search (e.g., it's a simple factual question
you can answer confidently, or it's not a research question at all), respond with:
SEARCH: NONE
followed by your direct answer.

Keep your response concise (200-400 words)."""


# ----------------------------
# Think phase
# ----------------------------

def _think_phase(
    client,
    messages: List[Dict[str, Any]],
    model: str,
    state: SessionState,
    on_status: Optional[Callable[[str], None]] = None,
    token_param: str = "max_tokens",
) -> Optional[str]:
    """
    Phase 1 of research pipeline: Ask model to draft an answer from knowledge
    and propose a search strategy. Returns the think text, or None if search
    is not needed (model says SEARCH: NONE).
    """
    question = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            question = m.get("content", "")
            break

    if not question:
        return None

    think_prompt = _build_think_prompt(question, state.search_depth)

    think_messages = [
        {"role": "system", "content": "You are a research scientist preparing to investigate a question."},
        {"role": "user", "content": think_prompt},
    ]

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": think_messages,
    }
    kwargs[token_param] = 2000

    if on_status:
        on_status("Thinking about the question...")

    try:
        resp = client.chat.completions.create(**kwargs)
        think_text = (resp.choices[0].message.content or "").strip()
    except Exception:
        return None

    if "SEARCH: NONE" in think_text.upper():
        return think_text

    if on_status:
        on_status("Search plan ready. Searching...")

    return think_text


# ----------------------------
# Research pipeline (Think → Search → Synthesize)
# ----------------------------

def run_research_pipeline(
    client,
    messages: List[Dict[str, Any]],
    model: str,
    state: SessionState,
    max_output_tokens: Optional[int],
    supports_tools: bool,
    on_tool_start=None,
    on_tool_result=None,
    on_status=None,
    on_think_draft=None,
    on_reasoning_content=None,
    post_process=None,
    extra_create_kwargs=None,
    override_tools=None,
    builtin_tool_handler=None,
    asst_msg_builder=None,
    token_param="max_tokens",
) -> ReplyBundle:
    """
    3-phase research pipeline:
    1. Think: Draft answer from knowledge + propose search plan
    2. Search: Agentic tool loop guided by think draft
    3. Synthesize: Force-answer with inventory + constitutional instructions
    """
    depth = getattr(state, "search_depth", "shallow")
    depth_cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["shallow"])
    if state.use_think_phase >= 0:
        use_think = bool(state.use_think_phase) and state.research_pipeline
    else:
        use_think = depth_cfg.get("use_think_phase", True) and state.research_pipeline

    think_draft = None
    if use_think and supports_tools:
        think_draft = _think_phase(
            client, messages, model, state,
            on_status=on_status, token_param=token_param,
        )

        if think_draft and "SEARCH: NONE" in think_draft.upper():
            parts = re.split(r'SEARCH:\s*NONE\s*', think_draft, flags=re.IGNORECASE)
            answer = parts[-1].strip() if len(parts) > 1 else think_draft
            if post_process:
                answer = post_process(answer)
            return ReplyBundle(text=answer, cited=[], consulted=[])

        if think_draft and on_think_draft:
            on_think_draft(think_draft)

        if think_draft:
            augmented_messages = list(messages)
            augmented_messages.append({
                "role": "system",
                "content": (
                    "RESEARCH CONTEXT — Your initial assessment and search plan:\n"
                    f"{think_draft}\n\n"
                    "Now execute your search plan. Use your tools to find evidence. "
                    "Pay special attention to areas where you expressed uncertainty."
                ),
            })
            messages = augmented_messages

    return run_agentic_loop(
        client=client,
        messages=messages,
        model=model,
        state=state,
        max_output_tokens=max_output_tokens,
        supports_tools=supports_tools,
        on_tool_start=on_tool_start,
        on_tool_result=on_tool_result,
        on_status=on_status,
        on_think_draft=on_think_draft,
        on_reasoning_content=on_reasoning_content,
        post_process=post_process,
        extra_create_kwargs=extra_create_kwargs,
        override_tools=override_tools,
        builtin_tool_handler=builtin_tool_handler,
        asst_msg_builder=asst_msg_builder,
        token_param=token_param,
        think_draft=think_draft,
    )


# ----------------------------
# Pre-loop search router
# ----------------------------

_ROUTER_PROMPT = """\
You are a routing assistant. Given a user question, decide whether web search is needed to answer it accurately.

Consider these criteria:
1. TIME-SENSITIVE: Does it ask about recent events, papers, news, current statistics, rankings, prices, or anything that changes over time? → Search needed
2. SPECIFIC VERIFICATION: Does it require current facts, URLs, or claims that could be outdated since early 2024? → Search needed
3. ESTABLISHED KNOWLEDGE: Is it about fundamental science (physics laws, equations, mechanisms), mathematics, well-known history, standard definitions, or concepts that are fixed and well-documented? → No search needed
4. NICHE/SPECIFIC: Is it highly specific (a particular paper, a niche technical detail, a specific gene or organism) where training-data coverage may be sparse? → Search probably needed

Question: {question}

Think step by step through these criteria. Then on a new line write exactly one of:
SEARCH: YES
SEARCH: NO"""


def _route_needs_search(
    client, model: str, question: str,
    on_status=None,
) -> tuple:
    """
    Pre-loop router: decide whether web search is needed for this question.
    Focuses on the NATURE of the question, not on model self-knowledge assessment.
    Returns (needs_search: bool, reasoning: str).
    Defaults to True (search) on any error.
    """
    if not (question or "").strip():
        return True, ""
    if on_status:
        on_status("Routing: deciding if search is needed…")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _ROUTER_PROMPT.format(question=question)}],
            max_tokens=300,
            temperature=0,
        )
        usage = (
            getattr(resp.usage, "prompt_tokens", 0) or 0,
            getattr(resp.usage, "completion_tokens", 0) or 0,
        ) if resp.usage else (0, 0)
        text = (resp.choices[0].message.content or "").strip()
        needs_search = True  # default to search on parse failure
        for line in reversed(text.splitlines()):
            line = line.strip().upper()
            if line.startswith("SEARCH:"):
                verdict = line.replace("SEARCH:", "").strip()
                needs_search = verdict.startswith("YES")
                break
        return needs_search, text, usage
    except Exception:
        return True, "", (0, 0)  # default to search on error


# ----------------------------
# Model limits
# ----------------------------

def _model_limits(model: str):
    """Get (context_tokens, max_output_tokens) for a model."""
    specs = get_model_specs()
    spec = specs.get(model, None)
    if spec:
        return int(spec.get("context", 32_000)), int(spec.get("max_output", 4_000))
    return 32_000, 4_000


# ----------------------------
# Agentic tool loop
# ----------------------------

def run_agentic_loop(
    client,
    messages: List[Dict[str, Any]],
    model: str,
    state: SessionState,
    max_output_tokens: Optional[int],
    supports_tools: bool,
    on_tool_start: Optional[Callable[[str, Dict], None]] = None,
    on_tool_result: Optional[Callable] = None,
    on_status: Optional[Callable[[str], None]] = None,
    on_think_draft: Optional[Callable[[str], None]] = None,
    on_reasoning_content: Optional[Callable[[str], None]] = None,
    post_process: Optional[Callable[[str], str]] = None,
    extra_create_kwargs: Optional[Dict[str, Any]] = None,
    override_tools: Optional[List[Dict[str, Any]]] = None,
    builtin_tool_handler: Optional[Callable] = None,
    asst_msg_builder: Optional[Callable] = None,
    token_param: str = "max_tokens",
    think_draft: Optional[str] = None,
) -> ReplyBundle:
    """
    Shared agentic tool loop for DeepSeek, Kimi, and OpenAI (non-web-search mode).
    """
    depth = getattr(state, "search_depth", "shallow")
    depth_cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["shallow"])
    force_answer_at = state.force_answer_at or depth_cfg["force_answer_at"]
    max_iterations = state.max_iterations or depth_cfg["max_iterations"]

    originating_question = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            originating_question = m.get("content", "")
            break

    ctx_limit, _ = _model_limits(model)
    ctx_chars = ctx_limit * 4

    loop_messages = [dict(m) for m in messages]

    # Inject pinned records (Group 1 — always in context, not compacted)
    _pinned = getattr(state, 'pinned_records', [])
    if _pinned:
        _pmsg = build_pinned_system_msg(_pinned)
        if _pmsg:
            # Insert after the main system message (position 0) if present
            insert_pos = 1 if loop_messages and loop_messages[0].get("role") == "system" else 0
            loop_messages.insert(insert_pos, {"role": "system", "content": _pmsg})

    search_mode = getattr(state, 'search_mode', 'auto')
    domain_filter = getattr(state, 'domain_filter', 'web')

    # Backward-compat: old search_mode values "web"/"academic" → treat as "auto"
    if search_mode not in ("auto", "on", "off"):
        search_mode = "auto"

    total_input_tokens = 0
    total_output_tokens = 0
    total_reasoning_tokens = 0   # known reasoning tokens (from completion_tokens_details)
    had_reasoning_unknown = False  # model reasoned but API didn't report count

    # Pre-loop router call for "auto" mode
    if search_mode == "auto" and supports_tools:
        needs_search, _router_reason, _router_usage = _route_needs_search(
            client, model, originating_question, on_status
        )
        total_input_tokens  += _router_usage[0]
        total_output_tokens += _router_usage[1]
        if needs_search:
            search_mode = "on"
            if on_status:
                on_status("Routing: search needed")
        else:
            search_mode = "off"
            if on_status:
                on_status("Routing: answering from training knowledge (no search)")

    # Inject search guidance for the agentic loop
    if supports_tools:
        cite_reminder = (
            " CITATION FORMAT: use Pandoc citation format [@ref_key] in your final answer"
            " (standard Pandoc/Quarto/R-Markdown syntax)."
            " The ref_key is shown in each read() result and in CITATION REMINDER messages below."
            " Example: \"This was shown by [@Lin2023explicit].\""
            " Do NOT invent keys."
        )
        if depth == "shallow":
            loop_guidance = (
                "SEARCH REMINDER: search() uses Brave — ask specific, natural questions."
                " Synthesize from snippets and LLM contexts before deciding to use read()."
                " Use read(url) only when snippets are clearly insufficient for a source you need."
                + cite_reminder
            )
        else:
            loop_guidance = (
                "SEARCH REMINDER: search() uses Brave — ask specific, natural questions."
                " After each search, synthesize what snippets and LLM contexts tell you."
                " Use read(url) on the most relevant sources for full text and proper citations."
                + cite_reminder
            )
        if domain_filter == "academic":
            loop_guidance += (
                " [Domain filter: ACADEMIC — academic sources are ranked higher in results;"
                " non-academic sources may still appear as fallback.]"
            )
        loop_messages.append({"role": "system", "content": loop_guidance})

    if search_mode == "off":
        loop_messages.append({
            "role": "system",
            "content": (
                "Web search is disabled for this session. Do NOT call search(). "
                "Answer directly from your knowledge. "
                "You may still use read() if a URL is explicitly provided in the conversation."
            ),
        })


    kwargs: Dict[str, Any] = {"model": model, "messages": loop_messages}
    if max_output_tokens is not None:
        kwargs[token_param] = int(max_output_tokens)

    if supports_tools:
        if override_tools is not None:
            eff_tools = override_tools
        elif search_mode == "off":
            eff_tools = [t for t in AGENTIC_TOOLS if t["function"]["name"] != "search"]
        else:
            eff_tools = AGENTIC_TOOLS
        kwargs["tools"] = eff_tools
        kwargs["tool_choice"] = "auto"

    if extra_create_kwargs:
        kwargs.update(extra_create_kwargs)

    all_consulted: List[Tuple[str, str]] = []
    all_records: List[RecordInfo] = []
    in_synthesis = False

    resp = None
    for iteration in range(max_iterations):
        if depth == "deep" and supports_tools and not in_synthesis:
            checkpoint_read = max(2, force_answer_at // 3)
            checkpoint_cite = max(checkpoint_read + 1, (force_answer_at * 2) // 3)
            checkpoint_gaps = max(checkpoint_cite + 1, force_answer_at - 2)
            # Ensure all checkpoints are before force_answer_at
            if checkpoint_gaps >= force_answer_at:
                checkpoint_gaps = force_answer_at - 1
            if checkpoint_cite >= checkpoint_gaps:
                checkpoint_cite = checkpoint_gaps - 1
            if checkpoint_read >= checkpoint_cite:
                checkpoint_read = checkpoint_cite - 1
            checkpoint_msg = None
            if iteration == checkpoint_read:
                checkpoint_msg = (
                    "CHECKPOINT: You've done initial searches. Now read the most promising papers "
                    "(5-8 in parallel) using read(url)."
                )
            elif iteration == checkpoint_cite:
                checkpoint_msg = (
                    "CHECKPOINT: Use get_paper_references on your best finds to explore citation "
                    "trails. Look for highly-cited papers that reference or are referenced by your key papers."
                )
            elif iteration == checkpoint_gaps:
                checkpoint_msg = (
                    "CHECKPOINT: Identify gaps in your coverage. Do targeted follow-up searches "
                    "for aspects you haven't found good sources on yet."
                )
            if checkpoint_msg:
                loop_messages.append({"role": "system", "content": checkpoint_msg})

        if iteration >= force_answer_at and supports_tools:
            if iteration == force_answer_at and not in_synthesis:
                in_synthesis = True
                loop_messages.append({
                    "role": "user",
                    "content": _build_force_answer_prompt(all_records, think_draft=think_draft),
                })
                # _build_force_answer_prompt calls build_record_inventory() which runs
                # deduplicate_ref_keys() and may mutate ref_keys in-place (e.g. adding
                # 'a'/'b' suffixes to colliding keys). Rebuild reread_registry so that
                # reread() calls in the synthesis phase use the same keys shown in the
                # SOURCE INVENTORY.
                registry = getattr(state, 'reread_registry', None)
                if registry is not None:
                    registry.clear()
                    for rec in all_records:
                        if rec.ref_key and rec.access_level in ("full_text", "abstract_only"):
                            registry[rec.ref_key] = {
                                "title": rec.title,
                                "url": rec.url,
                                "external_id": rec.external_id,
                                "local_path": getattr(rec, 'local_path', ''),
                            }
                kwargs.pop("tools", None)
                kwargs.pop("tool_choice", None)
                kwargs["messages"] = loop_messages

        resp = client.chat.completions.create(**kwargs)
        if resp.usage:
            total_input_tokens  += getattr(resp.usage, "prompt_tokens",     0) or 0
            total_output_tokens += getattr(resp.usage, "completion_tokens", 0) or 0
            details = getattr(resp.usage, "completion_tokens_details", None)
            rt = getattr(details, "reasoning_tokens", None) if details else None
            if rt is not None:
                total_reasoning_tokens += rt
            elif getattr(resp.choices[0].message if resp.choices else None, "reasoning_content", None):
                had_reasoning_unknown = True
        choice = resp.choices[0]
        finish_reason = choice.finish_reason

        # Verbose debug: show iteration state
        if getattr(state, 'verbose', False) and on_status:
            rc = getattr(choice.message, "reasoning_content", None) or ""
            ct = choice.message.content or ""
            on_status(
                f"[dim][debug] iter={iteration} finish={finish_reason} "
                f"content={len(ct)}ch reasoning={len(rc)}ch[/dim]"
            )

        if finish_reason != "tool_calls" or not supports_tools or in_synthesis:
            content = choice.message.content or ""
            reasoning = getattr(choice.message, "reasoning_content", None)
            # Surface model reasoning tokens (Qwen3 enable_thinking, gpt-oss-120b reasoning_content).
            if on_reasoning_content and reasoning:
                on_reasoning_content(reasoning)

            # Mid-loop: model stopped with no text (reasoning-only turn).
            # Push to synthesis on the next iteration instead of returning empty.
            if (not content.strip() and not in_synthesis
                    and supports_tools and iteration < max_iterations - 1):
                if getattr(state, 'verbose', False) and on_status:
                    on_status("[dim][debug] empty content mid-loop → forcing synthesis[/dim]")
                force_answer_at = min(force_answer_at, iteration + 1)
                kwargs["messages"] = loop_messages
                continue

            # Mid-loop: model stopped calling tools early and wrote a prose answer,
            # but the force-answer prompt (SOURCE INVENTORY + citation instructions)
            # was never injected. Trigger synthesis now so citations get formatted.
            if (content.strip() and not in_synthesis
                    and supports_tools and all_records
                    and iteration < max_iterations - 1):
                if getattr(state, 'verbose', False) and on_status:
                    on_status("[dim][debug] early stop with content → triggering synthesis for citations[/dim]")
                in_synthesis = True
                asst_early: Dict[str, Any] = {"role": "assistant", "content": content}
                reasoning_early = getattr(choice.message, "reasoning_content", None)
                if reasoning_early:
                    asst_early["reasoning_content"] = reasoning_early
                loop_messages.append(asst_early)
                loop_messages.append({
                    "role": "user",
                    "content": _build_force_answer_prompt(all_records, think_draft=think_draft),
                })
                kwargs.pop("tools", None)
                kwargs.pop("tool_choice", None)
                kwargs["messages"] = loop_messages
                continue

            # Synthesis: if model returned empty content, retry once with a
            # direct nudge. Some reasoning-heavy models think extensively then
            # emit no text on the first synthesis turn.
            if not content.strip() and in_synthesis:
                if getattr(state, 'verbose', False) and on_status:
                    on_status("[dim][debug] empty content at synthesis → retry[/dim]")
                # Include reasoning_content in the assistant slot so APIs that require
                # it (e.g. gpt-oss-120b) don't reject the message history.
                asst_retry: Dict[str, Any] = {"role": "assistant", "content": ""}
                if reasoning:
                    asst_retry["reasoning_content"] = reasoning
                loop_messages.append(asst_retry)
                loop_messages.append({
                    "role": "user",
                    "content": "Please write your complete answer as text now.",
                })
                retry_kwargs = dict(kwargs)
                retry_kwargs["messages"] = loop_messages
                try:
                    retry_resp = client.chat.completions.create(**retry_kwargs)
                    if retry_resp.usage:
                        total_input_tokens  += getattr(retry_resp.usage, "prompt_tokens",     0) or 0
                        total_output_tokens += getattr(retry_resp.usage, "completion_tokens", 0) or 0
                        details = getattr(retry_resp.usage, "completion_tokens_details", None)
                        rt = getattr(details, "reasoning_tokens", None) if details else None
                        if rt is not None:
                            total_reasoning_tokens += rt
                        elif getattr(retry_resp.choices[0].message if retry_resp.choices else None, "reasoning_content", None):
                            had_reasoning_unknown = True
                    retry_choice = retry_resp.choices[0]
                    content = retry_choice.message.content or ""
                    retry_rc = getattr(retry_choice.message, "reasoning_content", None)
                    if on_reasoning_content and retry_rc:
                        on_reasoning_content(retry_rc)
                except Exception:
                    pass  # fall through with empty content

            if post_process:
                content = post_process(content)
            if all_records:
                content = apply_references(content, all_records, state.citation_style)
            # Clear all snippet Records from /sources after synthesis — they remain
            # browsable via /snippets. Only snippets that were actually cited remain
            # visible in the rendered reference list.
            for rec in state.records:
                if not rec.cleared and rec.info.access_level == "snippet":
                    rec.cleared = True
            inventory = build_record_inventory(all_records)
            if had_reasoning_unknown:
                final_reasoning = -1
            elif total_reasoning_tokens > 0:
                final_reasoning = total_reasoning_tokens
            else:
                final_reasoning = None
            return ReplyBundle(
                text=content,
                cited=[],
                consulted=dedup_pairs(all_consulted),
                source_details=all_records,
                tool_context_summary=inventory if inventory else None,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                reasoning_tokens=final_reasoning,
            )

        assistant_msg = choice.message

        if asst_msg_builder:
            asst_dict = asst_msg_builder(assistant_msg)
        else:
            asst_dict = {
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": [tc.model_dump() for tc in assistant_msg.tool_calls],
            }
        loop_messages.append(asst_dict)

        deferred_messages: list = []

        for tc in assistant_msg.tool_calls:
            tool_name = tc.function.name

            if builtin_tool_handler and builtin_tool_handler(
                tc, loop_messages, on_tool_start, on_tool_result,
            ):
                continue

            try:
                tool_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_args = {}

            validation_error = REGISTRY.validate_args(tool_name, tool_args)
            if validation_error:
                loop_messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps({"success": False, "error": validation_error}),
                })
                continue

            if on_tool_start:
                on_tool_start(tool_name, tool_args)

            result, consulted, record_infos = execute_tool(tool_name, tool_args, state)
            all_consulted.extend(consulted)
            all_records.extend(record_infos)
            # Populate reread registry so the reread() tool can re-access papers later
            registry = getattr(state, 'reread_registry', None)
            if registry is not None:
                for rec in record_infos:
                    if rec.ref_key and rec.access_level in ("full_text", "abstract_only"):
                        registry[rec.ref_key] = {
                            "title": rec.title,
                            "url": rec.url,
                            "external_id": rec.external_id,
                            "local_path": getattr(rec, 'local_path', ''),
                        }

            content = json.dumps(result, ensure_ascii=False)
            original_len = len(content)
            limit = state.agentic_tool_max_chars
            if limit and original_len > limit:
                content = content[:limit] + f"\n... [truncated from {original_len:,} to {limit:,} chars]"

            REGISTRY.register_record(tool_name, state, tool_args, result, record_infos,
                             len(content), original_len if original_len > (limit or original_len + 1) else 0,
                             originating_question=originating_question)

            if on_tool_result:
                on_tool_result(tool_name, tool_args, result,
                               char_count=len(content), truncated_from=original_len if limit and original_len > limit else 0)

            loop_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content,
            })

            # After read/reread: tell the model the citation key for this source.
            # This ensures the model can cite it even if it exits before force_answer_at
            # (which is when the full SOURCE INVENTORY would otherwise be injected).
            if tool_name in ("read", "reread", "read_paper", "read_webpage") and not in_synthesis:
                citable = [
                    ri for ri in record_infos
                    if ri.ref_key and ri.access_level in ("full_text", "webpage")
                ]
                if citable:
                    key_hints = "; ".join(
                        f"[{ri.ref_key}] \"{ri.title[:60]}{'…' if len(ri.title) > 60 else ''}\""
                        f" ({ri.access_level})"
                        for ri in citable
                    )
                    deferred_messages.append({
                        "role": "system",
                        "content": f"CITATION REMINDER: {key_hints}. In your final answer use Pandoc citation format [@ref_key] for each source you draw on.",
                    })

        loop_messages.extend(deferred_messages)

        if not in_synthesis:
            total_chars = sum(len(m.get("content", "")) for m in loop_messages)
            if total_chars > int(ctx_chars * 0.72) and iteration + 1 < force_answer_at:
                force_answer_at = iteration + 1
                if on_status:
                    on_status(
                        f"Context limit reached ({total_chars:,} chars) — "
                        f"synthesizing with current sources."
                    )

        kwargs["messages"] = loop_messages

    # Guardrail: max iterations reached
    content = ""
    if resp:
        content = resp.choices[0].message.content or ""
    if post_process:
        content = post_process(content)
    if all_records:
        content = apply_references(content, all_records, state.citation_style)
    for rec in state.records:
        if not rec.cleared and rec.info.access_level == "snippet":
            rec.cleared = True
    inventory = build_record_inventory(all_records)
    return ReplyBundle(
        text=content, cited=[], consulted=dedup_pairs(all_consulted),
        source_details=all_records,
        tool_context_summary=inventory if inventory else None,
    )
