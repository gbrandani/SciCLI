"""
compaction.py — Conversation-level compaction for SciCLI.

Summarizes conversation history while preserving source inventory with
llm_contexts, so the model can still cite papers by ref_key after compaction.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from config import RecordInfo, SessionState


def _call_compact_model(client, compact_model: str, history_text: str) -> str:
    """Summarize a conversation history using the compact model.

    Returns the summary as a string, or empty string on failure.
    """
    summary_prompt = (
        "You are compacting a research chat transcript to keep it within context limits.\n\n"
        "Produce a concise but information-dense summary that preserves:\n"
        "- the user's research question(s) and goals\n"
        "- key findings discussed so far\n"
        "- important technical details and conclusions\n"
        "- open questions and next steps\n"
        "- any source ref_keys mentioned (e.g. Zhang2024explicit, snap3)\n\n"
        "Write the summary as plain text with clear bullet points.\n"
        "Target: 200-400 words."
    )
    try:
        resp = client.chat.completions.create(
            model=compact_model,
            messages=[
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": f"Conversation to compact:\n\n{history_text[:30000]}"},
            ],
            max_tokens=600,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def compact_conversation(
    messages: list,
    source_details: List[RecordInfo],
    client,
    compact_model: str,
    on_status: Optional[Callable[[str], None]] = None,
) -> list:
    """Summarize conversation history; preserve source inventory with llm_contexts.

    Returns a rebuilt messages list:
      [system_msg (if any), summary_assistant_msg, source_block_system_msg (if any)]
    """
    from records import build_record_inventory

    inventory = build_record_inventory(source_details) if source_details else ""

    history_text = "\n\n".join(
        f"[{m['role'].upper()}]: {m.get('content', '')}"
        for m in messages if m["role"] != "system"
    )
    summary = _call_compact_model(client, compact_model, history_text)
    if not summary:
        # Compaction failed — return messages unchanged
        return messages

    system_msgs = [m for m in messages if m["role"] == "system"][:1]
    new_messages = system_msgs + [
        {"role": "assistant", "content": f"[Conversation summary]\n{summary}"},
    ]
    if inventory:
        new_messages.append({
            "role": "system",
            "content": (
                "SOURCES FROM PRIOR RESEARCH (use these ref keys to cite):\n\n"
                + inventory
            ),
        })
    if on_status:
        on_status(f"Conversation compacted. {len(source_details)} sources preserved.")
    return new_messages
