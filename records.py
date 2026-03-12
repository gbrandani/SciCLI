"""
records.py — Reference formatting and source ordering for Chat CLI.

Handles post-processing of model output to replace [BibTeXKey] patterns
with display-format citations and build a References section.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional
from urllib.parse import urlsplit

from config import RecordInfo

# Matches [@key] or [@key1; @key2] (Pandoc-style citations)
_CITE_PAT = r'\[(@[A-Za-z][A-Za-z0-9_.]*(?:\s*[;,]\s*@[A-Za-z][A-Za-z0-9_.]*)*)\]'


def _split_keys(group: str) -> list:
    """Split '@key1; @key2' into bare keys ['key1', 'key2']."""
    return [k.lstrip('@').strip() for k in re.split(r'\s*[;,]\s*', group) if k.strip()]


def _first_surname(authors: str) -> str:
    """Extract the first author's surname from an author list string."""
    if not authors:
        return "?"
    first = authors.split(",")[0].strip()
    if not first:
        return "?"
    # If "John Smith" format, take last word
    parts = first.split()
    if len(parts) >= 2:
        return parts[-1]
    return first


def _build_verified_references(
    text: str,
    source_details: List[RecordInfo],
    verified_keys: set,
    citation_style: str = "numbered",
) -> str:
    """Post-process model output: replace verified [BibTeXKey] with display format
    and build a References section from verified citations only.

    citation_style: "numbered" -> [1], [2], ...  or "authoryear" -> [Author, Venue, Year]
    """
    if not source_details or not text:
        return text

    ordered = get_ordered_sources(source_details)
    if not ordered:
        return text

    # Build key→source mapping
    key_map = {s.ref_key: s for s in ordered if s.ref_key}

    # Determine display labels for each verified key
    display_labels: Dict[str, str] = {}  # bibtex_key → display label
    if citation_style == "numbered":
        # Assign numbers by first appearance in text
        key_order = []
        for m in re.finditer(_CITE_PAT, text):
            for k in _split_keys(m.group(1)):
                if k in verified_keys and k not in key_order:
                    key_order.append(k)
        # Add any verified keys not found in text
        for k in sorted(verified_keys):
            if k not in key_order:
                key_order.append(k)
        for i, k in enumerate(key_order, 1):
            display_labels[k] = str(i)
    elif citation_style == "pandoc":
        # Use ref_key itself as label → [@Lin2023explicit] stays as-is in output
        for k in verified_keys:
            display_labels[k] = k
    else:
        # authoryear style: [FirstAuthor, Venue, Year]
        raw_labels: Dict[str, str] = {}
        for k in verified_keys:
            s = key_map.get(k)
            if not s:
                continue
            surname = _first_surname(getattr(s, 'authors', '') or '')
            venue_short = getattr(s, 'venue', '') or ""
            year = getattr(s, 'year', '') or ""
            if s.access_level == "webpage":
                domain = urlsplit(s.url).netloc.replace("www.", "") if s.url else ""
                label_name = surname if surname != "?" else (s.title.split(" - ")[0][:30] if s.title else domain)
                raw_labels[k] = f"{label_name}, {domain}".rstrip(", ")
            else:
                raw_labels[k] = f"{surname}, {venue_short}, {year}".rstrip(", ")

        # Disambiguate collisions
        label_groups: Dict[str, List[str]] = {}
        for k, label in raw_labels.items():
            label_groups.setdefault(label, []).append(k)
        for label, keys in label_groups.items():
            if len(keys) == 1:
                display_labels[keys[0]] = label
            else:
                for i, k in enumerate(sorted(keys)):
                    display_labels[k] = f"{label}{chr(ord('a') + i)}"

    if not display_labels:
        return text

    # Replace [BibTeXKey] or [Key1; Key2; Key3] with display format in text.
    # Use bold brackets **[N]** — survives Rich Markdown rendering as visually distinct text.
    def _replace_citation(m):
        keys = _split_keys(m.group(1))
        labels = [display_labels[k] for k in keys if k in display_labels]
        if not labels:
            return m.group(0)  # unmatched: leave [@unknownKey] visible
        if citation_style == "pandoc":
            return m.group(0)  # keep [@key] unchanged for copy-paste
        # Ensure there is a space before the citation bracket when the preceding
        # character is not already whitespace (models often write "word[Key]" without a space).
        prefix = ""
        if m.start() > 0 and m.string[m.start() - 1] not in (' ', '\n', '\t', '\r'):
            prefix = " "
        return f"{prefix}**[{', '.join(labels)}]**"

    processed = re.sub(_CITE_PAT, _replace_citation, text)

    # Pull citations that ended up at the start of a line back inline.
    # Handles the case where the model writes "word\n[Key]" — the regex above
    # preserves the \n (treating it as whitespace), but the result looks like a
    # new paragraph in the terminal. Convert "\n  **[N]**" → " **[N]**".
    processed = re.sub(r'\n[ \t]*(\*\*\[[^\]]+\]\*\*)', r' \1', processed)

    # Strip any model-generated References section (covers many formats the model might use)
    ref_patterns = [
        r'\n#{1,3}\s*References?\s*\n.*$',          # ## References
        r'\n\*{0,2}References?\*{0,2}\s*:?\s*\n.*$', # **References** or *References*
        r'\nReferences?\s*:?\s*\n[-\[•*].*$',         # References:\n- or •
        r'\n---+\s*\nReferences?\s*\n.*$',             # ---\nReferences
        r'\n\s*References?\s*\n\s*[•\-\*\[].*$',       # plain References with bullets
        r'\n\s*References?\s*\n\s*\d+[\.\)].*$',        # References with numbered list
    ]
    for pat in ref_patterns:
        processed = re.sub(pat, '', processed, flags=re.DOTALL | re.IGNORECASE)
    processed = processed.rstrip()

    # Build canonical references section with a special delimiter that render_assistant
    # will detect and render with color (bypassing Markdown).
    ref_lines = ["\n\n\x00REFS\x00"]
    # Order: by display label (numeric sort for numbered style, lexicographic otherwise)
    def _label_sort_key(k):
        lbl = display_labels[k]
        try:
            return (0, int(lbl), "")
        except ValueError:
            return (1, 0, lbl)
    sorted_keys = sorted(display_labels.keys(), key=_label_sort_key)
    for k in sorted_keys:
        s = key_map.get(k)
        if not s:
            continue
        label = display_labels[k]
        author_bit = f"{getattr(s, 'authors', '') or ''}. " if getattr(s, 'authors', '') else ""
        year_bit = f"({getattr(s, 'year', '') or ''}). " if getattr(s, 'year', '') else ""
        title_bit = f'"{s.title}". ' if s.title else ""
        venue_bit = f"{getattr(s, 'venue', '') or ''}. " if getattr(s, 'venue', '') else ""
        url_bit = s.url or ""
        if s.access_level == "full_text":
            access_note = " (full text)"
        elif s.access_level == "abstract_only":
            access_note = " (abstract only)"
        elif s.access_level == "webpage":
            access_note = " (webpage)"
        elif s.access_level == "snippet":
            access_note = " (search snippet)"
        else:
            access_note = ""
        ref_line = f"\x00REF\x00[{label}]\x00{author_bit}{year_bit}{title_bit}{venue_bit}{url_bit}{access_note}"
        ref_lines.append(ref_line)

    if len(ref_lines) > 1:
        processed += "\n".join(ref_lines)

    return processed


def apply_references(
    text: str,
    source_details: List[RecordInfo],
    citation_style: str = "numbered",
) -> str:
    """Apply references to text by detecting inline [BibTeXKey] patterns
    and building a formatted References section."""
    if not text or not source_details:
        return text

    ordered = get_ordered_sources(source_details)
    if not ordered:
        return text

    # Detect [BibTeXKey] or [Key1; Key2] patterns inline
    key_map = {s.ref_key: s for s in ordered if s.ref_key}
    inline_keys = set()
    for m in re.finditer(_CITE_PAT, text):
        for k in _split_keys(m.group(1)):
            if k in key_map:
                inline_keys.add(k)

    if not inline_keys:
        return text

    return _build_verified_references(text, source_details, inline_keys, citation_style)


def get_ordered_sources(source_details: List[RecordInfo]) -> List[RecordInfo]:
    """Get deduplicated, ordered source list matching inventory numbering."""
    seen: set = set()
    deduped: List[RecordInfo] = []
    for s in source_details:
        key = (s.tool_name, s.external_id or s.url)
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    ordered: List[RecordInfo] = []
    for s in deduped:
        if s.access_level == "full_text":
            ordered.append(s)
    for s in deduped:
        if s.access_level == "webpage":
            ordered.append(s)
    for s in deduped:
        if s.access_level == "snippet":
            ordered.append(s)
    return ordered


def build_pinned_system_msg(pinned_records: list) -> Optional[str]:
    """Build a system message from all pinned records for injection into every API call.

    Pinned records (Group 1) always stay in context and are never removed by compaction.
    Each record is formatted with full metadata so the model can make best use of it.
    """
    if not pinned_records:
        return None

    SEP = "─" * 60
    parts = [
        "PINNED SOURCES",
        "These documents are permanently kept in context and will NOT be removed by compaction.",
        "Cite them using [@ref_key] in inline citations exactly as you would any source.",
        SEP,
    ]

    for pr in pinned_records:
        ref_key = getattr(pr, 'ref_key', '') or ''
        title = getattr(pr, 'title', '') or ''
        content = getattr(pr, 'content', '') or ''
        source_type = getattr(pr, 'source_type', '') or ''
        note = getattr(pr, 'note', '') or ''
        authors = getattr(pr, 'authors', '') or ''
        year = getattr(pr, 'year', '') or ''
        venue = getattr(pr, 'venue', '') or ''
        url = getattr(pr, 'url', '') or ''
        access_level = getattr(pr, 'access_level', '') or ''
        local_path = getattr(pr, 'local_path', '') or ''

        lines = [f"[{ref_key}] {title}"]

        # Authors / year / venue
        meta = []
        if authors:
            meta.append(authors)
        if year:
            meta.append(f"({year})")
        if venue:
            meta.append(venue)
        if meta:
            lines.append("  " + "  ".join(meta))

        # Source type / access level / URL
        src = []
        if source_type:
            src.append(source_type)
        if access_level:
            src.append(access_level)
        if url:
            src.append(url)
        elif local_path:
            src.append(f"local: {local_path}")
        if src:
            lines.append("  " + " | ".join(src))

        if note:
            lines.append(f"  Note: {note}")

        lines.append("")  # blank line before content
        lines.append(content)
        lines.append(SEP)

        parts.append("\n".join(lines))

    return "\n".join(parts)


def build_record_inventory(source_details: List[RecordInfo]) -> str:
    """Build a record inventory using ref keys for the force-answer prompt.

    Delegates to REGISTRY.build_inventory after pre-processing.
    """
    if not source_details:
        return ""

    from tool_registry import REGISTRY
    import tools as _tools  # noqa: F401 — ensure plugins are registered
    from utils import deduplicate_ref_keys

    # Assign ref_key to webpages and snippets that lack one
    snap_n = 0
    for i, s in enumerate(source_details):
        if s.access_level == "webpage" and not s.ref_key:
            s.ref_key = f"web{i+1}"
        elif s.access_level == "snippet" and not s.ref_key:
            snap_n += 1
            s.ref_key = f"snap{snap_n}"

    # Deduplicate ref keys only among citable records (full_text, webpage).
    # abstract_only, snippet, and search_result are excluded: not citable.
    inventory_records = [
        s for s in source_details
        if s.access_level not in ("search_result", "abstract_only", "snippet")
    ]
    deduplicate_ref_keys(inventory_records)

    return REGISTRY.build_inventory(source_details)
