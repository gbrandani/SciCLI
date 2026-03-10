"""
tool_registry.py — Plugin-based tool registry for Chat CLI.

Each tool is a self-contained ToolPlugin with schema, execute, and record
registration logic.  A global REGISTRY singleton collects all plugins.
"""

from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import RecordInfo, PaperRecord, Record, SessionState


# ----------------------------
# ToolPlugin dataclass
# ----------------------------

@dataclass
class ToolPlugin:
    """A self-contained agentic tool."""
    name: str
    schema: dict                    # OpenAI function-calling schema
    execute: Callable               # (args, state) -> {"result": ..., "consulted": [...], "source_infos": [...]}
    register_record: Callable       # (state, args, result, record_infos, char_count, truncated_from, question) -> None
    inventory_category: str         # grouping key for inventory display
    inventory_formatter: Callable   # (RecordInfo) -> Optional[str]  (one-line inventory entry, or None to skip)


# ----------------------------
# ToolRegistry
# ----------------------------

class ToolRegistry:
    """Central registry of all agentic tools."""

    def __init__(self) -> None:
        self._plugins: Dict[str, ToolPlugin] = {}
        self._order: List[str] = []   # insertion order for schema list

    def register(self, plugin: ToolPlugin) -> None:
        if plugin.name not in self._plugins:
            self._order.append(plugin.name)
        self._plugins[plugin.name] = plugin

    def get_schemas(self) -> List[dict]:
        return [self._plugins[n].schema for n in self._order if n in self._plugins]

    def has_tool(self, name: str) -> bool:
        return name in self._plugins

    def execute(
        self, name: str, args: Dict[str, Any], state: SessionState,
    ) -> Tuple[Dict[str, Any], List[Tuple[str, str]], List[RecordInfo]]:
        """Execute a tool by name.  Returns (result, consulted, record_infos)."""
        plugin = self._plugins.get(name)
        if not plugin:
            return {"success": False, "error": f"Unknown tool: {name}"}, [], []
        raw = plugin.execute(args, state)
        consulted = raw.pop("consulted", [])
        record_infos = raw.pop("source_infos", raw.pop("record_infos", []))
        result = raw.get("result", raw)
        return result, consulted, record_infos

    def register_record(
        self,
        name: str,
        state: SessionState,
        args: dict,
        result: dict,
        record_infos: list,
        char_count: int,
        truncated_from: int,
        originating_question: str = "",
    ) -> None:
        plugin = self._plugins.get(name)
        if plugin:
            plugin.register_record(
                state, args, result, record_infos,
                char_count, truncated_from, originating_question,
            )

    def build_inventory(self, source_details: List[RecordInfo]) -> str:
        """Build the SOURCE INVENTORY string from a list of RecordInfo objects.

        Delegates formatting to each plugin's inventory_formatter, grouping by
        inventory_category.
        """
        if not source_details:
            return ""

        # Deduplicate by (external_id or url or title), keeping the record with
        # the best access_level for each key (full_text > abstract_only > webpage >
        # search_result).  This ensures that if a paper was both searched and read,
        # the full-text record is shown rather than the search-result one being
        # silently dropped because it happened to appear first in the list.
        _ACCESS_RANK = {"full_text": 0, "abstract_only": 1, "webpage": 2, "snippet": 3}  # lower = better
        best_by_key: Dict[str, RecordInfo] = {}
        order_keys: List[str] = []   # first-seen order (for stable display)
        no_key_records: List[RecordInfo] = []

        for s in source_details:
            key = s.external_id or s.url or s.title
            if not key:
                no_key_records.append(s)
                continue
            if key not in best_by_key:
                order_keys.append(key)
                best_by_key[key] = s
            else:
                curr_rank = _ACCESS_RANK.get(best_by_key[key].access_level, 99)
                new_rank  = _ACCESS_RANK.get(s.access_level, 99)
                if new_rank < curr_rank:
                    best_by_key[key] = s   # upgrade to better record

        deduped = [best_by_key[k] for k in order_keys] + no_key_records

        # Group by category using plugin lookup
        groups: Dict[str, List[str]] = {}
        search_count = 0

        # Category display config  (order, header)
        category_meta = {
            "papers_full": "Papers read (full text):",
            "papers_abstract": "Papers consulted (abstract only):",
            "sequences": "Sequences fetched:",
            "web": "Web pages and search snippets (citable):",
        }

        for s in deduped:
            plugin = self._plugins.get(s.tool_name)
            if not plugin:
                continue

            cat = plugin.inventory_category

            # Determine effective category — some plugins use dynamic categories
            if cat == "papers":
                if s.access_level == "full_text":
                    cat = "papers_full"
                elif s.access_level == "abstract_only":
                    cat = "papers_abstract"
                else:
                    cat = "search"

            if cat == "search":
                if s.access_level == "snippet":
                    cat = "web"
                else:
                    search_count += 1
                    continue

            line = plugin.inventory_formatter(s)
            if line:
                groups.setdefault(cat, []).append(line)

        lines = ["SOURCE INVENTORY (what you have actually accessed):", ""]

        for cat_key in ["papers_full", "papers_abstract", "sequences", "web"]:
            entries = groups.get(cat_key, [])
            if entries:
                header = category_meta.get(cat_key, f"{cat_key}:")
                lines.append(header)
                for e in entries:
                    lines.append(f"  {e}")
                lines.append("")

        if search_count:
            lines.append(f"({search_count} additional search results seen but not read)")
            lines.append("")

        return "\n".join(lines)

    def validate_args(self, name: str, args: dict) -> Optional[str]:
        """Validate tool arguments against schema.  Returns error string or None."""
        plugin = self._plugins.get(name)
        if not plugin:
            return f"Unknown tool: {name}"

        params = plugin.schema.get("function", {}).get("parameters", {})
        required = params.get("required", [])
        properties = params.get("properties", {})

        # Check required params
        for r in required:
            if r not in args or args[r] is None or (isinstance(args[r], str) and not args[r].strip()):
                return f"Missing required parameter: {r}"

        # Check enum values
        for key, val in args.items():
            prop = properties.get(key, {})
            enum = prop.get("enum")
            if enum and val not in enum:
                return f"Invalid value for '{key}': {val!r}. Must be one of {enum}"

        return None


# ----------------------------
# Record registration helpers
# ----------------------------

def _make_record_entry(
    state: SessionState,
    info: RecordInfo,
    char_count: int,
    truncated_from: int,
    content_type: str,
    originating_question: str = "",
) -> "Record":
    """Create and append a Record entry to state. Returns the created Record."""
    entry = Record(
        id=state.records_next_id,
        info=info,
        char_count=char_count,
        truncated_from=truncated_from,
        content_type=content_type,
        timestamp=_time.time(),
        originating_question=originating_question,
    )
    state.records.append(entry)
    state.records_next_id += 1
    return entry


# ----------------------------
# Global singleton
# ----------------------------

REGISTRY = ToolRegistry()
