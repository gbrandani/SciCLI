"""
providers.py — LLM provider classes for Chat CLI.

OpenAI (Chat Completions), DeepSeek, Kimi, Sakura, and Together AI providers.
Agentic loop, records, and compaction logic live in separate modules.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI

from config import SessionState, ReplyBundle, STATS_FILE, get_model_full_id
from utils import extract_kimi_final, safe_read_json, safe_write_json
from tools import AGENTIC_TOOLS
from agentic import run_research_pipeline, _clean_dsml_artifacts, _clean_together_artifacts

# Backward-compat re-exports (tests and chat_cli may import these)
from agentic import (
    run_agentic_loop as _run_agentic_loop,
    _build_force_answer_prompt,
)
from records import build_record_inventory as _build_record_inventory


# ----------------------------
# Base
# ----------------------------

class ProviderBase:
    name: str

    def send(
        self,
        messages: List[Dict[str, str]],
        model: str,
        state: SessionState,
        max_output_tokens: Optional[int] = None,
        use_tools: bool = True,
        on_tool_start: Optional[Callable[[str, Dict], None]] = None,
        on_tool_result: Optional[Callable] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_think_draft: Optional[Callable[[str], None]] = None,
        on_reasoning_content: Optional[Callable[[str], None]] = None,
    ) -> ReplyBundle:
        raise NotImplementedError


# ----------------------------
# OpenAI (Chat Completions API with agentic tool loop)
# ----------------------------

class OpenAIChatProvider(ProviderBase):
    """OpenAI Chat Completions API with full agentic tool loop."""
    name = "openai"

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def send(
        self,
        messages: List[Dict[str, str]],
        model: str,
        state: SessionState,
        max_output_tokens: Optional[int] = None,
        use_tools: bool = True,
        on_tool_start: Optional[Callable[[str, Dict], None]] = None,
        on_tool_result: Optional[Callable] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_think_draft: Optional[Callable[[str], None]] = None,
        on_reasoning_content: Optional[Callable[[str], None]] = None,
    ) -> ReplyBundle:
        return run_research_pipeline(
            client=self.client,
            messages=messages,
            model=model,
            state=state,
            max_output_tokens=max_output_tokens,
            supports_tools=use_tools,
            on_tool_start=on_tool_start,
            on_tool_result=on_tool_result,
            on_status=on_status,
            on_think_draft=on_think_draft,
            on_reasoning_content=on_reasoning_content,
            token_param="max_completion_tokens",
        )


# Backward-compatible aliases
OpenAIProvider = OpenAIChatProvider
_build_source_inventory = _build_record_inventory


# ----------------------------
# OpenAI Responses API (native web_search — no agentic tool loop)
# ----------------------------

class OpenAIResponsesProvider(ProviderBase):
    """
    OpenAI Responses API with native web_search tool.

    Key differences from OpenAIChatProvider:
    - Uses client.responses.create() instead of chat.completions.create()
    - Web search is native/opaque — no Brave, no custom tool loop
    - Citations and consulted sources extracted from response annotations
    - No agentic pipeline, no SOURCE INVENTORY, no Brave/S2 usage
    - Compaction and context management still work (use_tools=False path)
    """
    name = "openai_responses"

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def send(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        state: SessionState,
        max_output_tokens: Optional[int] = None,
        use_tools: bool = True,
        **_callbacks,  # on_tool_start, on_tool_result, etc. — no-ops here
    ) -> ReplyBundle:
        kwargs: Dict[str, Any] = {"model": model, "input": messages}

        eff = (state.reasoning_effort or "").strip().lower()
        if eff and eff not in ("auto", ""):
            kwargs["reasoning"] = {"effort": eff}

        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = int(max_output_tokens)

        # use_tools=True → enable native web_search (controlled by search_mode in scicli.py)
        if use_tools:
            kwargs["tools"] = [{"type": "web_search"}]
            kwargs["tool_choice"] = "auto"
            kwargs["include"] = ["web_search_call.action.sources"]

        resp = self.client.responses.create(**kwargs)
        answer_text = getattr(resp, "output_text", "") or ""

        cited: List[Tuple[str, str]] = []
        consulted: List[Tuple[str, str]] = []

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                resp_d = resp.model_dump()
        except Exception:
            resp_d = {}

        for item in (resp_d.get("output") or []):
            itype = item.get("type")
            if itype == "message":
                for part in (item.get("content") or []):
                    if part.get("type") == "output_text":
                        for ann in (part.get("annotations") or []):
                            if ann.get("type") == "url_citation":
                                cited.append((ann.get("title") or "", ann.get("url") or ""))
            elif itype == "web_search_call":
                action = item.get("action") or {}
                for src in (action.get("sources") or []):
                    consulted.append((src.get("title") or "", src.get("url") or ""))

        seen: set = set()
        deduped_cited: List[Tuple[str, str]] = []
        deduped_consulted: List[Tuple[str, str]] = []
        for pairs, out in ((cited, deduped_cited), (consulted, deduped_consulted)):
            for title, url in pairs:
                key = url or title
                if key and key not in seen:
                    seen.add(key)
                    out.append((title, url))

        usage = getattr(resp, "usage", None)
        in_tok  = getattr(usage, "input_tokens",  0) or 0 if usage else 0
        out_tok = getattr(usage, "output_tokens", 0) or 0 if usage else 0

        # Responses API: reasoning tokens are under usage.output_tokens_details.reasoning_tokens
        r_tok: Optional[int] = None
        if usage:
            details = getattr(usage, "output_tokens_details", None)
            rt = getattr(details, "reasoning_tokens", None) if details else None
            if rt is not None:
                r_tok = int(rt)

        return ReplyBundle(
            text=answer_text,
            cited=deduped_cited,
            consulted=deduped_consulted,
            input_tokens=in_tok,
            output_tokens=out_tok,
            reasoning_tokens=r_tok,
        )


# ----------------------------
# Sakura Internet AI Engine (OpenAI-compatible Chat Completions)
# ----------------------------

SAKURA_USAGE_FILE = Path(".sakura_usage.json")  # legacy file; migrated to stats file
SAKURA_MONTHLY_LIMIT = 3000


class SakuraUsageTracker:
    """Tracks monthly Sakura API request count against the free-tier limit.

    Reads/writes the 'sakura' key in the shared .scicli_stats.json file so that
    all provider and Brave usage is consolidated in one place.
    """

    def __init__(self, usage_file: Path = STATS_FILE):
        self.usage_file = usage_file

    def _month_key(self) -> str:
        return datetime.now().strftime("%Y-%m")

    def _load(self) -> Dict[str, int]:
        """Return just the sakura monthly counts dict."""
        data = safe_read_json(self.usage_file, {})
        svc = data.get("sakura", {})
        # Backward compat: old format was {"YYYY-MM": N} at top level
        if svc and not isinstance(svc, dict):
            return {}
        return svc

    def _save(self, monthly: Dict[str, int]) -> None:
        """Update the sakura key in the stats file."""
        data = safe_read_json(self.usage_file, {})
        data["sakura"] = monthly
        safe_write_json(self.usage_file, data)

    def current_count(self) -> int:
        return self._load().get(self._month_key(), 0)

    def check_limit(self) -> Optional[str]:
        """Returns an error string if the monthly limit has been reached, else None."""
        count = self.current_count()
        if count >= SAKURA_MONTHLY_LIMIT:
            return (
                f"Sakura monthly request limit reached "
                f"({count}/{SAKURA_MONTHLY_LIMIT} for {self._month_key()}). "
                f"No further calls will be made."
            )
        return None

    def increment(self) -> int:
        """Increment the counter and return the new monthly total."""
        monthly = self._load()
        key = self._month_key()
        monthly[key] = monthly.get(key, 0) + 1
        self._save(monthly)
        return monthly[key]


class SakuraProvider(ProviderBase):
    """Sakura Internet AI Engine via OpenAI-compatible Chat Completions API."""
    name = "sakura"
    BASE_URL = "https://api.ai.sakura.ad.jp/v1"

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key, base_url=self.BASE_URL)
        self.usage = SakuraUsageTracker()

    def send(
        self,
        messages: List[Dict[str, str]],
        model: str,
        state: SessionState,
        max_output_tokens: Optional[int] = None,
        use_tools: bool = True,
        on_tool_start: Optional[Callable[[str, Dict], None]] = None,
        on_tool_result: Optional[Callable] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_think_draft: Optional[Callable[[str], None]] = None,
        on_reasoning_content: Optional[Callable[[str], None]] = None,
    ) -> ReplyBundle:
        err = self.usage.check_limit()
        if err:
            raise RuntimeError(err)
        self.usage.increment()

        is_qwen3 = "Qwen3" in model
        effort_raw = (state.reasoning_effort or "auto").strip().lower()
        extra_create_kwargs: Dict[str, Any] = {}
        asst_msg_builder = None

        if is_qwen3 and not use_tools and effort_raw != "none":
            # Qwen3 hybrid thinking: enable_thinking toggle (incompatible with tool calling)
            extra_create_kwargs["extra_body"] = {"enable_thinking": True}

        elif not is_qwen3:
            # gpt-oss-120b: reasoning_effort passed via extra_body
            # Map CLI levels to Sakura's low/medium/high; "none" → "low" (minimum reasoning)
            effort_map = {
                "none": "low", "auto": "medium", "low": "low",
                "medium": "medium", "high": "high", "xhigh": "high",
            }  # auto → medium: avoid over-thinking on most queries
            sakura_effort = effort_map.get(effort_raw, "medium")
            extra_create_kwargs["extra_body"] = {"reasoning_effort": sakura_effort}

            if use_tools:
                # Preserve reasoning_content in tool-loop messages to avoid API errors
                def _oss_asst_msg_builder(assistant_msg):
                    asst_dict: Dict[str, Any] = {
                        "role": "assistant",
                        "content": assistant_msg.content or "",
                        "tool_calls": [tc.model_dump() for tc in assistant_msg.tool_calls],
                    }
                    reasoning = getattr(assistant_msg, "reasoning_content", None)
                    if reasoning:
                        asst_dict["reasoning_content"] = reasoning
                    return asst_dict
                asst_msg_builder = _oss_asst_msg_builder

        return run_research_pipeline(
            client=self.client,
            messages=messages,
            model=model,
            state=state,
            max_output_tokens=max_output_tokens,
            supports_tools=use_tools,
            on_tool_start=on_tool_start,
            on_tool_result=on_tool_result,
            on_status=on_status,
            on_think_draft=on_think_draft,
            on_reasoning_content=on_reasoning_content,
            extra_create_kwargs=extra_create_kwargs if extra_create_kwargs else None,
            asst_msg_builder=asst_msg_builder,
        )


# ----------------------------
# DeepSeek (Chat Completions API + agentic tool loop for deepseek-chat)
# ----------------------------

class DeepSeekProvider(ProviderBase):
    name = "deepseek"

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def send(
        self,
        messages: List[Dict[str, str]],
        model: str,
        state: SessionState,
        max_output_tokens: Optional[int] = None,
        use_tools: bool = True,
        on_tool_start: Optional[Callable[[str, Dict], None]] = None,
        on_tool_result: Optional[Callable] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_think_draft: Optional[Callable[[str], None]] = None,
        on_reasoning_content: Optional[Callable[[str], None]] = None,
    ) -> ReplyBundle:
        supports_tools = use_tools and model == "deepseek-chat"

        return run_research_pipeline(
            client=self.client,
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
            post_process=_clean_dsml_artifacts,
        )


# ----------------------------
# Kimi (Moonshot AI — $web_search builtin + agentic custom tools)
# ----------------------------

class KimiProvider(ProviderBase):
    """
    Moonshot AI (Kimi) provider.
    - Custom agentic tools only (search_papers, read_paper, web_search, read_webpage)
    - $web_search builtin intentionally disabled for fair comparison with other providers
    - Reasoning effort: 'none' -> instant mode, anything else -> thinking enabled
    """
    name = "kimi"

    def __init__(self, api_key: str, base_url: str = "https://api.moonshot.ai/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def send(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        state: SessionState,
        max_output_tokens: Optional[int] = None,
        use_tools: bool = True,
        on_tool_start: Optional[Callable[[str, Dict], None]] = None,
        on_tool_result: Optional[Callable] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_think_draft: Optional[Callable[[str], None]] = None,
        on_reasoning_content: Optional[Callable[[str], None]] = None,
    ) -> ReplyBundle:
        thinking_disabled = (state.reasoning_effort or "").strip().lower() == "none"
        if use_tools:
            thinking_disabled = True

        extra_create_kwargs: Dict[str, Any] = {}
        if thinking_disabled:
            extra_create_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            extra_create_kwargs["temperature"] = 0.6
            extra_create_kwargs["top_p"] = 0.95
        else:
            extra_create_kwargs["temperature"] = 1.0
            extra_create_kwargs["top_p"] = 0.95

        kimi_tools: Optional[List[Dict[str, Any]]] = None
        if use_tools:
            kimi_tools = list(AGENTIC_TOOLS)  # $web_search builtin intentionally excluded

        def _kimi_asst_msg_builder(assistant_msg, thinking_off=thinking_disabled):
            asst_dict: Dict[str, Any] = {
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": [tc.model_dump() for tc in assistant_msg.tool_calls],
            }
            if not thinking_off:
                reasoning = getattr(assistant_msg, "reasoning_content", None)
                asst_dict["reasoning_content"] = reasoning or ""
            return asst_dict

        return run_research_pipeline(
            client=self.client,
            messages=messages,
            model=model,
            state=state,
            max_output_tokens=max_output_tokens,
            supports_tools=use_tools,
            on_tool_start=on_tool_start,
            on_tool_result=on_tool_result,
            on_status=on_status,
            on_think_draft=on_think_draft,
            on_reasoning_content=on_reasoning_content,
            post_process=extract_kimi_final,
            extra_create_kwargs=extra_create_kwargs,
            override_tools=kimi_tools,
            asst_msg_builder=_kimi_asst_msg_builder,
        )


# ----------------------------
# Together AI (OpenAI-compatible Chat Completions)
# ----------------------------

class TogetherProvider(ProviderBase):
    """Together AI via OpenAI-compatible Chat Completions API.

    Models are stored by short display name in settings.json (e.g. 'GLM-5').
    The provider resolves each short name to the full Together model ID
    (e.g. 'zai-org/GLM-5') before making API calls.

    For unknown models the user may pass the full 'org/model' string directly;
    a warning is already shown by cmd_model in scicli.py.
    """
    name = "togetherai"
    BASE_URL = "https://api.together.xyz/v1"

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key, base_url=self.BASE_URL)

    def _resolve_model_id(self, model: str) -> str:
        """Map short display name to Together AI full model ID.

        Falls back to model as-is if:
        - it already contains '/' (user typed full org/model)
        - it has no full_id entry in settings (unknown model)
        """
        full_id = get_model_full_id("togetherai", model)
        return full_id  # returns model unchanged if no full_id found

    def send(
        self,
        messages: List[Dict[str, str]],
        model: str,
        state: SessionState,
        max_output_tokens: Optional[int] = None,
        use_tools: bool = True,
        on_tool_start: Optional[Callable[[str, Dict], None]] = None,
        on_tool_result: Optional[Callable] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_think_draft: Optional[Callable[[str], None]] = None,
        on_reasoning_content: Optional[Callable[[str], None]] = None,
    ) -> ReplyBundle:
        full_model_id = self._resolve_model_id(model)

        # Some Together AI models reject multiple system messages in the same
        # conversation (e.g. Qwen3.5).  For those, inject loop guidance as
        # role "user" instead of role "system".
        _MULTI_SYSTEM_OK = {"openai/gpt-oss-20b", "openai/gpt-oss-120b", "zai-org/GLM-5"}
        guidance_role = "system" if full_model_id in _MULTI_SYSTEM_OK else "user"

        return run_research_pipeline(
            client=self.client,
            messages=messages,
            model=full_model_id,
            state=state,
            max_output_tokens=max_output_tokens,
            supports_tools=use_tools,
            on_tool_start=on_tool_start,
            on_tool_result=on_tool_result,
            on_status=on_status,
            on_think_draft=on_think_draft,
            on_reasoning_content=on_reasoning_content,
            post_process=_clean_together_artifacts,
            guidance_role=guidance_role,
        )
