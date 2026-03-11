"""
utils.py — Shared utilities for Chat CLI.

Includes: small helpers, rendering, file/URL ingestion, Semantic Scholar helpers,
BibTeX generation, and Brave Search standalone function.
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import mimetypes
import datetime as dt
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import requests
from bs4 import BeautifulSoup

try:
    from markdownify import markdownify as html_to_md
except Exception:
    html_to_md = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import PyPDF2  # type: ignore
except Exception:
    PyPDF2 = None

try:
    import docx  # type: ignore
except Exception:
    docx = None

try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None

try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape as _rich_escape
from rich.syntax import Syntax
from rich.table import Table as RichTable
from rich import box as rbox

from config import (
    UPLOAD_DIR, CONV_DIR, STATS_FILE,
    BRAVE_ENDPOINT, DEFAULT_BRAVE_COUNT,
    S2_ENDPOINT_SEARCH, S2_ENDPOINT_PAPER,
    S2_ENDPOINT_CITATIONS, S2_ENDPOINT_REFERENCES,
    DEFAULT_S2_LIMIT, DEFAULT_S2_LIMIT_MAX,
    MIN_TEXT_CHARS_TO_ASSUME_NOT_SCANNED, DEFAULT_OCR_DPI_SCALE,
    DEFAULT_MAX_FILE_BYTES,
    SessionState, AppStats, ReplyBundle,
)


# ----------------------------
# Retry decorator for API calls
# ----------------------------

def retry_api_call(func, max_retries=3, base_delay=1.0):
    """Simple exponential backoff for API calls.

    Retries on transient network and HTTP errors (429, 5xx).
    Non-retryable errors (4xx except 429) are raised immediately.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return func()
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, 'status_code', 0) if hasattr(e, 'response') else 0
            if status and 400 <= status < 500 and status != 429:
                raise  # client error, don't retry
            last_exc = e
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exc = e
        if attempt < max_retries - 1:
            time.sleep(base_delay * (2 ** attempt))
    raise last_exc  # type: ignore[misc]


# ----------------------------
# Small helpers
# ----------------------------

def extract_kimi_final(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    m = re.search(r"(?is)\[Response\]\s*(.*)\Z", t)
    if m:
        return m.group(1).strip()
    t = re.sub(r"(?is)\A\s*\[Thinking\].*?(?=\n\s*\[|\Z)", "", t).strip()
    t = re.sub(r"(?im)^\s*\[Thinking\].*$", "", t).strip()
    return t


def format_model_list(models: List[str]) -> str:
    return " | ".join(models) if models else "(none)"


def now_ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def safe_write_json(path: Path, data: Any) -> None:
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def increment_stat(service: str) -> int:
    """Increment the monthly usage counter for a service in the stats file.

    The stats file (.scicli_stats.json) stores per-service monthly counts:
      {"brave": {"2026-03": 42}, "openai": {"2026-03": 8}, ...}

    Thread-safety: not needed (single-user CLI).
    """
    data = safe_read_json(STATS_FILE, {})
    month = dt.datetime.now().strftime("%Y-%m")
    svc = data.setdefault(service, {})
    svc[month] = svc.get(month, 0) + 1
    data["last_updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    safe_write_json(STATS_FILE, data)
    return svc[month]


def estimate_tokens_fallback(text: str) -> int:
    text = text or ""
    return max(1, int(len(text) / 4))


def estimate_message_tokens(messages: List[Dict[str, str]], model: str) -> int:
    joined = ""
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        joined += f"[{role}]\n{content}\n"

    if tiktoken is None:
        return estimate_tokens_fallback(joined)

    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("o200k_base")

    return len(enc.encode(joined))


def ensure_dirs() -> None:
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def truncate_text(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n\n…(truncated)…"


def to_plain_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def normalize_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    u = u.split("#", 1)[0]
    try:
        parts = urlsplit(u)
        qs = [(k, v) for (k, v) in parse_qsl(parts.query, keep_blank_values=True)
              if not (k.lower().startswith("utm_") or k.lower() in ("ref", "source"))]
        new_query = urlencode(qs, doseq=True)
        u = urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, ""))
    except Exception:
        pass
    if u.endswith("/"):
        u = u[:-1]
    return u


def dedup_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    best_title: Dict[str, str] = {}
    order: List[str] = []

    for title, url in pairs:
        nu = normalize_url(url)
        if not nu:
            continue
        if nu not in seen:
            seen.add(nu)
            order.append(nu)
            best_title[nu] = (title or "").strip()
        else:
            if not best_title.get(nu) and (title or "").strip():
                best_title[nu] = (title or "").strip()

    return [(best_title.get(nu, ""), nu) for nu in order]


def is_url(s: str) -> bool:
    s = s.strip().lower()
    return s.startswith("http://") or s.startswith("https://")


# Curated academic domain whitelist — single canonical source used by:
#   is_academic_url()  — classify search results as academic
#   ACADEMIC_GOGGLE     — "academic" search mode (Brave Goggle re-ranking)
ACADEMIC_DOMAINS: frozenset = frozenset({
    # Preprint servers
    "arxiv.org", "biorxiv.org", "medrxiv.org", "chemrxiv.org",
    "edarxiv.org", "engrxiv.org", "essoar.org", "osf.io",
    "psyarxiv.com", "socarxiv.org", "techrxiv.org",
    # Major journals / publishers
    "nature.com", "science.org", "cell.com", "sciencedirect.com",
    "thelancet.com", "bmj.com", "pnas.org", "plos.org",
    "elifesciences.org", "embopress.org", "jbc.org",
    "bloodjournal.org", "haematologica.org",
    # Publisher portals
    "springer.com", "link.springer.com", "springerlink.com",
    "wiley.com", "onlinelibrary.wiley.com",
    "tandfonline.com", "informa.com",
    "elsevier.com", "academic.oup.com", "oup.com",
    "cambridge.org", "royalsocietypublishing.org",
    "sagepub.com", "journals.sagepub.com",
    "karger.com", "liebertpub.com",
    # Aggregators / indices
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "nih.gov",
    "semanticscholar.org", "europepmc.org", "openalex.org",
    "crossref.org", "doi.org", "unpaywall.org",
    "researchgate.net", "academia.edu",
    # Society / discipline journals
    "mdpi.com", "frontiersin.org", "hindawi.com",
    "acs.org", "pubs.acs.org", "rsc.org", "pubs.rsc.org",
    "ieeexplore.ieee.org", "dl.acm.org", "acm.org",
    "jmlr.org", "proceedings.mlr.press", "openreview.net",
    "neurips.cc", "icml.cc", "aclanthology.org",
    "iopsciencei.iop.org", "iopscience.iop.org",
    "physicsoflife.pl", "aps.org", "journals.aps.org",
    "ascelibrary.org", "aiaa.org",
    "biochemj.org", "biochemistry.org",
    "jneurosci.org", "neurology.org",
    "ahajournals.org", "asa.org",
    "asm.org", "journals.asm.org",
    "genetics.org", "g3journal.org", "peerj.com",
    "f1000research.com", "wellcomeopenresearch.org",
    # Repositories
    "zenodo.org", "figshare.com", "dryad.datadryad.org",
})

# Inline Brave Goggle string for "academic" mode — boosts academic domains in ranking.
ACADEMIC_GOGGLE: str = "\n".join(f"$boost=3,site={d}" for d in sorted(ACADEMIC_DOMAINS))

# Private alias kept for internal use within this module
_ACADEMIC_DOMAINS = ACADEMIC_DOMAINS

# URL path fragments that strongly suggest an academic paper regardless of domain
_ACADEMIC_PATH_FRAGMENTS: tuple = (
    "/doi/", "/abs/", "/paper/", "/article/",
    "/fulltext/", "/full-text/", "/pdfdownload/",
    "/content/", "/research/",
)


def is_academic_url(url: str) -> bool:
    """Return True if a URL likely points to an academic paper or preprint.

    Uses a curated domain whitelist (~100 entries) supplemented by URL path
    heuristics (/doi/, /abs/, DOI patterns) and .pdf extension detection.
    """
    u = (url or "").strip().lower()
    if not u:
        return False

    # Check domain whitelist
    try:
        from urllib.parse import urlsplit
        netloc = urlsplit(u).netloc
        # Strip www. prefix for comparison
        domain = netloc.lstrip("www.")
        # Match exact domain or any parent domain in the whitelist
        if domain in _ACADEMIC_DOMAINS:
            return True
        # Check if domain ends with a whitelisted domain (e.g. "journals.plos.org" matches "plos.org")
        if any(domain == d or domain.endswith("." + d) for d in _ACADEMIC_DOMAINS):
            return True
    except Exception:
        pass

    # Path-based heuristics
    if any(frag in u for frag in _ACADEMIC_PATH_FRAGMENTS):
        return True
    if u.endswith(".pdf"):
        return True
    if "/10." in u and ("doi" in u or "dx." in u):
        return True  # DOI redirect pattern

    # .edu domain with academic path patterns
    if ".edu/" in u and any(frag in u for frag in ("/paper", "/abs", "/pdf", "/research", "/pub")):
        return True

    return False


def parse_index_spec(spec: str, max_index: int) -> List[int]:
    spec = spec.strip()
    if not spec:
        return []
    indices: List[int] = []
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a = a.strip()
            b = b.strip()
            if not a.isdigit() or not b.isdigit():
                continue
            start = int(a)
            end = int(b)
            if start > end:
                start, end = end, start
            for k in range(start, end + 1):
                if 1 <= k <= max_index:
                    indices.append(k)
        else:
            if p.isdigit():
                k = int(p)
                if 1 <= k <= max_index:
                    indices.append(k)
    seen = set()
    out: List[int] = []
    for k in indices:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def parse_save_args_new(arg: str) -> Tuple[Optional[int], Optional[str]]:
    arg = (arg or "").strip()
    if not arg:
        return None, None

    parts = arg.split(maxsplit=1)
    first = parts[0].strip()

    if re.fullmatch(r"-?\d+", first):
        idx = int(first)
        rest = parts[1].strip() if len(parts) > 1 else None
        return idx, rest

    return None, arg


def safe_filename_from_url(url: str) -> str:
    name = re.sub(r"^https?://", "", url.strip(), flags=re.I)
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return (name[:120] or "web") + ".bin"


# ----------------------------
# Rendering
# ----------------------------

_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+\-]*)\n(.*?)\n```", re.DOTALL)
_REFS_MARKER = "\x00REFS\x00"
_REF_LINE_MARKER = "\x00REF\x00"
_CITE_INLINE_PAT = re.compile(r'\*\*\[([^\]]+)\]\*\*')
_BLOCK_ELEMENT_RE = re.compile(
    r'^\s*(?:#+|\|[^\n]+\||\d+\.|[-*])\s', re.MULTILINE
)  # Detect headings (any level ##, ###…), tables, lists — block-level prose segment

_HEADING_RE = re.compile(r'(?m)^(#{1,6})\s+(.+)$')

# LaTeX math region patterns
_MATH_DISPLAY_RE = re.compile(r'\\\[(.+?)\\\]|\$\$(.+?)\$\$', re.DOTALL)
_MATH_INLINE_RE  = re.compile(r'\\\((.+?)\\\)|\$([^$\n]+?)\$')


def _math_unicode_fallback(src: str) -> str:
    """Best-effort conversion of LaTeX math to Unicode readable text."""
    r = src.strip()
    # Multi-pass structural substitutions handle nesting (e.g. \hat{\mathbf{r}})
    for _ in range(3):
        r = re.sub(r'\\[td]?frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', r)
        r = re.sub(r'\\sqrt\{([^{}]*)\}', r'√(\1)', r)
        r = re.sub(
            r'\\(?:mathbf|mathrm|mathit|mathcal|mathbb|boldsymbol|text|operatorname)'
            r'\{([^{}]*)\}', r'\1', r)
        r = re.sub(r'\\hat\{([^{}]*)\}',      lambda m: m.group(1) + '\u0302', r)
        r = re.sub(r'\\vec\{([^{}]*)\}',      lambda m: m.group(1) + '\u20d7', r)
        r = re.sub(r'\\tilde\{([^{}]*)\}',    lambda m: m.group(1) + '\u0303', r)
        r = re.sub(r'\\overline\{([^{}]*)\}', lambda m: m.group(1) + '\u0305', r)
        r = re.sub(r'\\ddot\{([^{}]*)\}',    lambda m: m.group(1) + '\u0308', r)
        r = re.sub(r'\\dot\{([^{}]*)\}',     lambda m: m.group(1) + '\u0307', r)
        # Without braces: single-letter argument (e.g. \ddot r, \dot r)
        r = re.sub(r'\\ddot([a-zA-Z])', lambda m: m.group(1) + '\u0308', r)
        r = re.sub(r'\\dot([a-zA-Z])',  lambda m: m.group(1) + '\u0307', r)
        r = re.sub(r'_\{([^{}]*)\}',  r'_\1', r)
        r = re.sub(r'\^\{([^{}]*)\}', r'^\1', r)
    # Greek letters
    _GREEK = {
        'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ',
        'epsilon': 'ε', 'varepsilon': 'ε', 'zeta': 'ζ', 'eta': 'η',
        'theta': 'θ', 'vartheta': 'θ', 'iota': 'ι', 'kappa': 'κ',
        'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ',
        'pi': 'π', 'varpi': 'π', 'rho': 'ρ', 'varrho': 'ρ',
        'sigma': 'σ', 'varsigma': 'ς', 'tau': 'τ', 'upsilon': 'υ',
        'phi': 'φ', 'varphi': 'φ', 'chi': 'χ', 'psi': 'ψ', 'omega': 'ω',
        'Gamma': 'Γ', 'Delta': 'Δ', 'Theta': 'Θ', 'Lambda': 'Λ',
        'Xi': 'Ξ', 'Pi': 'Π', 'Sigma': 'Σ', 'Upsilon': 'Υ',
        'Phi': 'Φ', 'Psi': 'Ψ', 'Omega': 'Ω',
    }
    for name, ch in _GREEK.items():
        r = r.replace(f'\\{name}', ch)
    # Operators and symbols (order matters — longer tokens first)
    # Use (?![a-zA-Z]) instead of \b so patterns match before digits too (e.g. \approx6.674)
    _B = r'(?![a-zA-Z])'
    _OPS = [
        (rf'\\displaystyle{_B}\s*', ''), (rf'\\textstyle{_B}\s*', ''), (rf'\\scriptstyle{_B}\s*', ''),
        (rf'\\sum{_B}', '∑'), (rf'\\prod{_B}', '∏'),
        (rf'\\int{_B}', '∫'), (rf'\\oint{_B}', '∮'),
        (rf'\\partial{_B}', '∂'), (rf'\\nabla{_B}', '∇'),
        (rf'\\infty{_B}', '∞'), (rf'\\hbar{_B}', 'ℏ'), (rf'\\ell{_B}', 'ℓ'),
        (rf'\\pm{_B}', '±'), (rf'\\mp{_B}', '∓'), (rf'\\times{_B}', '×'),
        (rf'\\div{_B}', '÷'), (rf'\\cdot{_B}', '·'), (rf'\\circ{_B}', '∘'),
        (rf'\\leq{_B}', '≤'), (rf'\\geq{_B}', '≥'), (rf'\\neq{_B}', '≠'),
        (rf'\\ll{_B}', '≪'), (rf'\\gg{_B}', '≫'),
        (rf'\\approx{_B}', '≈'), (rf'\\sim{_B}', '∼'), (rf'\\simeq{_B}', '≃'),
        (rf'\\equiv{_B}', '≡'), (rf'\\propto{_B}', '∝'),
        (rf'\\Leftrightarrow{_B}', '⟺'), (rf'\\Rightarrow{_B}', '⟹'), (rf'\\Leftarrow{_B}', '⟸'),
        (rf'\\leftrightarrow{_B}', '↔'), (rf'\\rightarrow{_B}', '→'), (rf'\\leftarrow{_B}', '←'),
        (rf'\\to{_B}', '→'), (rf'\\gets{_B}', '←'),
        (rf'\\notin{_B}', '∉'), (rf'\\in{_B}', '∈'),   # notin before in
        (rf'\\subset{_B}', '⊂'), (rf'\\supset{_B}', '⊃'),
        (rf'\\cup{_B}', '∪'), (rf'\\cap{_B}', '∩'),
        (rf'\\forall{_B}', '∀'), (rf'\\exists{_B}', '∃'),
        (rf'\\langle{_B}', '⟨'), (rf'\\rangle{_B}', '⟩'),
        (rf'\\lVert{_B}', '‖'), (rf'\\rVert{_B}', '‖'),
        (rf'\\ldots{_B}', '…'), (rf'\\cdots{_B}', '⋯'), (rf'\\vdots{_B}', '⋮'),
        (r'\\[bB]igg?[lr]?\s*', ''), (r'\\middle\s*', ''),
        (r'\\left\s*', ''), (r'\\right\s*', ''),
        (r'\\[,;: ]\s*', ' '), (r'\\!\s*', ''),
        (rf'\\quad{_B}', '  '), (rf'\\qquad{_B}', '    '),
        (r'\\\\', ' '),
        (r'\\\{', '{'), (r'\\\}', '}'),
    ]
    for pat, rep in _OPS:
        r = re.sub(pat, rep, r)
    # Strip remaining bare braces (LaTeX grouping)
    r = r.replace('{', '').replace('}', '')
    return re.sub(r' {2,}', ' ', r).strip()


def _render_math(src: str, display: bool) -> str:
    """Render a LaTeX math expression to Unicode. Tries sympy first, falls back."""
    src = src.strip()
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            from sympy.parsing.latex import parse_latex
            from sympy import pretty
            expr = parse_latex(src)
            rendered = pretty(expr, use_unicode=True)
        if display:
            return '\n' + '\n'.join('  ' + l for l in rendered.splitlines()) + '\n'
        return rendered
    except Exception:
        result = _math_unicode_fallback(src)
        if display:
            return '\n  ' + result + '\n'
        return result


def _preprocess_math(text: str) -> str:
    """Replace LaTeX math delimiters with rendered Unicode equivalents."""
    text = _MATH_DISPLAY_RE.sub(
        lambda m: _render_math(m.group(1) or m.group(2) or '', display=True), text)
    text = _MATH_INLINE_RE.sub(
        lambda m: _render_math(m.group(1) or m.group(2) or '', display=False), text)
    return text


def _render_heading(console: Console, level: int, title: str) -> None:
    """Render a Markdown heading left-aligned (Rich centers headings; we override)."""
    safe = _md_inline_to_rich(title)
    console.print()
    if level == 1:
        console.print(f"[bold]{safe}[/bold]")
        console.rule(style="dim")
    elif level == 2:
        console.print(f"[bold]{safe}[/bold]")
    else:
        console.print(f"[bold]{safe}[/bold]")


def _md_inline_to_rich(text: str) -> str:
    """Convert Markdown inline formatting to Rich markup for short inline fragments.

    Escapes raw [ ] first so stray brackets in the text don't confuse Rich's
    markup parser, then re-applies bold/italic/code conversions.
    """
    text = _rich_escape(text)   # [ ] → \[ \]
    text = re.sub(r'\*\*(.+?)\*\*', r'[bold]\1[/bold]', text, flags=re.DOTALL)
    text = re.sub(r'\*([^*\n]+?)\*', r'[italic]\1[/italic]', text)
    text = re.sub(r'`([^`\n]+)`', r'[dim]\1[/dim]', text)
    return text

# Matches a full markdown pipe table: header row, separator row, one+ data rows.
# Rich's Markdown class does not render GFM tables; we handle them ourselves.
_TABLE_RE = re.compile(
    r'(?m)^'
    r'(\|[^\n]+\n'          # header row
    r'\|[ |:\-]+\n'         # separator row  (only |, -, :, space)
    r'(?:\|[^\n]*\n?)+)',   # one or more data rows
)


def _parse_table_row(line: str) -> List[str]:
    """Extract cell strings from a pipe-delimited table row."""
    return [c.strip() for c in line.strip().strip('|').split('|')]


def _render_md_table(console: Console, raw: str) -> None:
    """Render a GFM pipe table using Rich's native Table class."""
    lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
    if len(lines) < 3:
        try:
            console.print(Markdown(raw))
        except Exception:
            sys.stdout.write(raw + '\n')
        return

    headers = _parse_table_row(lines[0])
    sep_cells = _parse_table_row(lines[1])

    def _align(cell: str) -> str:
        c = cell.strip()
        if c.startswith(':') and c.endswith(':'):
            return 'center'
        if c.endswith(':'):
            return 'right'
        return 'left'

    tbl = RichTable(show_header=True, header_style="bold",
                    box=rbox.SIMPLE_HEAD, pad_edge=False)
    for i, h in enumerate(headers):
        justify = _align(sep_cells[i]) if i < len(sep_cells) else 'left'
        tbl.add_column(_md_inline_to_rich(h), justify=justify, overflow='fold')

    for line in lines[2:]:
        stripped = line.strip()
        if not stripped or re.match(r'^[\|: \-]+$', stripped):
            continue
        cells = _parse_table_row(line)
        while len(cells) < len(headers):
            cells.append('')
        tbl.add_row(*[_md_inline_to_rich(c) for c in cells[:len(headers)]])

    console.print(tbl)


def _render_prose_block(console: Console, text: str, code_theme: str) -> None:
    """Render a prose block (no tables) with heading lines rendered left-aligned."""
    pos = 0
    for m in _HEADING_RE.finditer(text):
        before = text[pos:m.start()].strip()
        if before:
            try:
                console.print(Markdown(before, code_theme=code_theme))
            except Exception:
                sys.stdout.write(before + '\n')
        _render_heading(console, len(m.group(1)), m.group(2).strip())
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        try:
            console.print(Markdown(tail, code_theme=code_theme))
        except Exception:
            sys.stdout.write(tail + '\n')


def _render_text_with_tables(console: Console, text: str, code_theme: str) -> None:
    """Render prose that may contain GFM pipe tables and Markdown headings."""
    pos = 0
    for m in _TABLE_RE.finditer(text):
        before = text[pos:m.start()].strip()
        if before:
            _render_prose_block(console, before, code_theme)
        _render_md_table(console, m.group(1))
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        _render_prose_block(console, tail, code_theme)


def _render_prose(console: Console, text: str, code_theme: str,
                  citation_color: str = "cyan") -> None:
    """Render a prose segment (no code fences) through Rich Markdown.

    GFM pipe tables are rendered via Rich's Table class (Rich's Markdown renderer
    does not support them). Citation markers **[N]** are rendered with citation_color.
    """
    if not text.strip():
        return

    parts = _CITE_INLINE_PAT.split(text)

    if len(parts) == 1:
        # No inline citations — use table-aware renderer
        _render_text_with_tables(console, text, code_theme)
        return

    # Text has inline citations — render prose pieces and citation labels.
    # Block-level prose (tables, headings, lists) gets full block rendering.
    # Pure inline prose is rendered as Rich markup (end="") so citations stay
    # on the same line as surrounding text.
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                if _BLOCK_ELEMENT_RE.search(part) or _TABLE_RE.search(part):
                    # Block content: render with full table-aware Markdown
                    _render_text_with_tables(console, part, code_theme)
                else:
                    # Inline text: convert Markdown formatting to Rich markup
                    # and print without trailing newline to preserve flow
                    try:
                        console.print(_md_inline_to_rich(part.rstrip()), end="", markup=True)
                    except Exception:
                        sys.stdout.write(part)
        else:
            # Citation labels e.g. "1" or "1, 2" or "Smith, Nature, 2024"
            console.print(
                f"[bold {citation_color}][{part}][/bold {citation_color}]",
                end="",
            )
    console.print()  # final newline after inline citation sequence


def _render_refs_section(console: Console, refs_text: str,
                         citation_color: str = "cyan") -> None:
    """Render the structured References section with colored citation numbers."""
    console.print()
    console.print(f"[bold {citation_color}]References[/bold {citation_color}]")
    for line in refs_text.strip().splitlines():
        if line.startswith(_REF_LINE_MARKER):
            # Format: \x00REF\x00[label]\x00rest_of_line
            rest = line[len(_REF_LINE_MARKER):]
            idx = rest.find("\x00")
            if idx >= 0:
                num_part = rest[:idx]     # e.g. "[1]"
                text_part = rest[idx+1:]  # e.g. "Authors. (Year). ..."
                console.print(
                    f"[bold {citation_color}]{_rich_escape(num_part)}[/bold {citation_color}]"
                    f" [dim]{_rich_escape(text_part)}[/dim]"
                )
            else:
                console.print(f"[dim]{rest}[/dim]")


def render_assistant(console: Console, text: str, codecolor: bool = True,
                     code_theme: str = "monokai",
                     citation_color: str = "cyan",
                     citation_style: str = "numbered") -> None:
    """Render an assistant response.

    codecolor=True (default): prose via Rich Markdown, each code block preceded by a
      dim rule showing snippet number and language. Use /save N to extract code.
    codecolor=False: raw text to stdout (terminal copy-safe fallback).

    A structured References section (delimited by \\x00REFS\\x00 / \\x00REF\\x00)
    is rendered separately with citation_color highlighting, bypassing Markdown.
    """
    if not text:
        return

    # Split off References section before any rendering
    refs_section = ""
    if _REFS_MARKER in text:
        text, refs_section = text.split(_REFS_MARKER, 1)

    # Pandoc display: wrap [@key] as inline code for visual distinction in the terminal.
    # last_assistant_text (set before this call) retains the plain [@key] for copy-paste.
    if citation_style == "pandoc":
        text = re.sub(
            r'\[(@[A-Za-z][A-Za-z0-9_.]*(?:\s*[;,]\s*@[A-Za-z][A-Za-z0-9_.]*)*)\]',
            lambda m: f'`{m.group(0)}`',
            text,
        )

    if not codecolor:
        # Raw mode: strip the special markers and write plain text
        sys.stdout.write(text)
        if refs_section.strip():
            sys.stdout.write("\nReferences\n")
            for line in refs_section.strip().splitlines():
                if line.startswith(_REF_LINE_MARKER):
                    rest = line[len(_REF_LINE_MARKER):]
                    idx = rest.find("\x00")
                    sys.stdout.write(f"{rest[:idx]} {rest[idx+1:]}\n" if idx >= 0 else f"{rest}\n")
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        return

    pos = 0
    snippet_idx = 0

    for m in _FENCE_RE.finditer(text):
        before = text[pos:m.start()].strip()
        if before:
            _render_prose(console, _preprocess_math(before), code_theme, citation_color=citation_color)

        lang = (m.group(1) or "").strip() or "text"
        code = m.group(2) or ""
        snippet_idx += 1

        console.print()
        console.print(f"[dim]snippet #{snippet_idx} · {lang}[/dim]")
        try:
            console.print(Syntax(code, lang, line_numbers=False, word_wrap=False,
                                 theme=code_theme))
        except Exception:
            sys.stdout.write(code + "\n")
        console.print()

        pos = m.end()

    # Render remaining prose after the last code block
    tail = text[pos:].strip()
    if tail:
        _render_prose(console, _preprocess_math(tail), code_theme, citation_color=citation_color)

    # Render References section with colored numbers
    if refs_section.strip():
        _render_refs_section(console, refs_section, citation_color=citation_color)


def print_sources(console: Console, bundle: ReplyBundle) -> None:
    has_old_style = bundle.cited or bundle.consulted
    has_details = bundle.source_details

    if not has_old_style and not has_details:
        return

    console.print()  # blank line before sources

    # If we have granular source_details, display categorised view
    if has_details:
        from config import SourceInfo  # local import to avoid circular at module level

        details = bundle.source_details or []

        # Categorise by access_level
        full_text = [s for s in details if s.access_level == "full_text"]
        abstract_only = [s for s in details if s.access_level == "abstract_only"]
        webpages = [s for s in details if s.access_level == "webpage"]
        search_results = [s for s in details if s.access_level == "search_result"]

        # Deduplicate within each category by URL
        def _dedup_sources(sources: list) -> list:
            seen: set = set()
            out = []
            for s in sources:
                key = normalize_url(s.url)
                if key and key not in seen:
                    seen.add(key)
                    out.append(s)
            return out

        idx = [1]  # mutable counter

        def _print_source(s, color: str = "blue") -> None:
            n = idx[0]
            idx[0] += 1
            author_year = ""
            authors = getattr(s, 'authors', '') or ''
            year = getattr(s, 'year', '') or ''
            if authors and year:
                author_year = f"{authors} ({year}) "
            elif year:
                author_year = f"({year}) "
            title = (s.title or "").strip()
            via = ""
            if s.tool_name == "read_paper" and s.access_level == "full_text":
                via = " [full text]"
            elif s.tool_name == "read_paper" and s.access_level == "abstract_only":
                via = " [abstract only]"
            console.print(f"  [{color}]\\[{n}] {author_year}\"{title}\"{via}[/{color}]")
            if s.url:
                console.print(f"      {s.url}", style="dim")

        full_text = _dedup_sources(full_text)
        abstract_only = _dedup_sources(abstract_only)
        webpages = _dedup_sources(webpages)

        if full_text:
            console.print("[bold blue]Papers read (full text):[/bold blue]")
            for s in full_text:
                _print_source(s, "blue")

        if abstract_only:
            console.print("[bold cyan]Papers consulted (abstract only):[/bold cyan]")
            for s in abstract_only:
                _print_source(s, "cyan")

        if webpages:
            console.print("[bold green]Web pages read:[/bold green]")
            for s in webpages:
                _print_source(s, "green")

        search_results = _dedup_sources(search_results)
        if search_results:
            console.print(f"[dim]({len(search_results)} additional search results consulted)[/dim]")

        snippets = [s for s in details if s.access_level == "snippet"]
        snippets = _dedup_sources(snippets)
        if snippets:
            console.print(f"[dim]({len(snippets)} search snippets)[/dim]")

        return

    # Fallback: old-style cited/consulted display (e.g. for OpenAI with native web search)
    def _print_list(label: str, items: List[Tuple[str, str]], color: str = "blue") -> None:
        if not items:
            return
        console.print(f"[bold {color}]{label}:[/bold {color}]")
        for i, (t, u) in enumerate(items, 1):
            title = (t or "").strip()
            url = (u or "").strip()
            if (not title) or (normalize_url(title) == normalize_url(url)) or is_url(title):
                console.print(f"  [{color}]\\[{i}] {url}[/{color}]")
            else:
                console.print(f"  [{color}]\\[{i}] {title}[/{color}]\n      {url}", style="dim")

    _print_list("Cited in answer", bundle.cited, "blue")
    if bundle.consulted:
        console.print()
        _print_list("Consulted by tool", bundle.consulted, "cyan")


def extract_code_snippets(text: str) -> List[Dict[str, str]]:
    fence_re = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)\n```", re.DOTALL)
    out: List[Dict[str, str]] = []
    for m in fence_re.finditer(text or ""):
        lang = (m.group(1) or "").strip() or "text"
        code = m.group(2) or ""
        out.append({"lang": lang, "code": code})
    return out


# ----------------------------
# File + URL ingestion
# ----------------------------

def _pdf_text_pymupdf(path: Path) -> str:
    assert fitz is not None
    doc = fitz.open(str(path))
    out: List[str] = []
    for page in doc:
        out.append(page.get_text("text") or "")
    return "\n".join(out).strip()


def _pdf_text_pypdf2(path: Path) -> str:
    assert PyPDF2 is not None
    reader = PyPDF2.PdfReader(str(path))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n\n".join(texts).strip()


def _pdf_ocr_pymupdf(path: Path, scale: float = DEFAULT_OCR_DPI_SCALE, max_pages: Optional[int] = None) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required for OCR rendering.")
    if pytesseract is None or Image is None:
        raise RuntimeError("OCR requires pytesseract and pillow. Install: pip install pytesseract pillow")

    doc = fitz.open(str(path))
    mat = fitz.Matrix(float(scale), float(scale))
    out: List[str] = []
    n_pages = len(doc)
    limit = n_pages if max_pages is None else min(n_pages, int(max_pages))

    for i in range(limit):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        txt = pytesseract.image_to_string(img)
        out.append(txt.strip())
    return "\n\n".join([t for t in out if t]).strip()


def process_pdf_to_text(path: Path, state: SessionState) -> Tuple[str, str]:
    note_parts: List[str] = []
    text = ""

    if fitz is not None:
        try:
            text = _pdf_text_pymupdf(path)
            note_parts.append("PDF extracted with PyMuPDF (fitz). Paragraph ordering may still be imperfect.")
        except Exception as e:
            note_parts.append(f"PyMuPDF extraction error: {e}")

    if not text and PyPDF2 is not None:
        try:
            text = _pdf_text_pypdf2(path)
            note_parts.append("PDF extracted with PyPDF2 fallback. Paragraph ordering can be inaccurate.")
        except Exception as e:
            note_parts.append(f"PyPDF2 extraction error: {e}")

    if (not text or len(text) < MIN_TEXT_CHARS_TO_ASSUME_NOT_SCANNED) and state.auto_ocr_on_scans:
        if fitz is not None and pytesseract is not None and Image is not None:
            try:
                ocr_text = _pdf_ocr_pymupdf(path, scale=state.ocr_scale)
                if ocr_text and len(ocr_text) > len(text):
                    text = ocr_text
                    note_parts.append("OCR was used (scan-like PDF). Some words may be incorrect.")
            except Exception as e:
                note_parts.append(f"OCR attempt failed: {e}")
        else:
            note_parts.append("OCR not available (install PyMuPDF + pytesseract + pillow).")

    note = " ".join([p for p in note_parts if p]).strip()
    return text, note


def process_file_to_text_ex(path: Path, state: SessionState, max_bytes: int = DEFAULT_MAX_FILE_BYTES) -> Tuple[str, str, str]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"File too large ({size/1024/1024:.1f} MB). Limit is {max_bytes/1024/1024:.1f} MB.")

    suffix = path.suffix.lower()
    display_name = path.name

    if suffix in [".txt", ".md", ".py", ".json", ".csv", ".log", ".yaml", ".yml"]:
        return display_name, path.read_text(encoding="utf-8", errors="replace"), ""

    if suffix == ".pdf":
        text, note = process_pdf_to_text(path, state=state)
        if not text:
            raise RuntimeError("Could not extract text from PDF (empty output).")
        return display_name, text, note

    if suffix == ".docx":
        if docx is None:
            raise RuntimeError("python-docx not installed. Run: pip install python-docx")
        d = docx.Document(str(path))
        return display_name, "\n".join(p.text for p in d.paragraphs), ""

    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("text/"):
        return display_name, path.read_text(encoding="utf-8", errors="replace"), ""

    raise ValueError(f"Unsupported file type: {suffix}. Try converting to .txt or install needed libraries.")


def download_url_to_file(url: str, dest: Path, timeout: int = 30) -> Tuple[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ChatCLI/1.0; +https://example.invalid)",
        "Accept": "*/*",
    }
    with requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        final_url = r.url
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 128):
                if chunk:
                    f.write(chunk)
    return final_url, ctype


def fetch_url_as_markdown_or_pdf_text(url: str, state: SessionState, timeout: int = 20) -> Tuple[str, str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ChatCLI/1.0; +https://example.invalid)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.7",
    }

    with requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        final_url = r.url

        if ("application/pdf" in ctype) or final_url.lower().endswith(".pdf"):
            tmp = UPLOAD_DIR / f"{now_ts()}_{safe_filename_from_url(final_url).replace('.bin', '.pdf')}"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        f.write(chunk)
            text, note = process_pdf_to_text(tmp, state=state)
            title = tmp.name
            note = (note + " Source: URL PDF.").strip()
            return title, text, note

        body = r.text

    soup = BeautifulSoup(body, "html.parser")
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    body_html = str(soup.body or soup)
    if html_to_md is not None:
        md = html_to_md(body_html, heading_style="ATX")
    else:
        md = soup.get_text("\n", strip=True)

    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return title or final_url, md, ""


# ----------------------------
# Semantic Scholar integration
# ----------------------------

def s2_headers() -> Dict[str, str]:
    key = (os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or "").strip()
    h = {"Accept": "application/json"}
    if key:
        h["x-api-key"] = key
    return h


def s2_search(query: str, limit: int = DEFAULT_S2_LIMIT, offset: int = 0) -> Dict[str, Any]:
    fields = ",".join([
        "paperId", "title", "authors", "year", "venue", "journal",
        "citationCount", "referenceCount", "url", "abstract", "tldr",
        "fieldsOfStudy", "externalIds", "isOpenAccess", "openAccessPdf",
    ])
    params = {
        "query": query,
        "limit": max(1, min(DEFAULT_S2_LIMIT_MAX, int(limit))),
        "offset": max(0, int(offset)),
        "fields": fields,
    }
    def _call():
        r = requests.get(S2_ENDPOINT_SEARCH, headers=s2_headers(), params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    return retry_api_call(_call)


def s2_paper(paper_id: str) -> Dict[str, Any]:
    fields = ",".join([
        "paperId", "title", "authors", "year", "venue", "journal",
        "citationCount", "referenceCount", "url", "abstract", "tldr",
        "fieldsOfStudy", "externalIds", "isOpenAccess", "openAccessPdf",
    ])
    url = S2_ENDPOINT_PAPER.format(paper_id)
    def _call():
        r = requests.get(url, headers=s2_headers(), params={"fields": fields}, timeout=30)
        r.raise_for_status()
        return r.json()
    return retry_api_call(_call)


def s2_citations(paper_id: str, limit: int = 100) -> List[Dict]:
    """Papers that cite the given paper. Returns list of citing paper dicts."""
    fields = ",".join([
        "paperId", "title", "authors", "year", "venue", "journal",
        "citationCount", "referenceCount", "url", "abstract",
        "isOpenAccess", "openAccessPdf",
    ])
    url = S2_ENDPOINT_CITATIONS.format(paper_id)
    params = {"fields": f"citingPaper.{fields}", "limit": min(1000, max(1, int(limit)))}
    def _call():
        r = requests.get(url, headers=s2_headers(), params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    data = retry_api_call(_call).get("data", [])
    return [item.get("citingPaper", item) for item in data if item.get("citingPaper")]


def s2_references(paper_id: str, limit: int = 100) -> List[Dict]:
    """Papers referenced by the given paper. Returns list of cited paper dicts."""
    fields = ",".join([
        "paperId", "title", "authors", "year", "venue", "journal",
        "citationCount", "referenceCount", "url", "abstract",
        "isOpenAccess", "openAccessPdf",
    ])
    url = S2_ENDPOINT_REFERENCES.format(paper_id)
    params = {"fields": f"citedPaper.{fields}", "limit": min(1000, max(1, int(limit)))}
    def _call():
        r = requests.get(url, headers=s2_headers(), params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    data = retry_api_call(_call).get("data", [])
    return [item.get("citedPaper", item) for item in data if item.get("citedPaper")]


def _author_list(p: Dict[str, Any], max_n: int = 3) -> str:
    authors = p.get("authors") or []
    names = [a.get("name", "").strip() for a in authors if a.get("name")]
    if not names:
        return ""
    if len(names) <= max_n:
        return ", ".join(names)
    return ", ".join(names[:max_n]) + f", …(+{len(names) - max_n})"


def _paper_venue(p: Dict[str, Any]) -> str:
    j = p.get("journal") or {}
    if isinstance(j, dict) and j.get("name"):
        return str(j.get("name"))
    v = p.get("venue")
    if v:
        return str(v)
    return ""


def _paper_doi(p: Dict[str, Any]) -> str:
    ext = p.get("externalIds") or {}
    if isinstance(ext, dict):
        return str(ext.get("DOI") or ext.get("doi") or "") or ""
    return ""


def _paper_pdf_url(p: Dict[str, Any]) -> str:
    oap = p.get("openAccessPdf")
    if isinstance(oap, dict):
        return str(oap.get("url") or "") or ""
    return ""


def _short_abstract(p: Dict[str, Any], n: int = 280) -> str:
    abs_ = p.get("abstract") or ""
    if not abs_ and isinstance(p.get("tldr"), dict):
        abs_ = p["tldr"].get("text") or ""
    abs_ = (abs_ or "").strip().replace("\n", " ")
    return abs_[:n] + ("…" if len(abs_) > n else "")


def build_s2_metadata_block(query: str, offset: int, total: Optional[int], papers: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    hdr = f"Semantic Scholar results (query={query!r}, offset={offset}, returned={len(papers)}"
    if total is not None:
        hdr += f", total={total}"
    hdr += ")\n"
    lines.append(hdr)

    for i, p in enumerate(papers, 1):
        pid = p.get("paperId", "")
        title = (p.get("title") or "").strip()
        year = p.get("year") or ""
        cites = p.get("citationCount")
        refs = p.get("referenceCount")
        venue = _paper_venue(p)
        authors = _author_list(p, max_n=6)
        doi = _paper_doi(p)
        url = p.get("url") or ""
        pdf = _paper_pdf_url(p)
        is_oa = p.get("isOpenAccess")
        fos = p.get("fieldsOfStudy") or []
        abs_short = _short_abstract(p, n=380)

        lines.append(f"[{offset + i}] {title}")
        lines.append(f"  paperId: {pid}")
        lines.append(f"  year: {year} | citations: {cites} | references: {refs}")
        if venue:
            lines.append(f"  venue/journal: {venue}")
        if authors:
            lines.append(f"  authors: {authors}")
        if doi:
            lines.append(f"  DOI: {doi}")
        if url:
            lines.append(f"  url: {url}")
        lines.append(f"  open_access: {bool(is_oa)}")
        if pdf:
            lines.append(f"  open_access_pdf: {pdf}")
        if fos:
            lines.append(f"  fieldsOfStudy: {', '.join(str(x) for x in fos[:8])}{'…' if len(fos) > 8 else ''}")
        if abs_short:
            lines.append(f"  abstract/tldr: {abs_short}")
        lines.append("")
    return "\n".join(lines).strip()


# ----------------------------
# Duplicate paper detection
# ----------------------------

def _normalize_title(title: str) -> str:
    """Normalize a paper title for comparison: lowercase, strip punctuation, collapse whitespace."""
    t = (title or "").lower()
    t = re.sub(r'[^a-z0-9\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def _title_similarity(a: str, b: str) -> float:
    """Simple word-overlap similarity between two normalized titles."""
    wa = set(a.split())
    wb = set(b.split())
    if not wa or not wb:
        return 0.0
    intersection = wa & wb
    union = wa | wb
    return len(intersection) / len(union)


def find_duplicate_paper(
    new_paper: Dict[str, Any],
    existing_sources: List[Any],
) -> Optional[Any]:
    """Check if a paper is a duplicate of an existing source (by DOI or title similarity > 0.85).

    Args:
        new_paper: dict with keys like 'title', 'externalIds'
        existing_sources: list of SourceInfo objects with title, paper_id fields

    Returns:
        The matching SourceInfo if duplicate found, else None
    """
    # Check by DOI
    new_ext = new_paper.get("externalIds") or {}
    new_doi = (new_ext.get("DOI") or new_ext.get("doi") or "").strip().lower()

    new_title_norm = _normalize_title(new_paper.get("title", ""))

    for src in existing_sources:
        # Skip non-paper sources
        if not hasattr(src, 'external_id') or not src.external_id:
            continue

        # Compare DOI if available (not implemented on SourceInfo, but can check title)
        src_title_norm = _normalize_title(src.title)
        if src_title_norm and new_title_norm and _title_similarity(new_title_norm, src_title_norm) > 0.85:
            return src

    return None


def deduplicate_paper_list(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate papers from a list, keeping the one with more metadata."""
    seen_titles: Dict[str, int] = {}  # normalized title -> index in result
    result: List[Dict[str, Any]] = []

    for p in papers:
        norm = _normalize_title(p.get("title", ""))
        if not norm:
            result.append(p)
            continue

        if norm in seen_titles:
            # Keep the one with more metadata (DOI, venue, higher citations)
            existing_idx = seen_titles[norm]
            existing = result[existing_idx]
            new_score = (1 if _paper_doi(p) else 0) + (1 if _paper_venue(p) else 0) + (p.get("citationCount") or 0)
            old_score = (1 if _paper_doi(existing) else 0) + (1 if _paper_venue(existing) else 0) + (existing.get("citationCount") or 0)
            if new_score > old_score:
                result[existing_idx] = p
        else:
            seen_titles[norm] = len(result)
            result.append(p)

    return result


# ----------------------------
# BibTeX
# ----------------------------

def bibtex_escape(s: str) -> str:
    s = (s or "").replace("\\", "\\\\")
    s = s.replace("{", "\\{").replace("}", "\\}")
    s = s.replace("\n", " ").strip()
    return s


def make_ref_key(p: Dict[str, Any]) -> str:
    year = str(p.get("year") or "").strip() or "n.d."
    authors = p.get("authors") or []
    first_author = ""
    if authors and isinstance(authors, list):
        first_author = (authors[0].get("name") or "").split()[-1]
    first_author = re.sub(r"[^A-Za-z0-9]+", "", first_author) or "anon"
    title = (p.get("title") or "").strip()
    first_word = re.sub(r"[^A-Za-z0-9]+", "", (title.split()[:1] or ["paper"])[0]).lower() or "paper"
    return f"{first_author}{year}{first_word}"


def rank_papers_by_relevance(
    papers: List[Dict],
    query_texts: List[str],
    top_n: int = 100,
) -> List[Dict]:
    """Rank papers by TF-IDF relevance × citation weight × recency factor.

    Args:
        papers: List of paper dicts (with title, abstract, citationCount, year)
        query_texts: List of query strings (user question + search queries)
        top_n: Number of top papers to return
    """
    import math
    from collections import Counter

    if not papers or not query_texts:
        return papers[:top_n]

    def _tokenize(text: str) -> List[str]:
        return re.findall(r'[a-zA-Z]{2,}', (text or "").lower())

    # Build document corpus: each paper's title+abstract
    docs = []
    for p in papers:
        doc_text = f"{p.get('title', '')} {p.get('abstract', '')}"
        docs.append(_tokenize(doc_text))

    # Query tokens
    query_tokens = []
    for q in query_texts:
        query_tokens.extend(_tokenize(q))
    query_tf = Counter(query_tokens)
    query_total = sum(query_tf.values()) or 1

    # IDF: computed from all docs + query
    all_docs = docs + [query_tokens]
    n_docs = len(all_docs)
    doc_freq: Dict[str, int] = {}
    for doc in all_docs:
        for term in set(doc):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    current_year = dt.date.today().year

    scored = []
    for i, p in enumerate(papers):
        doc_tokens = docs[i]
        if not doc_tokens:
            scored.append((0.0, i))
            continue

        doc_tf = Counter(doc_tokens)
        doc_total = sum(doc_tf.values()) or 1

        # TF-IDF relevance
        tfidf_score = 0.0
        for term in set(query_tokens) & set(doc_tokens):
            q_tf = query_tf[term] / query_total
            d_tf = doc_tf[term] / doc_total
            idf = math.log(n_docs / (doc_freq.get(term, 1)))
            tfidf_score += q_tf * d_tf * idf * idf

        # Citation weight: log(1 + citations)
        cites = p.get("citationCount") or 0
        cite_weight = math.log(1 + cites)

        # Recency factor
        year = p.get("year")
        if year and isinstance(year, (int, float)):
            age = current_year - int(year)
            if age <= 5:
                recency = 1.0 + 0.1 * (5 - age)
            else:
                recency = 1.0
        else:
            recency = 1.0

        score = tfidf_score * cite_weight * recency
        scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [papers[i] for _, i in scored[:top_n]]


def compact_search_batch(content: str, query_texts: List[str], top_n: int = 10) -> Tuple[str, int]:
    """Compact a search result JSON by re-ranking with TF-IDF and keeping top_n.

    Args:
        content: JSON string from a search_papers or get_paper_references tool result
        query_texts: List of query strings for relevance ranking
        top_n: Number of top papers to keep

    Returns:
        (new JSON string, number of papers removed)
    """
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content, 0

    if not isinstance(data, dict):
        return content, 0

    results = data.get("results")
    if not isinstance(results, list) or len(results) <= top_n:
        return content, 0

    original_count = len(results)
    ranked = rank_papers_by_relevance(results, query_texts, top_n=top_n)
    data["results"] = ranked
    data["returned"] = len(ranked)
    if "compacted_from" not in data:
        data["compacted_from"] = original_count

    new_content = json.dumps(data, ensure_ascii=False)
    return new_content, original_count - len(ranked)


def deduplicate_ref_keys(sources: List[Any]) -> None:
    """Resolve ref_key collisions with a/b/c suffixes. Modifies sources in place."""
    key_groups: Dict[str, List[Any]] = {}
    for s in sources:
        k = getattr(s, "ref_key", "") or ""
        if k:
            key_groups.setdefault(k, []).append(s)
    for key, group in key_groups.items():
        if len(group) > 1:
            for i, s in enumerate(group):
                s.ref_key = f"{key}{chr(ord('a') + i)}"


def paper_to_bibtex(p: Dict[str, Any]) -> str:
    title = bibtex_escape(p.get("title") or "")
    year = str(p.get("year") or "").strip()
    venue = bibtex_escape(_paper_venue(p))
    doi = bibtex_escape(_paper_doi(p))
    url = bibtex_escape(p.get("url") or "")
    authors = p.get("authors") or []
    author_str = " and ".join([bibtex_escape(a.get("name") or "") for a in authors if a.get("name")])

    entry_type = "article" if venue else "misc"
    key = make_ref_key(p)

    fields: List[Tuple[str, str]] = []
    if author_str:
        fields.append(("author", author_str))
    if title:
        fields.append(("title", title))
    if year:
        fields.append(("year", year))
    if venue:
        fields.append(("journal", venue))
    if doi:
        fields.append(("doi", doi))
    if url:
        fields.append(("url", url))

    pid = bibtex_escape(str(p.get("paperId") or ""))
    if pid:
        fields.append(("note", f"Semantic Scholar paperId: {pid}"))

    body = ",\n".join([f"  {k} = {{{v}}}" for k, v in fields])
    return f"@{entry_type}{{{key},\n{body}\n}}"


# ----------------------------
# Brave Search (standalone)
# ----------------------------

def brave_search(query: str, count: int = DEFAULT_BRAVE_COUNT, offset: int = 0,
                 domain_filter: str = "web") -> Dict[str, Any]:
    """Standalone Brave Search function (used by both /web command and agentic tools).

    domain_filter:
      "web"      — no filtering (default)
      "academic" — academic domains boosted via inline Goggle re-ranking (soft preference;
                   falls back to "web" silently if Goggles are unsupported on the API tier)
    """
    api_key = os.environ.get("BRAVE_API_KEY") or ""
    if not api_key:
        raise RuntimeError("Missing Brave key. Set BRAVE_API_KEY.")

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }

    params = {
        "q": query,
        "count": max(1, int(count)),  # Brave API hard cap is 20/call; pass value as-is
        "offset": max(0, min(9, int(offset))),
        "extra_snippets": True,
        "safesearch": "moderate",
    }

    # Academic mode: add inline Goggle to re-rank academic domains higher
    if domain_filter == "academic":
        params["goggles"] = ACADEMIC_GOGGLE

    def _call():
        resp = requests.get(BRAVE_ENDPOINT, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()

    if domain_filter == "academic":
        try:
            result = retry_api_call(_call)
        except Exception:
            # Goggles may not be supported on all Brave API tiers — fall back silently
            params.pop("goggles", None)
            result = retry_api_call(_call)
    else:
        result = retry_api_call(_call)

    increment_stat("brave")
    return result


# Backward-compatible aliases
make_bibtex_key = make_ref_key
deduplicate_bibtex_keys = deduplicate_ref_keys
