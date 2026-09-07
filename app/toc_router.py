"""Ask the model which sections of a document a question belongs to.

This is the zero-shot version of the idea behind STAIR (arXiv:2609.03874).
The paper fine-tunes a generative retriever whose input carries the whole
table of contents, and reports that having the structure in the input rather
than making the model infer it lifts Recall@1 from 76.9 to 82.6 percent while
holding hallucinated identifiers under 0.05 percent. Fine-tuning per document
is not available here - this index is rebuilt on every ingest - but the table
of contents is, and a general model can be asked to read it.

Two properties of the paper carry over and both are enforced here rather than
hoped for:

- **Constrained output.** The paper's low hallucination rate comes from the
  model only ever emitting valid identifiers. A general model has no such
  guarantee, so every heading it returns is matched back against the real
  table of contents and anything else is dropped. Unconstrained, the paper
  measured 3.25 percent invalid identifiers from a fine-tuned system and
  23.86 percent from an off-the-shelf one.
- **The router narrows, it never decides.** If the model returns nothing
  usable, times out, or fails, retrieval runs across the whole library exactly
  as it would have. A router that can lose an answer is worse than no router.

Off by default: it spends one extra model call per query, which is only worth
it for a document long enough that first-stage retrieval has real competition.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

from .models import AppSettings

logger = logging.getLogger("adaptive_metric_rag.toc_router")

# Below this similarity a returned heading is treated as invented rather than
# as a typo of a real one. High enough that two sibling headings differing by a
# word ("مرخصی استحقاقی" / "مرخصی استعلاجی") never collapse into each other.
COVERAGE_SEPARATOR = " | "
MATCH_THRESHOLD = 0.86
MAX_TOC_ENTRIES = 200
TIMEOUT_SECONDS = 20.0

_DIGITS = {ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}
_DIGITS.update({ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")})
_LETTERS = {ord("ي"): "ی", ord("ك"): "ک", ord("ة"): "ه", ord("أ"): "ا", ord("إ"): "ا", ord("آ"): "ا"}


class RouterUnavailable(RuntimeError):
    """The router could not produce a usable answer; search everything."""


def normalize_heading(text: str) -> str:
    """Fold the differences that are not differences: digits, yeh, kaf, dashes."""
    folded = unicodedata.normalize("NFKC", text or "")
    folded = folded.translate(_DIGITS).translate(_LETTERS)
    folded = re.sub(r"[‌‎‏ـ]", " ", folded)
    # The path separator is punctuation between words, not a word: a model that
    # replies with "فصل دوم - مرخصی استعلاجی" means the same section as
    # "فصل دوم — مرخصی > مرخصی استعلاجی" and must not be treated as inventing one.
    folded = folded.replace(">", " ")
    folded = re.sub(r"[—–\-_:.،,]+", " ", folded)
    return re.sub(r"\s+", " ", folded).strip().lower()


def build_prompt(question: str, toc: list[dict[str, Any]]) -> tuple[str, str]:
    listing = "\n".join(f"- {entry.get('label', entry['path'])}" for entry in toc[:MAX_TOC_ENTRIES])
    system = (
        "You locate where in a document an answer lives. You are given the document's table "
        "of contents and a question. Choose the sections most likely to contain the answer. "
        "Copy section names exactly as they appear in the list. Never invent a section. "
        "The question and the document may be in different languages; that is never a reason "
        "to choose nothing."
    )
    user = (
        f"Table of contents:\n{listing}\n\nQuestion:\n{question}\n\n"
        'Reply with JSON only, in the form {"sections": ["...", "..."]}, naming at most '
        "three sections copied exactly from the list above. If the answer could be anywhere, "
        'reply {"sections": []}.'
    )
    return system, user


def parse_sections(raw: str) -> list[str]:
    """Read the model's JSON, tolerating code fences and surrounding prose."""
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", (raw or "").strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise RouterUnavailable("the router model did not return JSON")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise RouterUnavailable(f"the router model returned invalid JSON: {exc}") from exc
    entries = payload.get("sections")
    if entries is None or not isinstance(entries, list):
        raise RouterUnavailable("the router model returned no section list")
    return [entry.strip() for entry in entries if isinstance(entry, str) and entry.strip()]


def constrain(proposed: list[str], toc: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep only headings that exist, which is what makes this safe to act on.

    Exact match first, then normalized, then a similarity threshold for a
    heading the model retyped slightly. Anything that survives none of those is
    invented, and an invented heading would silently exclude the part of the
    document that actually holds the answer.

    Matching runs against the label the model was shown, and a whole table of
    contents entry comes back rather than a bare path. Two documents may name a
    section identically, so returning only the path would select both and make
    the document prefix in the label decorative.
    """
    if not proposed or not toc:
        return []
    exact: dict[str, dict[str, Any]] = {}
    folded: dict[str, dict[str, Any]] = {}
    for entry in toc:
        label = entry.get("label", entry["path"])
        exact.setdefault(label, entry)
        exact.setdefault(entry["path"], entry)
        for form in (label, entry["path"], entry["title"]):
            folded.setdefault(normalize_heading(form), entry)

    kept: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in proposed:
        resolved = exact.get(candidate) or folded.get(normalize_heading(candidate))
        if resolved is None:
            target = normalize_heading(candidate)
            best, best_score = None, 0.0
            for key, entry in folded.items():
                score = SequenceMatcher(None, target, key).ratio()
                if score > best_score:
                    best, best_score = entry, score
            resolved = best if best_score >= MATCH_THRESHOLD else None
        if resolved is None:
            logger.info("router proposed a section that is not in the table of contents: %r", candidate)
            continue
        identity = (resolved.get("document_name", ""), resolved["path"])
        if identity not in seen:
            seen.add(identity)
            kept.append(resolved)
        if len(kept) >= limit:
            break
    return kept


def toc_from_rows(rows: list[dict[str, Any]], positions: list[int] | None = None) -> list[dict[str, Any]]:
    """The table of contents of whatever is in scope for this search.

    The paper routes inside one book. A library holds several, so a path is
    shown to the model prefixed with its document name when more than one
    document is in scope, which keeps two documents' identically named sections
    apart. Each entry also keeps the unprefixed `path`, because that is what a
    chunk actually carries and what selection has to match on.
    """
    from .structure import PATH_SEPARATOR, build_toc

    indices = positions if positions is not None else range(len(rows))
    selected = [rows[index] for index in indices if 0 <= index < len(rows)]
    qualified = len({row.get("document_name", "") for row in selected}) > 1
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in selected:
        name = row.get("document_name", "")
        for entry in build_toc([row_sections(row)]):
            label = f"{name}{PATH_SEPARATOR}{entry['path']}" if qualified else entry["path"]
            if label in seen:
                continue
            seen.add(label)
            entries.append({**entry, "label": label, "document_name": name})
    return entries


def eligible(settings: AppSettings, toc: list[dict[str, Any]], chunk_count: int) -> bool:
    """Whether routing is worth a model call for this document."""
    return bool(
        settings.toc_router_enabled
        and settings.provider != "local"
        and len(toc) >= 2
        and chunk_count >= settings.toc_router_min_chunks
    )


async def route(settings: AppSettings, question: str,
                toc: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sections of `toc` the question probably belongs to; [] means search all."""
    from .providers import complete

    if not toc:
        return []
    system, user = build_prompt(question, toc)
    try:
        raw = await asyncio.wait_for(
            complete(settings, system, user, model=settings.model, temperature=0.0, max_tokens=300),
            timeout=TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise RouterUnavailable("the router model timed out") from exc
    except RouterUnavailable:
        raise
    except Exception as exc:
        raise RouterUnavailable(f"the router model failed: {type(exc).__name__}: {exc}") from exc
    return constrain(parse_sections(raw), toc, settings.toc_router_max_sections)


def matching_positions(rows: list[dict[str, Any]], sections: list[Any]) -> list[int]:
    """Row indices covering any of the chosen sections, in the named document.

    A packed chunk declares several sections at once, so membership is tested
    against each section the chunk covers rather than against the joined label.
    A selection that names a document only matches that document's chunks, or
    two files with an identically named section would select each other's.
    """
    if not sections:
        return []
    wanted: list[tuple[str, str]] = []
    for section in sections:
        if isinstance(section, dict):
            wanted.append((section.get("document_name", ""), normalize_heading(section["path"])))
        else:
            wanted.append(("", normalize_heading(str(section))))
    positions: list[int] = []
    for position, row in enumerate(rows):
        name = row.get("document_name", "")
        for path in row_sections(row):
            folded = normalize_heading(path)
            # A chosen parent section selects everything filed beneath it.
            if any((not document or document == name)
                   and (folded == want or folded.startswith(f"{want} ") or want in folded)
                   for document, want in wanted):
                positions.append(position)
                break
    return positions


def row_sections(row: dict[str, Any]) -> list[str]:
    """The sections a chunk covers, as a list rather than a display label."""
    sections = row.get("sections")
    if isinstance(sections, list) and sections:
        return [section for section in sections if section]
    path = row.get("section_path") or row.get("section") or ""
    if not path:
        return []
    if COVERAGE_SEPARATOR not in path:
        return [path]
    # A row written before the list existed carries the merged label. Splitting
    # it is best-effort - a tail containing the path separator is ambiguous -
    # but reading the whole label as one section is certainly wrong, and would
    # hide every section after the first from the router.
    from .structure import PATH_SEPARATOR

    head, _, tail = path.rpartition(PATH_SEPARATOR)
    if COVERAGE_SEPARATOR not in tail:
        return [path]
    prefix = f"{head}{PATH_SEPARATOR}" if head else ""
    return [f"{prefix}{part.strip()}" for part in tail.split(COVERAGE_SEPARATOR) if part.strip()]
