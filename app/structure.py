"""Document structure: headings, section paths, and a table of contents.

Length-based chunking throws away the one signal a document gives away for
free. A heading tells you what the text under it is about, often in words the
body itself never repeats: a clause headed "مرخصی استعلاجی" may say only
"حداکثر هشت روز در سال" and never name the kind of leave again. Cutting every
`chunk_size` characters files that sentence under nothing, and retrieval is
left to match a question against a body that does not contain the question's
subject.

So headings are extracted here and carried on the chunk, and section
boundaries become chunk boundaries. The idea is taken from STAIR
(arXiv:2609.03874), which showed that giving a retriever the corpus structure
explicitly beats making it infer the structure from the text; the fine-tuned
generative retriever the paper builds on top of that is not adopted, since
this index is rebuilt on every ingest.

Detection is deliberately conservative. A wrong section path is worse than no
section path: an empty one costs a signal, a wrong one adds a misleading term
to the very field that is supposed to disambiguate.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any

PATH_SEPARATOR = " > "

# A heading has to look like a heading: short, not a sentence, and either
# marked up (Markdown, HTML) or matching a numbering convention.
MAX_HEADING_LENGTH = 120

_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_NUM = r"[0-9۰-۹]"
# Persian ordinal words used by legal and policy documents, which number their
# divisions in words rather than digits: "فصل دوم", "بخش چهارم".
_ORDINAL = ("اول|یکم|نخست|دوم|سوم|چهارم|پنجم|ششم|هفتم|هشتم|نهم|دهم|"
            "یازدهم|دوازدهم|سیزدهم|چهاردهم|پانزدهم|شانزدهم|هفدهم|هجدهم|نوزدهم|بیستم")

# Ordered by nesting depth: a فصل contains ماده contains تبصره, so a تبصره
# heading must not close the ماده above it. Each pattern matches the numbering
# only; what follows it is judged by `_titleish_tail`, because "Section 4." and
# "Section 3 covers rollback and recovery." differ in their tail, not in their
# numbering and not in how they end.
_PERSIAN_PATTERNS: list[tuple[int, re.Pattern[str]]] = [
    (1, re.compile(rf"^(?:فصل|باب)\s+(?:{_ORDINAL}|{_NUM}+)")),
    (2, re.compile(rf"^(?:بخش|مبحث|گفتار)\s+(?:{_ORDINAL}|{_NUM}+)")),
    (2, re.compile(rf"^ماده\s+{_NUM}+")),
    (3, re.compile(rf"^(?:تبصره|بند)(?:\s+(?:{_NUM}+|{_ORDINAL}))?")),
]

_ENGLISH_PATTERNS: list[tuple[int, re.Pattern[str]]] = [
    (1, re.compile(r"^(?:chapter|part)\s+(?:[0-9]+|[ivxlcdm]+)\b", re.IGNORECASE)),
    (2, re.compile(r"^(?:section|article|appendix)\s+(?:[0-9]+|[ivxlcdm]+)\b", re.IGNORECASE)),
]

# "2.", "3.1", "4.2.1 Rollback" - depth comes from how many components there are.
_DOTTED = re.compile(rf"^({_NUM}+(?:\.{_NUM}+)*)\.?\s+\S.*$")

# A title follows its numbering after a separator; a sentence just continues.
_TITLE_SEPARATOR = re.compile(r"^[-—–:.،]")
MAX_TITLE_WORDS_WITHOUT_SEPARATOR = 4


def _titleish_tail(tail: str) -> bool:
    """Whether what follows a numbering reads as a title rather than a clause.

    Nothing at all ("تبصره ۱"), or a separator ("ماده ۱۲:", "Section 1. Network
    summary", "فصل اول — کلیات"), or a short label with no separator
    ("ماده ۱ طرفین قرارداد"). A sentence that merely opens with the keyword
    ("ماده ۵ حقوق را تعیین می‌کند.") does none of these.
    """
    tail = tail.strip()
    if not tail:
        return True
    if _TITLE_SEPARATOR.match(tail):
        return True
    return len(tail.split()) <= MAX_TITLE_WORDS_WITHOUT_SEPARATOR and not tail.endswith(".")


_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(\S.*?)\s*#*\s*$")
_SETEXT = re.compile(r"^\s*(=+|-+)\s*$")
_YAML_KEY = re.compile(r"^[\"\']?[^\s:#][^:]*[\"\']?\s*:")


@dataclass(frozen=True)
class Heading:
    """One heading, and where in its block's text it starts."""

    level: int
    title: str
    offset: int


def _clean_title(text: str) -> str:
    """Strip markup leftovers and collapse whitespace, keeping the words."""
    title = re.sub(r"[ \t]+", " ", text).strip()
    title = title.strip("#").strip()
    return re.sub(r"\s*[-—–:]\s*$", "", title).strip()


# A heading is a label, not a sentence. Every heading in the evaluation corpus
# is five words or fewer; the prose that was being mistaken for one - "بند اول
# این است که همه کارکنان باید حضور داشته باشند", "Section 3 of the report was
# written by the finance team" - runs to ten or more.
MAX_HEADING_WORDS = 9
# Not "." or ":": "Section 4." and "ماده ۱۲:" are ordinary heading forms, and
# `_clean_title` already strips a trailing colon. Prose that opens with a
# heading keyword is caught by the word count instead.
_SENTENCE_END = ("،", ",", "؛", ";", "؟", "?", "!", "۔")


def _looks_like_prose(line: str) -> bool:
    """Reject a numbered line that is really a sentence or a list item."""
    stripped = line.strip()
    if not stripped or len(stripped) > MAX_HEADING_LENGTH:
        return True
    if len(stripped.split()) > MAX_HEADING_WORDS:
        return True
    # A heading does not close like a sentence.
    return stripped.endswith(_SENTENCE_END)


def _pattern_level(line: str) -> int | None:
    stripped = line.strip()
    if not stripped or _looks_like_prose(line):
        return None
    for level, pattern in _PERSIAN_PATTERNS + _ENGLISH_PATTERNS:
        match = pattern.match(stripped)
        if match and _titleish_tail(stripped[match.end():]):
            return level
    dotted = _DOTTED.match(stripped)
    if dotted:
        components = dotted.group(1).count(".") + 1
        # A bare "1 Introduction" is too weak a signal on its own; require at
        # least one dot, so prose starting with a year or a quantity is safe.
        if components >= 2:
            return min(components, 6)
    return None


def _front_matter_end(lines: list[str]) -> int:
    """Line index just past a YAML front matter block, or 0 when there is none.

    Its closing "---" is a valid setext underline, so without this the line
    above it becomes a heading and the top of the document gets filed under
    something like "author: Ops".
    """
    if not lines or lines[0].strip() != "---":
        return 0
    keyed = False
    for index in range(1, min(len(lines), 200)):
        stripped = lines[index].strip()
        if stripped in {"---", "..."}:
            # A leading "---" with no key above the closing one is a horizontal
            # rule, and skipping to it would swallow the document's headings.
            return index + 1 if keyed else 0
        if stripped and not stripped.startswith("#") and _YAML_KEY.match(stripped):
            keyed = True
    return 0


def detect_markdown(text: str) -> list[Heading]:
    headings: list[Heading] = []
    offset = 0
    lines = text.splitlines(keepends=True)
    skip_until = _front_matter_end(lines)
    fenced = False
    for position, line in enumerate(lines):
        if position < skip_until:
            offset += len(line)
            continue
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fenced = not fenced
        elif not fenced:
            match = _MARKDOWN_HEADING.match(line.rstrip("\n"))
            if match:
                headings.append(Heading(len(match.group(1)), _clean_title(match.group(2)), offset))
            elif _SETEXT.match(line) and position > skip_until:
                previous = lines[position - 1].strip()
                if previous and not _MARKDOWN_HEADING.match(previous):
                    level = 1 if line.strip().startswith("=") else 2
                    start = offset - len(lines[position - 1])
                    headings.append(Heading(level, _clean_title(previous), max(start, 0)))
        offset += len(line)
    return headings


def detect_plain(text: str) -> list[Heading]:
    """Headings in text that carries no markup: numbering conventions only."""
    headings: list[Heading] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        level = _pattern_level(line)
        if level is not None:
            headings.append(Heading(level, _clean_title(line), offset))
        offset += len(line)
    return headings


def detect_html(payload: str) -> list[tuple[str, int | None, str]]:
    """Split HTML into (text, page, section_path) blocks along h1..h6.

    BeautifulSoup's `get_text` flattens headings into the body, which is what
    the previous ingest did and why an HTML document arrived structureless.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(payload, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    root = soup.body or soup
    stack: list[tuple[int, str]] = []
    blocks: list[tuple[str, int | None, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(part for part in buffer if part.strip())
        if text.strip():
            blocks.append((text, None, join_path(stack)))
        buffer.clear()

    for element in root.find_all(True, recursive=True):
        if element.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            flush()
            level = int(element.name[1])
            title = _clean_title(element.get_text(" ", strip=True))
            if title:
                stack = [(lv, t) for lv, t in stack if lv < level] + [(level, title)]
                # The heading opens its own section's text. Markdown and plain
                # text keep it because their sections are sliced out of the raw
                # string; dropping it here would make a term that appears only
                # in a heading impossible to find lexically.
                buffer.append(title)
        elif element.name in {"p", "li", "td", "th", "pre", "blockquote", "dd", "dt", "figcaption"}:
            if not element.find(["p", "li", "td", "th", "pre", "blockquote"]):
                text = element.get_text(" ", strip=True)
                if text:
                    buffer.append(text)
    flush()
    if not blocks:
        text = root.get_text("\n", strip=True)
        return [(text, None, "")] if text.strip() else []
    return blocks


def pdf_outline(reader: Any) -> dict[int, list[str]]:
    """Bookmark titles per page, from the PDF's own table of contents.

    A real outline beats any heuristic, so it is tried first. Malformed
    outlines are common in the wild; every failure degrades to "no outline".
    """
    try:
        raw = reader.outline
    except Exception:
        return {}
    found: list[tuple[int, int, str]] = []

    def walk(items: Any, depth: int) -> None:
        if depth > 6:
            return
        for item in items or []:
            if isinstance(item, list):
                walk(item, depth + 1)
                continue
            try:
                title = _clean_title(str(item.title))
                page = reader.get_destination_page_number(item)
            except Exception:
                continue
            if title:
                found.append((int(page), depth, title))

    try:
        walk(raw, 1)
    except Exception:
        return {}
    if not found:
        return {}
    found.sort(key=lambda entry: entry[0])
    per_page: dict[int, list[str]] = {}
    stack: list[tuple[int, str]] = []
    for page, depth, title in found:
        stack = [(d, t) for d, t in stack if d < depth] + [(depth, title)]
        per_page.setdefault(page + 1, [part for _, part in stack])
    return per_page


def join_path(stack: list[tuple[int, str]]) -> str:
    return PATH_SEPARATOR.join(title for _, title in stack)


def section_paths(text: str, headings: list[Heading], base: str = "") -> list[tuple[int, int, str]]:
    """Slice a block into (start, end, section_path) spans.

    Text before the first heading keeps the base path, so a preamble is never
    filed under a heading that comes after it.
    """
    if not headings:
        return [(0, len(text), base)]
    spans: list[tuple[int, int, str]] = []
    stack: list[tuple[int, str]] = []
    ordered = sorted(headings, key=lambda h: h.offset)
    if ordered[0].offset > 0:
        spans.append((0, ordered[0].offset, base))
    for position, heading in enumerate(ordered):
        stack = [(lv, t) for lv, t in stack if lv < heading.level] + [(heading.level, heading.title)]
        end = ordered[position + 1].offset if position + 1 < len(ordered) else len(text)
        path = join_path(stack)
        if base:
            path = f"{base}{PATH_SEPARATOR}{path}" if path else base
        spans.append((heading.offset, end, path))
    return spans


COVERAGE_SEPARATOR = " | "


def common_prefix(paths: list[str]) -> list[str]:
    """The section components every one of these paths starts with."""
    if not paths:
        return []
    split = [path.split(PATH_SEPARATOR) if path else [] for path in paths]
    prefix: list[str] = []
    for parts in zip(*split):
        if len(set(parts)) != 1:
            break
        prefix.append(parts[0])
    return prefix


def merge_section_paths(paths: list[str]) -> str:
    """One readable line naming every section a chunk covers.

    Packing sibling sections into one chunk is a trade: it keeps chunks big
    enough to rank well, at the cost of a chunk that is about more than one
    thing. The chunk then has to say so, or the section path becomes a lie -
    it would name one heading while the text answers questions filed under
    another. Shared ancestry is written once and the differing tails listed:

        فصل اول > دورکاری | حضور در دفتر | ماموریت

    This is a label, for a citation and for the structure signal to match
    against. It is deliberately not a parseable encoding: a tail can itself
    contain the path separator, so a reader cannot tell which "A > B > c | d"
    was two sections or three. Code that needs the individual sections reads
    the `sections` list carried alongside the chunk instead.
    """
    distinct = list(dict.fromkeys(path for path in paths if path))
    if not distinct:
        return ""
    if len(distinct) == 1:
        return distinct[0]
    prefix = common_prefix(distinct)
    tails = []
    for path in distinct:
        tail = PATH_SEPARATOR.join(path.split(PATH_SEPARATOR)[len(prefix):])
        if tail and tail not in tails:
            tails.append(tail)
    if not tails:
        return PATH_SEPARATOR.join(prefix)
    joined = COVERAGE_SEPARATOR.join(tails)
    return f"{PATH_SEPARATOR.join(prefix)}{PATH_SEPARATOR}{joined}" if prefix else joined


def build_toc(section_lists: list[list[str]]) -> list[dict[str, Any]]:
    """A flat, ordered table of contents from the chunks' own section lists."""
    toc: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sections in section_lists:
        for path in sections:
            if not path or path in seen:
                continue
            seen.add(path)
            parts = path.split(PATH_SEPARATOR)
            toc.append({"path": path, "title": parts[-1], "depth": len(parts)})
    return toc


def looks_like_heading(line: str, unambiguous: bool = False) -> bool:
    """Whether a line is a heading rather than a sentence of the text.

    A heading is kept in its section's text so that a term appearing only in a
    heading is still findable. It is a label, though, so it must not be quoted
    back as the answer to the question it names.

    `unambiguous` drops the bare dotted-number rule, which cannot tell
    "4.2.1 Rollback procedure" from "7.4.0 fixes the memory leak". At ingest a
    false positive only adds a section boundary; when splitting a chunk to
    quote from it, a false positive makes that line unquotable, so the reader
    would never be shown the sentence that answers.
    """
    stripped = line.strip()
    if not stripped:
        return False
    if _MARKDOWN_HEADING.match(stripped):
        return True
    if _looks_like_prose(stripped):
        return False
    patterns = _PERSIAN_PATTERNS + _ENGLISH_PATTERNS
    if any(pattern.match(stripped) for _, pattern in patterns):
        return True
    return False if unambiguous else _pattern_level(stripped) is not None


def detect(text: str, suffix: str) -> list[Heading]:
    """Headings for a block of plain text, chosen by file type."""
    if suffix in {".md", ".markdown"}:
        headings = detect_markdown(text)
        # A Markdown file may still number its sections in prose style; only
        # fall back when the markup produced nothing, so "#" always wins.
        return headings or detect_plain(text)
    if suffix in {".txt", ""}:
        return detect_plain(text)
    return []
