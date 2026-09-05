"""Turning extracted blocks into what is retrieved and what is answered from.

Two sizes, because retrieval and generation want opposite things. A small chunk
retrieves precisely: fewer unrelated sentences to dilute the embedding and the
term statistics. A large chunk answers well: the sentence that carries the fact
usually needs the two around it to be worth anything. The previous single size,
900 characters, was a compromise that served neither, and it cut wherever the
character count ran out - between a heading and its paragraph, or between a
number and its unit.

So a document becomes a stream of blocks that remembers which page and which
section each piece came from; children of about 250 tokens are what the index
holds, and every child points at a parent window of about 900 tokens that is
what the answering model is given. Chunk boundaries follow sentences, and never
cross a heading.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .pdf import Block

# A sentence ends at Latin or Persian punctuation, or at a blank line.
_SENTENCE_END = re.compile(r"(?<=[.!?؟؛…])\s+|\n{2,}|\n(?=[-*•])")
# Characters per token for a multilingual subword vocabulary, counting the
# spaces the way those vocabularies do. Latin text runs about four; Persian
# tokenizes shorter, because a multilingual vocabulary holds fewer whole Persian
# words and falls back to pieces more often. Both are estimates and only have to
# be stable: the chunk sizes below were chosen by measuring the golden set with
# this counter, not by matching one model's vocabulary.
_LATIN_CHARS_PER_TOKEN = 4.0
_PERSIAN_CHARS_PER_TOKEN = 3.0
_PERSIAN = re.compile(r"[؀-ۿ]")


def count_tokens(text: str) -> int:
    """Estimate how many tokens an embedding model would see."""
    if not text.strip():
        return 0
    persian = len(_PERSIAN.findall(text))
    return max(1, math.ceil(persian / _PERSIAN_CHARS_PER_TOKEN
                            + (len(text) - persian) / _LATIN_CHARS_PER_TOKEN))


def sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_END.split(text) if part and part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


@dataclass
class Piece:
    """A child chunk: what the index holds and what a citation points at."""

    content: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    section_path: str = ""
    parent: int = 0


@dataclass
class Window:
    """A parent window: what the answering model is given."""

    content: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: str = ""
    children: list[int] = field(default_factory=list)


SECTION_SEPARATOR = " › "


def section_paths(blocks: list[Block]) -> list[tuple[Block, tuple[str, ...]]]:
    """Attach the heading stack above each body block.

    A level-2 heading replaces the previous level-2 and everything under it,
    which is what makes «Section 3 › Cold chain incident» rather than a flat
    list of every heading seen so far.
    """
    stack: list[tuple[int, str]] = []
    attached: list[tuple[Block, tuple[str, ...]]] = []
    for block in blocks:
        if block.heading_level:
            while stack and stack[-1][0] >= block.heading_level:
                stack.pop()
            stack.append((block.heading_level, block.text.strip()))
            attached.append((block, tuple(title for _, title in stack)))
        else:
            attached.append((block, tuple(title for _, title in stack)))
    return attached


@dataclass
class _Unit:
    """One sentence, with everything needed to place it in a chunk."""

    text: str
    tokens: int
    page: int | None
    path: tuple[str, ...]


def _units(blocks: list[Block]) -> list[_Unit]:
    units: list[_Unit] = []
    for block, path in section_paths(blocks):
        if block.kind == "heading":
            # The heading itself leads its section rather than forming a chunk.
            continue
        pieces = [block.text] if block.kind == "table" else sentences(block.text)
        for piece in pieces:
            clean = re.sub(r"[ \t]+", " ", piece).strip()
            if clean:
                units.append(_Unit(clean, count_tokens(clean), block.page, path))
    return units


def _join(units: list[_Unit]) -> str:
    return " ".join(unit.text for unit in units).strip()


def _span(units: list[_Unit]) -> tuple[int | None, int | None]:
    pages = [unit.page for unit in units if unit.page is not None]
    return (min(pages), max(pages)) if pages else (None, None)


def common_path(paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    """The headings every one of these sentences sits under."""
    if not paths:
        return ()
    shared = list(paths[0])
    for path in paths[1:]:
        while shared and tuple(shared) != path[:len(shared)]:
            shared.pop()
        if not shared:
            break
    return tuple(shared)


def _child_ranges(units: list[_Unit], child_tokens: int, overlap_tokens: int) -> list[tuple[int, int]]:
    """Sentence ranges for the child chunks, as half-open [start, end) indices.

    A child ends when the next sentence would take it over the budget, or when
    the section changes and the chunk is already substantial. The second half of
    that rule matters: an HR policy of six two-sentence sections would otherwise
    become six 60-token chunks, each too short for either the term statistics or
    the embedding to say anything about, and each missing the context of the one
    beside it. Small neighbouring sections are packed together instead, and the
    chunk is then labelled with the heading they share.

    Consecutive children inside a section overlap by up to `overlap_tokens`, so
    a fact that sits exactly on a boundary is still whole in one of them.
    """
    ranges: list[tuple[int, int]] = []
    start = 0
    tokens = 0
    # The sentence that answers the question is often the one the boundary cuts,
    # and it is regularly longer than the overlap budget - the page break in the
    # evaluation corpus falls inside a 60-token sentence. Carrying nothing there
    # is the failure the overlap exists to prevent, so one sentence is always
    # carried when it fits in half a chunk, budget or no budget.
    minimum_carry = child_tokens // 2
    substantial = child_tokens // 2
    for position, unit in enumerate(units):
        if position > start:
            section_changed = unit.path != units[position - 1].path and tokens >= substantial
            if section_changed or tokens + unit.tokens > child_tokens:
                ranges.append((start, position))
                back = position
                carried = 0
                if not section_changed:
                    while back > start and carried + units[back - 1].tokens <= overlap_tokens:
                        back -= 1
                        carried += units[back].tokens
                    if back == position and units[position - 1].tokens <= minimum_carry:
                        back = position - 1
                start = position if back <= start or back >= position else back
                tokens = sum(item.tokens for item in units[start:position])
        tokens += unit.tokens
    if start < len(units):
        ranges.append((start, len(units)))
    return ranges


def _window_ranges(units: list[_Unit], children: list[tuple[int, int]],
                   parent_tokens: int) -> list[tuple[int, int, list[int]]]:
    """Group children into parent windows: (start, end, child indices).

    A window is built from the same sentence stream rather than by concatenating
    its children, so the overlap between children is not repeated in it.
    """
    windows: list[tuple[int, int, list[int]]] = []
    for index, (start, end) in enumerate(children):
        if windows:
            open_start, open_end, members = windows[-1]
            # A window may span sibling sections but never a top-level one: the
            # model should not be handed two unrelated chapters as one passage.
            same_chapter = units[start].path[:1] == units[open_start].path[:1]
            size = sum(item.tokens for item in units[open_start:max(open_end, end)])
            if same_chapter and size <= parent_tokens:
                windows[-1] = (open_start, max(open_end, end), members + [index])
                continue
        windows.append((start, end, [index]))
    return windows


def build_pieces(blocks: list[Block], child_tokens: int, overlap_tokens: int,
                 parent_tokens: int) -> tuple[list[Piece], list[Window]]:
    """Split a document into child chunks and the parent windows above them.

    Both are cut from the same sentence stream, so a child is always a
    contiguous run of its parent. Pages are recorded but never used as a
    boundary: the sentence that answers the question is regularly the one that
    straddles the page break, and page-scoped chunking is what made it
    unreachable.
    """
    units = _units(blocks)
    if not units:
        return [], []

    children = _child_ranges(units, child_tokens, overlap_tokens)
    windows = _window_ranges(units, children, parent_tokens)
    parent_of = {child: index for index, (_, _, members) in enumerate(windows) for child in members}

    pieces: list[Piece] = []
    for index, (start, end) in enumerate(children):
        span = units[start:end]
        page_start, page_end = _span(span)
        # A chunk that packed two small sections is labelled with the heading
        # they share, never with whichever of them happened to come first.
        path = common_path([unit.path for unit in span]) or span[0].path
        pieces.append(Piece(content=_join(span), page_start=page_start, page_end=page_end,
                            section=path[-1] if path else None,
                            section_path=SECTION_SEPARATOR.join(path),
                            parent=parent_of[index]))

    built_windows: list[Window] = []
    for start, end, members in windows:
        span = units[start:end]
        page_start, page_end = _span(span)
        path = common_path([unit.path for unit in span]) or span[0].path
        built_windows.append(Window(content=_join(span), page_start=page_start, page_end=page_end,
                                    section_path=SECTION_SEPARATOR.join(path),
                                    children=list(members)))
    return pieces, built_windows
