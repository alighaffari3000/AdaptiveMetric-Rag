"""One place where Persian text is folded to a comparable form.

The same Persian word is written several ways. A keyboard produces the Arabic
`ي` and `ك` where Persian uses `ی` and `ک`; a half space may be a zero width
non-joiner, a plain space, or nothing at all; digits arrive as `۱۴۰۳`, `١٤٠٣`
or `1403`. None of that is visible to a reader, and all of it is fatal to a
lexical match: before this module, the question «قرارداد ۱۳۷» and the document
"قرارداد 137" shared no token, and the year pattern in `DATE_RE` could not match
`۱۴۰۲` at all because its `1` and `4` were literal Latin digits.

Every stage that compares text - ingestion tokens, BM25, the query analyser, the
entity, numeric and temporal signals - folds through `normalize` here, so the
folding is decided once instead of being approximated differently in each place.

The folding is for matching only. Stored chunk content and everything shown to a
reader stay exactly as written.
"""

from __future__ import annotations

import re

# Bumped whenever folding changes in a way that invalidates stored tokens or
# feature-hashed vectors; `app.main` re-derives them on the next start.
NORMALIZER_VERSION = "2"

_DIGITS = {ord(char): str(value) for value, char in enumerate("۰۱۲۳۴۵۶۷۸۹")}
_DIGITS.update({ord(char): str(value) for value, char in enumerate("٠١٢٣٤٥٦٧٨٩")})

_LETTERS = {
    ord("ي"): "ی", ord("ﻯ"): "ی", ord("ﻱ"): "ی", ord("ﻲ"): "ی", ord("ى"): "ی",
    ord("ك"): "ک", ord("ﻙ"): "ک", ord("ﻚ"): "ک", ord("ﻛ"): "ک",
    ord("أ"): "ا", ord("إ"): "ا", ord("آ"): "ا", ord("ٱ"): "ا", ord("ﺍ"): "ا",
    ord("ة"): "ه", ord("ۀ"): "ه",
    ord("ؤ"): "و",
}

_STRIP = {
    ord("‌"): " ",  # zero width non-joiner (نیم‌فاصله)
    ord("‍"): "",   # zero width joiner
    ord("‎"): " ",  # left-to-right mark
    ord("‏"): " ",  # right-to-left mark
    ord("⁦"): "",   # bidi isolates
    ord("⁧"): "",
    ord("⁨"): "",
    ord("⁩"): "",
    ord("ـ"): "",   # kashida
    ord("٬"): "",        # arabic thousands separator: ۲٬۴۰۰ and 2400 are one number
    ord("٫"): ".",       # arabic decimal separator
}

# Harakat and the superscript alef; invisible to a reader, fatal to a match.
_DIACRITICS = re.compile(r"[ً-ْٰٓ-ٕ]")
_WHITESPACE = re.compile(r"\s+")

TOKEN_RE = re.compile(r"[\w؀-ۿ.-]+", re.UNICODE)

PERSIAN_STOP = {"از", "به", "در", "با", "برای", "که", "این", "آن", "را", "و", "یا",
                "چه", "چرا", "چگونه", "است", "شد", "می"}
ENGLISH_STOP = {"the", "a", "an", "of", "to", "in", "for", "is", "was", "and", "or",
                "what", "why", "how", "does"}


def normalize(text: str) -> str:
    """Fold spelling, digit and spacing variants into one comparable form."""
    if not text:
        return ""
    folded = text.translate(_DIGITS).translate(_LETTERS).translate(_STRIP)
    folded = _DIACRITICS.sub("", folded)
    return _WHITESPACE.sub(" ", folded).strip().lower()


# The stop lists are written the way a reader writes them, so they are folded
# once here rather than at every lookup.
STOPWORDS = {normalize(word) for word in PERSIAN_STOP | ENGLISH_STOP}


def tokenize(text: str) -> list[str]:
    """The token stream behind BM25, feature hashing and every keyword signal."""
    return [token for token in TOKEN_RE.findall(normalize(text))
            if len(token) > 1 and token not in STOPWORDS]


def language_of(text: str) -> str:
    """The language the answer should be written in: any Persian script means fa."""
    return "fa" if re.search(r"[؀-ۿ]", text) else "en"


# High-frequency Persian words. None of them is a common word when spelled
# backwards, so counting both spellings tells a correctly extracted page from a
# mirrored one without needing a dictionary.
_MIRROR_MARKERS = frozenset({
    "است", "برای", "این", "که", "های", "شود", "با", "در", "از", "را", "به", "هر", "نیز",
    "شرکت", "قرارداد", "تاریخ", "سال", "می", "ماه", "روز", "درصد", "ریال", "مبلغ", "شماره",
    "کل", "بر", "تا", "یک", "دو", "خدمات", "فنی", "مورد", "طرف", "پس", "شده", "کرد",
})
# Digits and Latin letters keep their own direction inside a Persian line, so a
# run of them is reversed back after the line is un-mirrored.
_DIRECTIONAL_RUN = re.compile(r"[0-9A-Za-z۰-۹٠-٩](?:[0-9A-Za-z۰-۹٠-٩.,:/٫٬%+-]*[0-9A-Za-z۰-۹٠-٩])?")


def mirror_score(text: str) -> int:
    """How much more the text reads backwards than forwards, in marker words.

    Counted as whole words. Substring counting looked simpler and was wrong:
    «را» and «در» occur by accident inside longer words in both directions, so
    a short mirrored line scored zero and went unrepaired.
    """
    tokens = re.findall(r"[^\W\d_]+", normalize(text), re.UNICODE)
    forward = sum(1 for token in tokens if token in _MIRROR_MARKERS)
    backward = sum(1 for token in tokens if token[::-1] in _MIRROR_MARKERS)
    return backward - forward


def looks_mirrored(text: str) -> bool:
    """Whether a PDF extractor returned Persian text in visual order.

    Persian in a PDF is a sequence of positioned glyphs with no inherent
    direction. Extractors disagree about who already applied the bidi
    reordering, and when both do it - or neither - every word comes out
    spelled backwards: «قرارداد» as «دادرارق». Reading it is impossible and
    matching it is worse, because the tokens are silently wrong rather than
    missing.
    """
    return language_of(text) == "fa" and mirror_score(text) > 0


def unmirror(text: str) -> str:
    """Undo visual ordering, line by line, keeping numbers and Latin readable."""
    lines = []
    for line in text.splitlines():
        flipped = line[::-1]
        lines.append(_DIRECTIONAL_RUN.sub(lambda match: match.group(0)[::-1], flipped))
    return "\n".join(lines)


# Grouped forms first, so 2,400,000 is one number rather than three.
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def number_terms(text: str) -> list[str]:
    """Numbers in a form that survives the way they were typed.

    `۲٬۴۰۰٬۰۰۰٬۰۰۰`, `2,400,000,000` and `2400000000` are the same amount and
    must compare equal; `137` and `1370` must not, which substring matching on
    raw text could not tell apart.
    """
    found = _NUMBER_RE.findall(normalize(text))
    return list(dict.fromkeys(item.replace(",", "") for item in found))


JALALI_MONTHS = {
    "فروردین": 1, "اردیبهشت": 2, "خرداد": 3, "تیر": 4, "مرداد": 5, "شهریور": 6,
    "مهر": 7, "آبان": 8, "آذر": 9, "دی": 10, "بهمن": 11, "اسفند": 12,
}
GREGORIAN_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_NAMES = {normalize(name): number for name, number in JALALI_MONTHS.items()}
_MONTH_NAMES.update(GREGORIAN_MONTHS)
_MONTH_ALTERNATION = "|".join(sorted((re.escape(name) for name in _MONTH_NAMES), key=len, reverse=True))

_YEAR_FIRST = re.compile(r"\b(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})\b")
_YEAR_LAST = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b")
_SHORT_DATE = re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2}\b")
_DAY_MONTH_YEAR = re.compile(rf"\b(\d{{1,2}})\s+({_MONTH_ALTERNATION})\s+(\d{{4}})\b")
_MONTH_DAY_YEAR = re.compile(rf"\b({_MONTH_ALTERNATION})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b")
_MONTH_YEAR = re.compile(rf"\b({_MONTH_ALTERNATION})\s+(\d{{4}})\b")
_YEAR = re.compile(r"\b(1[34]\d{2}|19\d{2}|20\d{2})\b")


def _calendar_terms(year: int, month: int, day: int | None) -> list[str]:
    """A date as a year, a month and (when known) a day, coarsest last.

    Emitting every level is what lets «خرداد ۱۴۰۲» match a document that writes
    the same date as ۱۴۰۲/۰۳/۱۵: they meet at `1402-03`.
    """
    if not 1 <= month <= 12:
        return [str(year)]
    terms = [f"{year:04d}-{month:02d}", str(year)]
    if day is not None and 1 <= day <= 31:
        terms.insert(0, f"{year:04d}-{month:02d}-{day:02d}")
    return terms


def date_terms(text: str) -> list[str]:
    """Dates reduced to one calendar-agnostic shape.

    Jalali and Gregorian dates are folded the same way and are not converted
    into each other: `1402-03-15` is the Jalali date and `2025-03-12` the
    Gregorian one, and they simply never collide.

    A day-first reading is assumed for `11/03/2027`, matching how both Persian
    and European documents write it. The assumption is applied to questions and
    documents alike, so a wrong guess cannot make the two disagree.
    """
    folded = normalize(text)
    terms: list[str] = []
    for year, month, day in _YEAR_FIRST.findall(folded):
        terms += _calendar_terms(int(year), int(month), int(day))
    for day, month, year in _YEAR_LAST.findall(folded):
        terms += _calendar_terms(int(year), int(month), int(day))
    for day, month, year in _DAY_MONTH_YEAR.findall(folded):
        terms += _calendar_terms(int(year), _MONTH_NAMES[month], int(day))
    for month, day, year in _MONTH_DAY_YEAR.findall(folded):
        terms += _calendar_terms(int(year), _MONTH_NAMES[month], int(day))
    for month, year in _MONTH_YEAR.findall(folded):
        terms += _calendar_terms(int(year), _MONTH_NAMES[month], None)
    # A two-digit year cannot be placed in a century; keep the literal so the
    # form is still a temporal term rather than disappearing.
    terms += _SHORT_DATE.findall(folded)
    terms += _YEAR.findall(folded)
    return list(dict.fromkeys(terms))
