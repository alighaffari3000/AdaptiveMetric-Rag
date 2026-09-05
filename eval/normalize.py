"""Text normalization used only for judging golden-set matches.

Phase 4 introduces `app/text.py` as the single normalizer for the application
itself. This module stays independent on purpose: the yardstick must not move
when the code under test changes.
"""

from __future__ import annotations

import re


_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")})
_LETTER_MAP = {
    ord("ي"): "ی", ord("ﻱ"): "ی", ord("ی"): "ی",
    ord("ك"): "ک", ord("ﻙ"): "ک",
    ord("ة"): "ه", ord("أ"): "ا", ord("إ"): "ا", ord("آ"): "ا",
    ord("ؤ"): "و",
}
_STRIP = {
    ord("‌"): " ",   # zero width non-joiner
    ord("‏"): None,  # right-to-left mark
    ord("‎"): None,  # left-to-right mark
    ord("ـ"): None,  # tatweel
    ord("٬"): None,  # arabic thousands separator
    ord("٫"): ".",   # arabic decimal separator
}
_DIACRITICS = re.compile(r"[ً-ْٰ]")


def normalize(text: str) -> str:
    """Fold Persian/Arabic spelling variants so golden matches are stable."""
    if not text:
        return ""
    folded = text.translate(_DIGIT_MAP).translate(_LETTER_MAP).translate(_STRIP)
    folded = _DIACRITICS.sub("", folded)
    folded = folded.replace("⁦", "").replace("⁧", "").replace("⁩", "")
    return re.sub(r"\s+", " ", folded).strip().lower()


def contains(haystack: str, needle: str) -> bool:
    return normalize(needle) in normalize(haystack)
