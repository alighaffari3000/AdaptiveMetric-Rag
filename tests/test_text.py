"""The shared normalizer: spelling, digits, spacing, numbers and dates."""

from app.text import date_terms, normalize, number_terms, tokenize


def test_arabic_letters_fold_to_persian():
    assert normalize("كتاب عربي") == normalize("کتاب عربی")


def test_half_space_and_plain_space_agree():
    assert normalize("وب‌سایت") == normalize("وب سایت") == "وب سایت"
    assert tokenize("می‌شود") == tokenize("می شود")


def test_kashida_and_diacritics_disappear():
    assert normalize("لطفـــاً") == "لطفا"
    assert normalize("مُحَمَّد") == "محمد"


def test_persian_arabic_and_latin_digits_are_one_number():
    assert normalize("۱۴۰۳") == normalize("١٤٠٣") == "1403"
    assert tokenize("قرارداد ۱۳۷") == tokenize("قرارداد 137")


def test_grouping_separators_do_not_change_a_number():
    assert number_terms("۲٬۴۰۰٬۰۰۰٬۰۰۰ ریال") == ["2400000000"]
    assert number_terms("2,400,000,000") == ["2400000000"]
    assert number_terms("2400000000") == ["2400000000"]


def test_a_number_is_not_a_prefix_of_another():
    """137 used to match 1370 because the comparison was substring search."""
    assert number_terms("قرارداد ۱۳۷") == ["137"]
    assert "137" not in number_terms("سال ۱۳۷۰")


def test_decimal_separator_folds():
    assert number_terms("۹۹٫۵ درصد") == ["99.5"]


def test_jalali_year_is_found_despite_persian_digits():
    """The old pattern could not match ۱۴۰۲: its 1 and 4 were Latin literals."""
    assert "1402" in date_terms("در سال ۱۴۰۲")


def test_two_spellings_of_one_jalali_date_meet():
    numeric = set(date_terms("۱۴۰۲/۰۳/۱۵"))
    written = set(date_terms("۱۵ خرداد ۱۴۰۲"))
    assert "1402-03-15" in numeric & written
    assert "1402-03" in set(date_terms("خرداد ۱۴۰۲")) & numeric


def test_gregorian_dates_are_folded_the_same_way():
    assert "2025-03-12" in date_terms("effective 12 March 2025")
    assert "2025-03-12" in date_terms("March 12, 2025")
    assert "2025-03-12" in date_terms("2025/03/12")


def test_jalali_and_gregorian_dates_do_not_collide():
    assert not set(date_terms("۱۵ خرداد ۱۴۰۲")) & set(date_terms("5 June 2023"))


def test_stop_words_are_dropped_after_folding():
    assert "را" not in tokenize("کتاب را بخوان")
    assert "the" not in tokenize("read the book")
