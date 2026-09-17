"""Metin normalizasyonu testleri (spec §18)."""

from app.normalization.text import dedup_key, normalize_whitespace, turkish_lower


def test_trims_and_collapses_spaces():
    assert normalize_whitespace("  X   firmasinin   satislari  ") == "X firmasinin satislari"


def test_normalizes_non_breaking_space():
    assert normalize_whitespace("X firmasi") == "X firmasi"


def test_removes_zero_width_characters():
    assert normalize_whitespace("X​firmasi﻿") == "Xfirmasi"


def test_normalizes_tabs_and_newlines():
    assert normalize_whitespace("X\tfirmasi\nsatislari") == "X firmasi satislari"


def test_does_not_remove_words():
    """Spec §18: gereksiz gorunuyor diye kelime silinmez."""
    text = "Mumkunse X firmasinin satis sayisini getirir misin?"
    assert normalize_whitespace(text) == text


def test_preserves_turkish_characters():
    text = "İstanbul şubesindeki satışları göster"
    assert normalize_whitespace(text) == text


def test_turkish_lower_dotted_capital_i():
    assert turkish_lower("İstanbul") == "istanbul"


def test_turkish_lower_dotless_capital_i():
    assert turkish_lower("IZMIR") == "ızmır"


def test_dedup_key_ignores_trailing_punctuation():
    assert dedup_key("X firmasının satışlarını getir.") == dedup_key(
        "X firmasının satışlarını getir"
    )


def test_dedup_key_ignores_case_and_spacing():
    assert dedup_key("  X FİRMASININ   satışları ") == dedup_key("x firmasının satışları")


def test_dedup_key_separates_different_questions():
    assert dedup_key("X firmasının satışları") != dedup_key("Y firmasının satışları")
