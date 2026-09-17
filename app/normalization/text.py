"""Hafif metin on isleme (spec §18).

Kullanici girdisi agresif sekilde degistirilmez. Kelime silinmez, dolgu
ifadeleri atilmaz; anlamsal yorum modele aittir.

`normalize_whitespace` modele giden metin uzerinde calisir.
`dedup_key` YALNIZCA dataset icindeki tekrar/sizinti tespiti icindir ve
uretilen deger asla modele veya egitim verisine yazilmaz.
"""

from __future__ import annotations

import re
import unicodedata

# Unicode bosluk benzeri karakterler: NBSP, ince bosluk, satir ayiricilar vb.
_UNICODE_SPACES = "               　"
_SPACE_TRANSLATION = {ord(ch): " " for ch in _UNICODE_SPACES}

# Genislik sifir karakterler: tamamen silinir (gorunmez, tokenizer'i bozar).
_ZERO_WIDTH = "​‌‍﻿"
_ZERO_WIDTH_TRANSLATION = {ord(ch): None for ch in _ZERO_WIDTH}

_MULTISPACE = re.compile(r"\s+")

# Turkce'ye ozgu buyuk/kucuk harf eslemesi.
# str.lower() "I" -> "i" verir, oysa Turkce'de "I" -> "ı" olmalidir.
_TR_LOWER_MAP = {
    ord("İ"): "i",
    ord("I"): "ı",
}


def normalize_whitespace(text: str) -> str:
    """Bosluklari duzlestirir ve kirpar. Kelime icerigine dokunmaz."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.translate(_ZERO_WIDTH_TRANSLATION)
    normalized = normalized.translate(_SPACE_TRANSLATION)
    return _MULTISPACE.sub(" ", normalized).strip()


def turkish_lower(text: str) -> str:
    """Turkce nokta kurallarina saygi duyan kucuk harfe cevirme."""
    return text.translate(_TR_LOWER_MAP).lower()


def dedup_key(text: str) -> str:
    """Iki girdinin 'ayni soru' sayilip sayilmayacagini belirleyen anahtar.

    Yalnizca dataset analizi icin kullanilir (spec §10, §21).
    """
    key = turkish_lower(normalize_whitespace(text))
    # Sondaki noktalama: "... getir." ile "... getir" ayni kabul edilir.
    key = key.rstrip(".?!…,;: ")
    return key
