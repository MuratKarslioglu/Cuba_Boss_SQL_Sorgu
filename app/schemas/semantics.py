"""Filtre listesi icin semantik celiski tespiti (V1 ek dogrulama katmani).

Pydantic'in yapisal dogrulamasi (enum uyumu, `between`/`in` sekli, vb.)
hicbir zaman "ayni alanda iki farkli `eq` degeri" gibi MANTIKSAL celiskileri
yakalamaz -- boyle bir sorgu semaya tamamen uyar ama gercek bir SQL'e
cevrildiginde HICBIR SATIRLA eslesmez:

    WHERE company = 'ABC' AND company = 'XYZ'   -- her zaman bos sonuc

Bu, gercek bir model calistirmasinda gozlemlendi: "ABC ile XYZ firmalarini
karsilastir" istegi icin model iki ayri `company eq` filtresi uretti --
sema gecerliydi ama anlamsizdi.

Bu modul, ayni QueryRequest icindeki filtreleri alan bazinda gruplayip
asagidaki celiski siniflarini tespit eder:

    1. ayni alanda birden fazla FARKLI 'eq' degeri
    2. ayni alanda ayni degerde 'eq' ve 'neq'
    3. 'eq' degeri, ayni alandaki bir 'in' listesinin disinda
    4. siralanabilir (sayisal/ISO-tarih) degerlerde bos aralik
       (ornegin ayni alanda 'gt 100' ve 'lt 50')
    5. 'eq' degeri, ayni alandaki aralik/`between` sinirlarinin disinda

Yalnizca SAYISAL veya ISO-8601 TARIH bicimindeki degerler karsilastirilir;
karsilastirilamayan (veya turleri uyusmayan) degerler icin hicbir celiski
raporlanmaz -- yanlis pozitiften kacinmak, yakalamaktan onceliklidir.

`contains` operatoru icin bir celiski kurali YOKTUR: alt dize eslesmesi
digerleriyle guvenli bicimde birlikte akil yurutulemeyecek kadar belirsizdir.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.schemas.enums import FilterOperator

if TYPE_CHECKING:
    from app.schemas.query import FilterCondition

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class FilterContradiction:
    field: str
    message: str


def _freeze(value: Any) -> str:
    """Degeri karsilastirilabilir/hashlenebilir bir anahtara cevirir."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _as_comparable(value: Any) -> float | str | None:
    """Sayisal veya ISO-tarih bicimindeyse karsilastirilabilir bir deger doner.

    ISO-8601 tarihler ("YYYY-MM-DD...") lexicographic siralamada da dogru
    sonuc verdigi icin string olarak birakilir.
    """
    if isinstance(value, bool):
        return None  # bool, int alt sinifi oldugu icin ayrica elenir.
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and _ISO_DATE_RE.match(value):
        return value
    return None


def _max_lower_bound(conditions: list[FilterCondition]) -> tuple[float | str | None, bool]:
    """`gt`/`gte`/`between`'in alt sinirlarindan en siki olanini secer.

    Doner: (deger, strict) -- strict=True ise sinir haric (`gt`).
    Farkli tipte (biri sayi biri tarih) adaylar varsa None doner (guvenli
    karsilastirma yapilamaz).
    """
    candidates: list[tuple[float | str, bool]] = []
    for c in conditions:
        if c.operator is FilterOperator.GT:
            v = _as_comparable(c.value)
            if v is not None:
                candidates.append((v, True))
        elif c.operator is FilterOperator.GTE:
            v = _as_comparable(c.value)
            if v is not None:
                candidates.append((v, False))
        elif (
            c.operator is FilterOperator.BETWEEN and isinstance(c.value, list) and len(c.value) == 2
        ):
            v = _as_comparable(c.value[0])
            if v is not None:
                candidates.append((v, False))

    return _pick_tightest(candidates, prefer_larger=True)


def _min_upper_bound(conditions: list[FilterCondition]) -> tuple[float | str | None, bool]:
    """`lt`/`lte`/`between`'in ust sinirlarindan en siki olanini secer."""
    candidates: list[tuple[float | str, bool]] = []
    for c in conditions:
        if c.operator is FilterOperator.LT:
            v = _as_comparable(c.value)
            if v is not None:
                candidates.append((v, True))
        elif c.operator is FilterOperator.LTE:
            v = _as_comparable(c.value)
            if v is not None:
                candidates.append((v, False))
        elif (
            c.operator is FilterOperator.BETWEEN and isinstance(c.value, list) and len(c.value) == 2
        ):
            v = _as_comparable(c.value[1])
            if v is not None:
                candidates.append((v, False))

    return _pick_tightest(candidates, prefer_larger=False)


def _pick_tightest(
    candidates: list[tuple[float | str, bool]], prefer_larger: bool
) -> tuple[float | str | None, bool]:
    """Ayni turden adaylar arasindan en siki siniri secer; karisik turde
    aday varsa (sayi + tarih) guvenli karsilastirma yapilamayacagi icin
    None doner."""
    if not candidates:
        return None, False

    types = {type(v) for v, _ in candidates}
    if len(types) > 1:
        return None, False

    best_value, best_strict = candidates[0]
    for value, strict in candidates[1:]:
        is_tighter = (value > best_value) if prefer_larger else (value < best_value)
        is_tie_but_stricter = value == best_value and strict and not best_strict
        if is_tighter or is_tie_but_stricter:
            best_value, best_strict = value, strict
    return best_value, best_strict


def _check_field(field: str, conditions: list[FilterCondition]) -> list[FilterContradiction]:
    issues: list[FilterContradiction] = []

    eq_values = [c.value for c in conditions if c.operator is FilterOperator.EQ]
    neq_values = [c.value for c in conditions if c.operator is FilterOperator.NEQ]
    in_lists = [c.value for c in conditions if c.operator is FilterOperator.IN]

    distinct_eq_keys = {_freeze(v) for v in eq_values}

    # 1) ayni alanda birden fazla farkli eq degeri
    if len(distinct_eq_keys) > 1:
        rendered = ", ".join(repr(v) for v in eq_values)
        issues.append(
            FilterContradiction(field, f"'{field}' alaninda celisen 'eq' degerleri: {rendered}")
        )
        return issues  # digerleri anlamsiz -- ilerlemeye gerek yok

    single_eq = eq_values[0] if len(distinct_eq_keys) == 1 else None

    # 2) eq ve neq ayni degerde
    if single_eq is not None:
        eq_key = _freeze(single_eq)
        for nv in neq_values:
            if _freeze(nv) == eq_key:
                issues.append(
                    FilterContradiction(
                        field,
                        f"'{field}' alaninda 'eq {single_eq!r}' ve 'neq {single_eq!r}' celisiyor",
                    )
                )

    # 3) eq degeri in listesinde yok
    if single_eq is not None:
        eq_key = _freeze(single_eq)
        for lst in in_lists:
            if isinstance(lst, list) and eq_key not in {_freeze(v) for v in lst}:
                issues.append(
                    FilterContradiction(
                        field, f"'{field}' alaninda 'eq {single_eq!r}' 'in {lst!r}' listesinde yok"
                    )
                )

    # 4) siralanabilir aralik celiskisi
    lower, lower_strict = _max_lower_bound(conditions)
    upper, upper_strict = _min_upper_bound(conditions)
    if lower is not None and upper is not None and type(lower) is type(upper):
        empty_range = lower > upper or (lower == upper and (lower_strict or upper_strict))
        if empty_range:
            issues.append(
                FilterContradiction(
                    field,
                    f"'{field}' alaninda bos aralik: alt sinir {lower!r}, ust sinir {upper!r}",
                )
            )

    # 5) eq degeri aralik/between sinirlarinin disinda
    if single_eq is not None:
        comparable_eq = _as_comparable(single_eq)
        if comparable_eq is not None:
            if (
                lower is not None
                and type(lower) is type(comparable_eq)
                and (comparable_eq < lower or (comparable_eq == lower and lower_strict))
            ):
                issues.append(
                    FilterContradiction(
                        field,
                        f"'{field}' alaninda 'eq {single_eq!r}' alt sinirin disinda ({lower!r})",
                    )
                )
            if (
                upper is not None
                and type(upper) is type(comparable_eq)
                and (comparable_eq > upper or (comparable_eq == upper and upper_strict))
            ):
                issues.append(
                    FilterContradiction(
                        field,
                        f"'{field}' alaninda 'eq {single_eq!r}' ust sinirin disinda ({upper!r})",
                    )
                )

    return issues


def find_filter_contradictions(filters: list[FilterCondition]) -> list[FilterContradiction]:
    """Bir filtre listesindeki tum alan-bazli mantiksal celiskileri bulur."""
    by_field: dict[str, list[FilterCondition]] = defaultdict(list)
    for condition in filters:
        by_field[condition.field].append(condition)

    issues: list[FilterContradiction] = []
    for field, conditions in by_field.items():
        issues.extend(_check_field(field, conditions))
    return issues
