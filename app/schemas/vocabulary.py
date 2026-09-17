"""V1 sentetik alan adlari (spec §5).

Bunlar gecicidir. Isverenin gercek veritabani semasi geldiginde bu dosya
degisecek; Pydantic modelleri degismeyecek. Bu ayrim bilinclidir.

Bu listeler sema dogrulamasinda ZORLANMAZ; yalnizca
`scripts/validate_dataset.py --strict-vocabulary` ile uygulanir.
"""

from __future__ import annotations

KNOWN_FIELDS: frozenset[str] = frozenset(
    {
        "company",
        "product",
        "customer",
        "category",
        "city",
        "region",
        "store",
        "salesperson",
        "date",
        "quantity",
        "sales_amount",
        "unit_price",
        "status",
    }
)

KNOWN_TARGETS: frozenset[str] = frozenset(
    {
        "sales",
        "sales_amount",
        "quantity",
        "customer",
        "product",
        "order",
    }
)


def unknown_fields(names: list[str]) -> list[str]:
    return sorted({n for n in names if n not in KNOWN_FIELDS})


def is_known_target(target: str) -> bool:
    return target in KNOWN_TARGETS
