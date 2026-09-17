"""Ortak test yardimcilari.

Repoda dataset dosyasi tutulmaz (spec §38: dataset ayrica saglanacak);
her test kendi gecici JSONL'ini uretir.
"""

import json
from pathlib import Path

import pytest


def example(text: str, **overrides) -> dict:
    """Gecerli bir dataset satiri uretir."""
    output = {
        "operation": "count",
        "target": "sales",
        "filters": [{"field": "company", "operator": "eq", "value": "X"}],
        "group_by": [],
        "order_by": None,
        "limit": None,
    }
    output.update(overrides)
    return {"input": text, "output": output}


@pytest.fixture
def write_jsonl(tmp_path: Path):
    """Satirlari gecici bir .jsonl dosyasina yazar ve yolunu doner.

    Satirlar dict ise JSON'a cevrilir; str ise oldugu gibi yazilir
    (bozuk JSON senaryolari icin).
    """

    def _write(name: str, rows) -> Path:
        path = tmp_path / name
        lines = []
        for row in rows:
            lines.append(row if isinstance(row, str) else json.dumps(row, ensure_ascii=False))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    return _write
