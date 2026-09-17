"""JSONL dataset yukleme ve butunluk kontrolleri (spec §9, §10, §21).

Tasarim karari: hatalar ilk bulundugunda firlatilmaz, hepsi toplanir.
Boylece `scripts/validate_dataset.py` bozuk bir dosyanin tum sorunlarini
tek geciste raporlayabilir.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.normalization.text import dedup_key, normalize_whitespace
from app.schemas.query import QueryRequest


class ExampleMetadata(BaseModel):
    """Uretici tarafindan eklenen izlenebilirlik bilgisi (opsiyonel).

    Spec §9'un minimal formatinda yer almaz ama gercek dataset uretimleri
    ekliyor. Farkli dataset uretimlerinin alanlarini birlikte destekler:

        V1.1:              level, split, tags
        V2:                split, family, noise, difficulty
        Harici (donusum):  source (ornegin "turspider", "tur2sql")

    Hicbiri zorunlu degildir; yoksa RawExample.metadata None kalir.
    """

    model_config = ConfigDict(extra="forbid")

    split: str | None = None
    level: str | None = None
    tags: list[str] = Field(default_factory=list)
    family: str | None = None
    noise: str | None = None
    difficulty: str | None = None
    source: str | None = None


class RawExample(BaseModel):
    """Dataset dosyasindaki tek bir denetimli ornek (spec §9).

    `id` ve `metadata` spec'in minimal sozlesmesinde yoktur ama gercek
    dataset uretimlerinde izlenebilirlik icin yaygindir; bu yuzden birinci
    sinif ve opsiyonel alanlar olarak taninirlar. `extra="forbid"` yine de
    yazim hatasi/uydurma anahtarlari yakalamaya devam eder.
    """

    model_config = ConfigDict(extra="forbid")

    input: str
    output: QueryRequest
    id: str | None = None
    metadata: ExampleMetadata | None = None

    @field_validator("input")
    @classmethod
    def _validate_input(cls, v: str) -> str:
        if not normalize_whitespace(v):
            raise ValueError("input bos olamaz")
        return v


@dataclass(frozen=True)
class DatasetIssue:
    """Tek bir sorun. `line` 1-tabanlidir; dosya disi sorunlarda None."""

    path: str
    line: int | None
    code: str
    message: str

    def format(self) -> str:
        where = f"{self.path}:{self.line}" if self.line is not None else self.path
        return f"{where}: [{self.code}] {self.message}"


@dataclass
class LoadResult:
    path: Path
    examples: list[RawExample] = field(default_factory=list)
    # Basariyla yuklenen orneklerin satir numaralari (examples ile ayni sirada).
    line_numbers: list[int] = field(default_factory=list)
    issues: list[DatasetIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def iter_jsonl(path: Path) -> Iterator[tuple[int, str]]:
    """Dosyayi (1-tabanli satir no, ham satir) olarak uretir. Bos satirlar atlanir."""
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if raw.strip():
                yield line_no, raw


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def load_examples(path: Path) -> LoadResult:
    """Bir JSONL split'ini yukler ve tum sorunlari toplar."""
    result = LoadResult(path=path)

    if not path.exists():
        result.issues.append(DatasetIssue(str(path), None, "FILE_NOT_FOUND", "dosya bulunamadi"))
        return result

    for line_no, raw in iter_jsonl(path):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            result.issues.append(
                DatasetIssue(str(path), line_no, "INVALID_JSON", f"gecersiz JSON: {exc.msg}")
            )
            continue

        if not isinstance(payload, dict):
            result.issues.append(
                DatasetIssue(
                    str(path), line_no, "NOT_AN_OBJECT", "satir bir JSON nesnesi olmalidir"
                )
            )
            continue

        for key in ("input", "output"):
            if key not in payload:
                result.issues.append(
                    DatasetIssue(str(path), line_no, "MISSING_KEY", f"'{key}' alani eksik")
                )
        if "input" not in payload or "output" not in payload:
            continue

        try:
            example = RawExample.model_validate(payload)
        except ValidationError as exc:
            result.issues.append(
                DatasetIssue(str(path), line_no, "SCHEMA_ERROR", _format_validation_error(exc))
            )
            continue

        result.examples.append(example)
        result.line_numbers.append(line_no)

    if not result.examples and not result.issues:
        result.issues.append(DatasetIssue(str(path), None, "EMPTY_FILE", "dosya bos"))

    return result


def find_duplicate_inputs(result: LoadResult) -> list[DatasetIssue]:
    """Ayni dosya icindeki tekrar eden girdiler (spec §21).

    Tam ayni metin DUPLICATE_INPUT, yalnizca normalize edildiginde ayni olan
    metin NEAR_DUPLICATE_INPUT olarak raporlanir.
    """
    issues: list[DatasetIssue] = []
    exact_seen: dict[str, int] = {}
    key_seen: dict[str, int] = {}

    for example, line_no in zip(result.examples, result.line_numbers, strict=True):
        text = example.input
        if text in exact_seen:
            issues.append(
                DatasetIssue(
                    str(result.path),
                    line_no,
                    "DUPLICATE_INPUT",
                    f"girdi {exact_seen[text]}. satirla birebir ayni",
                )
            )
            continue
        exact_seen[text] = line_no

        key = dedup_key(text)
        if key in key_seen:
            issues.append(
                DatasetIssue(
                    str(result.path),
                    line_no,
                    "NEAR_DUPLICATE_INPUT",
                    f"girdi {key_seen[key]}. satirla normalize edildiginde ayni",
                )
            )
            continue
        key_seen[key] = line_no

    return issues


def find_split_overlap(left: LoadResult, right: LoadResult) -> list[DatasetIssue]:
    """Iki split arasindaki veri sizintisi (spec §10)."""
    issues: list[DatasetIssue] = []

    left_by_key: dict[str, int] = {}
    left_exact: dict[str, int] = {}
    for example, line_no in zip(left.examples, left.line_numbers, strict=True):
        left_exact.setdefault(example.input, line_no)
        left_by_key.setdefault(dedup_key(example.input), line_no)

    for example, line_no in zip(right.examples, right.line_numbers, strict=True):
        if example.input in left_exact:
            issues.append(
                DatasetIssue(
                    str(right.path),
                    line_no,
                    "SPLIT_LEAKAGE",
                    f"girdi {left.path}:{left_exact[example.input]} ile birebir ayni",
                )
            )
            continue
        key = dedup_key(example.input)
        if key in left_by_key:
            issues.append(
                DatasetIssue(
                    str(right.path),
                    line_no,
                    "SPLIT_NEAR_LEAKAGE",
                    f"girdi {left.path}:{left_by_key[key]} ile normalize edildiginde ayni",
                )
            )

    return issues


def find_metadata_split_mismatch(result: LoadResult, expected_split: str) -> list[DatasetIssue]:
    """`metadata.split` dosyanin ait oldugu split ile tutarsizsa raporlar.

    Yalnizca `metadata` saglanan orneklerde calisir; saglanmamissa sessizce
    atlanir (spec'in minimal formatinda bu alan zorunlu degildir).

    `--test` bayragiyla bilerek `challenge_test.jsonl` verilmesi (V2: daha
    zorlu bir degerlendirme kumesi) mesru bir kullanimdir; bu durumda
    `metadata.split="challenge_test"` bir tutarsizlik degildir.
    """
    acceptable = {expected_split}
    if expected_split == "test":
        acceptable.add("challenge_test")

    issues: list[DatasetIssue] = []
    for example, line_no in zip(result.examples, result.line_numbers, strict=True):
        if example.metadata is None or example.metadata.split is None:
            continue
        if example.metadata.split not in acceptable:
            issues.append(
                DatasetIssue(
                    str(result.path),
                    line_no,
                    "SPLIT_METADATA_MISMATCH",
                    f"metadata.split={example.metadata.split!r} ancak dosya "
                    f"{expected_split!r} split'inde",
                )
            )
    return issues


def distribution_report(result: LoadResult) -> dict[str, dict[str, int]]:
    """Operasyon / operator / karmasiklik histogrami (spec §11; V2: status/family/noise).

    Esik zorlamaz; dagilimi gozle kontrol etmek icindir. V2'de `operation`/
    `target` non-valid orneklerde None olabilir; bu ornekler o iki sayacta
    atlanir (sayilamayan bir seyi saymaya calismak yerine).
    """
    operations: Counter[str] = Counter()
    operators: Counter[str] = Counter()
    filter_counts: Counter[str] = Counter()
    features: Counter[str] = Counter()
    targets: Counter[str] = Counter()
    levels: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    families: Counter[str] = Counter()
    noise_types: Counter[str] = Counter()

    for example in result.examples:
        query = example.output
        if query.operation is not None:
            operations[query.operation.value] += 1
        if query.target is not None:
            targets[query.target] += 1
        filter_counts[str(len(query.filters))] += 1
        for condition in query.filters:
            operators[condition.operator.value] += 1
        if query.group_by:
            features["group_by"] += 1
        if query.order_by is not None:
            features["order_by"] += 1
        if query.limit is not None:
            features["limit"] += 1
        statuses[query.status.value] += 1

        if example.metadata is not None:
            if example.metadata.level:
                levels[example.metadata.level] += 1
            if example.metadata.family:
                families[example.metadata.family] += 1
            if example.metadata.noise:
                noise_types[example.metadata.noise] += 1

    report = {
        "operation": dict(sorted(operations.items())),
        "target": dict(sorted(targets.items())),
        "filter_operator": dict(sorted(operators.items())),
        "filter_count": dict(sorted(filter_counts.items(), key=lambda kv: int(kv[0]))),
        "feature": dict(sorted(features.items())),
        "status": dict(sorted(statuses.items())),
    }
    if levels:
        report["level"] = dict(sorted(levels.items()))
    if families:
        report["family"] = dict(sorted(families.items()))
    if noise_types:
        report["noise"] = dict(sorted(noise_types.items()))
    return report


def collect_field_names(result: LoadResult) -> dict[str, list[int]]:
    """Datasette gecen tum filtre/group_by/order_by alan adlari -> satir numaralari."""
    seen: dict[str, list[int]] = defaultdict(list)
    for example, line_no in zip(result.examples, result.line_numbers, strict=True):
        query = example.output
        for condition in query.filters:
            seen[condition.field].append(line_no)
        for name in query.group_by:
            seen[name].append(line_no)
        if query.order_by is not None:
            seen[query.order_by.field].append(line_no)
    return dict(seen)
