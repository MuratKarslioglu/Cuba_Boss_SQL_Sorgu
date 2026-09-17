"""Degerlendirme metrikleri (spec §22).

Bu modul torch'tan BAGIMSIZDIR: girdi olarak beklenen bir `QueryRequest` ile
modelin urettigi `ExtractionResult`i (bkz. `app.inference.parsing`) alir.
Boylece gercek model cikarimindan tamamen ayri, hizli ve saf test edilebilir;
`scripts/benchmark.py` gercek modelle uretilen kayitlari buraya besler.

BLEU/ROUGE KULLANILMAZ (spec §22). Hesaplanan metrikler:
    - valid JSON orani            (§22.1)
    - sema gecerlilik orani       (§22.2)
    - strict exact match          (§22.3)
    - alan bazli dogruluk         (§22.4)
    - semantic structural match   (§22.5)

Semantik esitlik ve alan-bazli dogruluk icin tam algoritma spec'te
tarif edilmemistir; asagidaki tasarim kararlari bilerek belgelenmistir:

    - filtrelerin ve group_by'in SIRASI onemsizdir (spec §22.5 ornegi),
      ancak coklukleri (bir alanin kac kez gectigi) esit olmalidir -- bu
      yuzden `set` degil `Counter` (multiset) kullanilir.
    - operation/target/order_by/limit skaler oldugundan tam esitlik aranir.
    - Gecersiz JSON/sema orneklerinde tahmin edilen bir QueryRequest
      olmadigindan, o ornekteki TUM alanlar yanlis sayilir; boylece her
      alanin paydasi hep `total_examples` olur ve oranlar karsilastirilabilir
      kalir.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from app.inference.parsing import ExtractionResult
from app.schemas.enums import QueryStatus
from app.schemas.query import FilterCondition, QueryRequest, to_canonical_json

if TYPE_CHECKING:
    from training.dataset import ExampleMetadata

# Bir kayit extraction'da tamamen basarisiz olduysa (JSON/sema hatasi), model
# hicbir status sinyali VEREMEMISTIR -- bu, "valid" ile ayni sey degildir ve
# reddetme metriklerinde ayri bir etiket olarak ele alinir (bkz. `evaluate_rejection`).
EXTRACTION_FAILED_LABEL = "extraction_failed"


class RecordStatus(str, Enum):
    OK = "ok"  # gecerli JSON + semaya uygun
    JSON_INVALID = "json_invalid"  # JSON izole/parse edilemedi (veya bos girdi)
    SCHEMA_INVALID = "schema_invalid"  # JSON gecerli ama QueryRequest semasina uymuyor


def classify(result: ExtractionResult) -> RecordStatus:
    if result.success:
        return RecordStatus.OK
    if result.error is not None and result.error.type == "JSON_PARSE_ERROR":
        return RecordStatus.JSON_INVALID
    return RecordStatus.SCHEMA_INVALID


@dataclass
class EvaluationRecord:
    """Tek bir test ornegi icin beklenen/tahmin edilen cift + zamanlama."""

    input: str
    expected: QueryRequest
    result: ExtractionResult
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    # V2: status/family/noise bazinda kirilim raporlamak icin (bkz. group_summaries).
    metadata: ExampleMetadata | None = None

    @property
    def status(self) -> RecordStatus:
        return classify(self.result)

    @property
    def predicted_status_label(self) -> str:
        """Modelin urettigi V2 `status` degeri, ya da extraction basarisizsa
        `EXTRACTION_FAILED_LABEL` (spec: 'valid' ile karistirilmamalidir)."""
        if self.status is RecordStatus.OK and self.result.data is not None:
            return self.result.data.status.value
        return EXTRACTION_FAILED_LABEL

    def to_dict(self) -> dict[str, object]:
        """Ornek bazli {input, expected, predicted, error, exact_match, latency_ms}.

        `scripts/benchmark.py`'nin `--dump_predictions` ciktisi bunu kullanir;
        siniflandirma/exact-match `classify`/`exact_match` ile ayni kaynaktan gelir.
        """
        predicted = self.result.data if self.status is RecordStatus.OK else None
        return {
            "input": self.input,
            "expected": self.expected.model_dump(mode="json"),
            "predicted": predicted.model_dump(mode="json") if predicted is not None else None,
            "error": self.result.error.model_dump(mode="json") if self.result.error else None,
            "exact_match": predicted is not None and exact_match(self.expected, predicted),
            "latency_ms": self.latency_ms,
        }


def _filter_signature(f: FilterCondition) -> tuple[str, str, str]:
    """Bir filtreyi sira/nesne kimliginden bagimsiz, karsilastirilabilir bir
    imzaya cevirir (deger, ic ice liste/None olsa da JSON'a sabit sekilde
    serilestirilerek hashlenebilir hale getirilir)."""
    return (f.field, f.operator.value, json.dumps(f.value, sort_keys=True, ensure_ascii=False))


def semantic_match(expected: QueryRequest, predicted: QueryRequest) -> bool:
    """Yapi-duyarli esitlik (spec §22.5): bkz. modul docstring'i."""
    if expected.operation != predicted.operation:
        return False
    if expected.target != predicted.target:
        return False
    if Counter(_filter_signature(f) for f in expected.filters) != Counter(
        _filter_signature(f) for f in predicted.filters
    ):
        return False
    if Counter(expected.group_by) != Counter(predicted.group_by):
        return False
    if expected.order_by != predicted.order_by:
        return False
    if expected.limit != predicted.limit:
        return False
    return True


def exact_match(expected: QueryRequest, predicted: QueryRequest) -> bool:
    """Strict exact match (spec §22.3): kanonik JSON birebir esitligi."""
    return to_canonical_json(expected) == to_canonical_json(predicted)


@dataclass
class FieldAccuracy:
    """spec §22.4'teki her alan icin dogru/toplam sayaci."""

    correct: int = 0
    total: int = 0

    def add(self, is_correct: bool) -> None:
        self.total += 1
        if is_correct:
            self.correct += 1

    @property
    def rate(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class FieldLevelAccumulator:
    """spec §22.4'teki alanlarin her biri icin ayri dogruluk takibi."""

    operation: FieldAccuracy = field(default_factory=FieldAccuracy)
    target: FieldAccuracy = field(default_factory=FieldAccuracy)
    filter_fields: FieldAccuracy = field(default_factory=FieldAccuracy)
    filter_operators: FieldAccuracy = field(default_factory=FieldAccuracy)
    filter_values: FieldAccuracy = field(default_factory=FieldAccuracy)
    group_by: FieldAccuracy = field(default_factory=FieldAccuracy)
    order_by: FieldAccuracy = field(default_factory=FieldAccuracy)
    limit: FieldAccuracy = field(default_factory=FieldAccuracy)
    status: FieldAccuracy = field(default_factory=FieldAccuracy)

    def _all(self) -> list[FieldAccuracy]:
        return [
            self.operation,
            self.target,
            self.filter_fields,
            self.filter_operators,
            self.filter_values,
            self.group_by,
            self.order_by,
            self.limit,
            self.status,
        ]

    def add(self, expected: QueryRequest, predicted: QueryRequest | None) -> None:
        """`predicted=None`: gecersiz cikti -- tum alanlar yanlis sayilir."""
        if predicted is None:
            for acc in self._all():
                acc.add(False)
            return

        self.operation.add(expected.operation == predicted.operation)
        self.target.add(expected.target == predicted.target)
        self.filter_fields.add(
            Counter(f.field for f in expected.filters)
            == Counter(f.field for f in predicted.filters)
        )
        self.filter_operators.add(
            Counter(f.operator for f in expected.filters)
            == Counter(f.operator for f in predicted.filters)
        )
        self.filter_values.add(
            Counter(_filter_signature(f) for f in expected.filters)
            == Counter(_filter_signature(f) for f in predicted.filters)
        )
        self.group_by.add(Counter(expected.group_by) == Counter(predicted.group_by))
        self.order_by.add(expected.order_by == predicted.order_by)
        self.limit.add(expected.limit == predicted.limit)
        self.status.add(expected.status == predicted.status)

    def to_dict(self) -> dict[str, float]:
        return {
            "operation": self.operation.rate,
            "target": self.target.rate,
            "filter_field": self.filter_fields.rate,
            "filter_operator": self.filter_operators.rate,
            "filter_value": self.filter_values.rate,
            "group_by": self.group_by.rate,
            "order_by": self.order_by.rate,
            "limit": self.limit.rate,
            "status": self.status.rate,
        }


@dataclass
class EvaluationSummary:
    total: int
    valid_json_rate: float
    schema_validity_rate: float
    exact_match_rate: float
    semantic_match_rate: float
    field_accuracy: dict[str, float]
    status_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_examples": self.total,
            "valid_json_rate": self.valid_json_rate,
            "schema_validity_rate": self.schema_validity_rate,
            "exact_match_rate": self.exact_match_rate,
            "semantic_match_rate": self.semantic_match_rate,
            "field_accuracy": self.field_accuracy,
            "status_counts": self.status_counts,
        }


def evaluate_records(records: list[EvaluationRecord]) -> EvaluationSummary:
    """spec §22'nin tum metriklerini tek bir gecişte hesaplar."""
    total = len(records)
    if total == 0:
        raise ValueError("degerlendirilecek en az bir kayit olmalidir")

    status_counts: Counter[str] = Counter()
    valid_json = 0
    schema_valid = 0
    exact = 0
    semantic = 0
    field_acc = FieldLevelAccumulator()

    for record in records:
        status = record.status
        status_counts[status.value] += 1

        if status is not RecordStatus.JSON_INVALID:
            valid_json += 1
        if status is RecordStatus.OK:
            schema_valid += 1

        predicted = record.result.data if status is RecordStatus.OK else None
        field_acc.add(record.expected, predicted)

        if predicted is not None:
            if exact_match(record.expected, predicted):
                exact += 1
            if semantic_match(record.expected, predicted):
                semantic += 1

    return EvaluationSummary(
        total=total,
        valid_json_rate=valid_json / total,
        schema_validity_rate=schema_valid / total,
        exact_match_rate=exact / total,
        semantic_match_rate=semantic / total,
        field_accuracy=field_acc.to_dict(),
        status_counts=dict(status_counts),
    )


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Dogrusal enterpolasyonlu yuzdelik (numpy bagimliligi olmadan)."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct
    lower = int(k)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (k - lower)


def latency_stats(records: list[EvaluationRecord]) -> dict[str, float]:
    """spec §23: ortalama/medyan/p95 gecikme (yalnizca olculmus kayitlar uzerinden)."""
    values = sorted(r.latency_ms for r in records if r.latency_ms is not None)
    if not values:
        return {"count": 0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values),
        "median_ms": _percentile(values, 0.5),
        "p95_ms": _percentile(values, 0.95),
        "min_ms": values[0],
        "max_ms": values[-1],
    }


def token_stats(records: list[EvaluationRecord]) -> dict[str, float]:
    """spec §23: ortalama prompt/uretilen token sayisi."""

    def _mean(xs: list[int]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    prompt = [r.prompt_tokens for r in records if r.prompt_tokens is not None]
    generated = [r.generated_tokens for r in records if r.generated_tokens is not None]
    return {
        "avg_prompt_tokens": _mean(prompt),
        "avg_generated_tokens": _mean(generated),
    }


@dataclass
class RejectionMetrics:
    """V2 'reddetme' davranisini ikili siniflandirma olarak olcer.

    Pozitif sinif: modelin "bu istek dogrudan cevaplanamaz" demesi
    GEREKTIGI durum (`expected.status != VALID`).

    Extraction tamamen basarisiz olursa (JSON/sema hatasi), model HICBIR
    status sinyali VEREMEMISTIR -- bu, aktif olarak "non-valid" demekten
    farklidir. Bu yuzden:

        - beklenen non-valid iken extraction basarisiz olmasi -> FN
          (sistem dogru reddetmeyi KACIRDI; en azindan "valid" da demedi,
          ama kullaniciya bir aciklama da sunamadi)
        - beklenen valid iken extraction basarisiz olmasi -> ne FP ne TN;
          `extraction_failed_on_valid` adinda AYRI, bilgilendirici bir
          sayaçta tutulur (bu zaten valid_json_rate/schema_validity_rate
          tarafindan da yakalanan bir bozulmadir; precision'i etkilememelidir)
    """

    true_positive: int = 0  # beklenen non-valid, basariyla non-valid tahmin edildi
    false_positive: int = 0  # beklenen valid, basariyla non-valid tahmin edildi (asiri red)
    false_negative: int = 0  # beklenen non-valid, ama valid dendi VEYA extraction basarisiz oldu
    true_negative: int = 0  # beklenen valid, basariyla valid tahmin edildi
    extraction_failed_on_valid: int = 0  # beklenen valid ama extraction hic basarili olmadi

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
            "extraction_failed_on_valid": self.extraction_failed_on_valid,
            "precision": self.precision,
            "recall": self.recall,
        }


def evaluate_rejection(records: list[EvaluationRecord]) -> RejectionMetrics:
    """spec (DATASET_CARD.md V2): 'rejection precision/recall'."""
    metrics = RejectionMetrics()
    for record in records:
        expected_non_valid = record.expected.status is not QueryStatus.VALID
        extraction_ok = record.status is RecordStatus.OK
        predicted_non_valid = extraction_ok and record.result.data.status is not QueryStatus.VALID

        if expected_non_valid:
            if predicted_non_valid:
                metrics.true_positive += 1
            else:
                metrics.false_negative += 1
        elif not extraction_ok:
            metrics.extraction_failed_on_valid += 1
        elif predicted_non_valid:
            metrics.false_positive += 1
        else:
            metrics.true_negative += 1
    return metrics


def status_confusion_matrix(records: list[EvaluationRecord]) -> dict[str, dict[str, int]]:
    """expected.status -> tahmin edilen status (veya `extraction_failed`) -> sayi.

    spec (DATASET_CARD.md V2): 'status confusion matrix'.
    """
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        matrix[record.expected.status.value][record.predicted_status_label] += 1
    return {expected: dict(predicted) for expected, predicted in matrix.items()}


def group_summaries(
    records: list[EvaluationRecord], key_fn: Callable[[EvaluationRecord], str | None]
) -> dict[str, EvaluationSummary]:
    """Kayitlari `key_fn`e gore gruplayip her grup icin ayri bir ozet uretir.

    spec (DATASET_CARD.md V2): 'Report metrics by status, family, and noise'.
    `key_fn` bir kayit icin None donerse (ornegin metadata eksikse) o kayit
    hicbir gruba dahil edilmez -- tahmin uretmek yerine dislanir.
    """
    groups: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        key = key_fn(record)
        if key is not None:
            groups[key].append(record)
    return {key: evaluate_records(group) for key, group in groups.items()}


def group_by_status(records: list[EvaluationRecord]) -> dict[str, EvaluationSummary]:
    return group_summaries(records, lambda r: r.expected.status.value)


def group_by_family(records: list[EvaluationRecord]) -> dict[str, EvaluationSummary]:
    return group_summaries(
        records, lambda r: r.metadata.family if r.metadata and r.metadata.family else None
    )


def group_by_noise(records: list[EvaluationRecord]) -> dict[str, EvaluationSummary]:
    return group_summaries(
        records, lambda r: r.metadata.noise if r.metadata and r.metadata.noise else None
    )
