"""Degerlendirme metrikleri testleri (spec §22, §24).

Bu modul torch gerektirmez -- `training.evaluate` tamamen saftir.
"""

import pytest

from app.inference.parsing import ExtractionResult, JsonParseError, QueryValidationError
from app.schemas.query import QueryRequest
from training.evaluate import (
    EvaluationRecord,
    RecordStatus,
    classify,
    evaluate_records,
    exact_match,
    latency_stats,
    semantic_match,
    token_stats,
)


def q(**overrides) -> QueryRequest:
    payload = {
        "operation": "count",
        "target": "sales",
        "filters": [{"field": "company", "operator": "eq", "value": "X"}],
        "group_by": [],
        "order_by": None,
        "limit": None,
    }
    payload.update(overrides)
    return QueryRequest.model_validate(payload)


def record(expected: QueryRequest, result: ExtractionResult, **kwargs) -> EvaluationRecord:
    return EvaluationRecord(input="dummy", expected=expected, result=result, **kwargs)


# --- classify / RecordStatus ---


def test_classify_success_is_ok():
    assert classify(ExtractionResult.ok(q())) is RecordStatus.OK


def test_classify_json_parse_error():
    assert classify(ExtractionResult.fail(JsonParseError("x"))) is RecordStatus.JSON_INVALID


def test_classify_validation_error_is_schema_invalid():
    assert classify(ExtractionResult.fail(QueryValidationError("x"))) is RecordStatus.SCHEMA_INVALID


# --- exact_match ---


def test_exact_match_identical():
    assert exact_match(q(), q()) is True


def test_exact_match_different_value():
    a = q(filters=[{"field": "company", "operator": "eq", "value": "X"}])
    b = q(filters=[{"field": "company", "operator": "eq", "value": "Y"}])
    assert exact_match(a, b) is False


def test_exact_match_cares_about_filter_order():
    """Strict exact match spec'te sira-duyarlidir; semantic_match sira-duyarsizdir."""
    a = q(
        filters=[
            {"field": "company", "operator": "eq", "value": "X"},
            {"field": "product", "operator": "eq", "value": "Y"},
        ]
    )
    b = q(
        filters=[
            {"field": "product", "operator": "eq", "value": "Y"},
            {"field": "company", "operator": "eq", "value": "X"},
        ]
    )
    assert exact_match(a, b) is False
    assert semantic_match(a, b) is True


# --- semantic_match ---


def test_semantic_match_ignores_filter_order():
    a = q(
        filters=[
            {"field": "company", "operator": "eq", "value": "X"},
            {"field": "product", "operator": "eq", "value": "Y"},
        ]
    )
    b = q(
        filters=[
            {"field": "product", "operator": "eq", "value": "Y"},
            {"field": "company", "operator": "eq", "value": "X"},
        ]
    )
    assert semantic_match(a, b) is True


def test_semantic_match_ignores_group_by_order():
    a = q(group_by=["product", "city"])
    b = q(group_by=["city", "product"])
    assert semantic_match(a, b) is True


def test_semantic_match_respects_filter_multiplicity():
    """Ayni imzali filtre iki kez varsa, tek sefer olan esit sayilmamalidir."""
    a = q(
        filters=[
            {"field": "company", "operator": "eq", "value": "X"},
            {"field": "company", "operator": "eq", "value": "X"},
        ]
    )
    b = q(filters=[{"field": "company", "operator": "eq", "value": "X"}])
    assert semantic_match(a, b) is False


def test_semantic_match_false_on_different_operation():
    assert semantic_match(q(operation="count"), q(operation="sum")) is False


def test_semantic_match_false_on_different_order_by():
    a = q(order_by={"field": "sales", "direction": "desc"})
    b = q(order_by={"field": "sales", "direction": "asc"})
    assert semantic_match(a, b) is False


def test_semantic_match_false_on_different_limit():
    assert semantic_match(q(limit=5), q(limit=10)) is False


def test_semantic_match_true_for_identical():
    assert semantic_match(q(), q()) is True


# --- evaluate_records: temel oranlar ---


def test_evaluate_records_requires_at_least_one():
    with pytest.raises(ValueError):
        evaluate_records([])


def test_all_correct_gives_perfect_scores():
    records = [record(q(), ExtractionResult.ok(q())) for _ in range(5)]
    summary = evaluate_records(records)
    assert summary.valid_json_rate == 1.0
    assert summary.schema_validity_rate == 1.0
    assert summary.exact_match_rate == 1.0
    assert summary.semantic_match_rate == 1.0
    assert summary.status_counts == {"ok": 5}


def test_json_invalid_counts_against_all_rates():
    records = [record(q(), ExtractionResult.fail(JsonParseError("x")))]
    summary = evaluate_records(records)
    assert summary.valid_json_rate == 0.0
    assert summary.schema_validity_rate == 0.0
    assert summary.exact_match_rate == 0.0
    assert summary.semantic_match_rate == 0.0
    assert summary.status_counts == {"json_invalid": 1}


def test_schema_invalid_counts_as_valid_json_but_not_schema_valid():
    records = [record(q(), ExtractionResult.fail(QueryValidationError("x")))]
    summary = evaluate_records(records)
    assert summary.valid_json_rate == 1.0
    assert summary.schema_validity_rate == 0.0


def test_mixed_batch_rates():
    records = [
        record(q(), ExtractionResult.ok(q())),  # ok, exact
        record(q(), ExtractionResult.ok(q(target="sales_amount"))),  # ok ama yanlis
        record(q(), ExtractionResult.fail(QueryValidationError("x"))),  # sema gecersiz
        record(q(), ExtractionResult.fail(JsonParseError("x"))),  # json gecersiz
    ]
    summary = evaluate_records(records)
    assert summary.total == 4
    assert summary.valid_json_rate == 0.75  # 3/4 (json_invalid haric)
    assert summary.schema_validity_rate == 0.5  # 2/4
    assert summary.exact_match_rate == 0.25  # yalnizca ilk kayit
    assert summary.semantic_match_rate == 0.25


# --- field-level dogruluk ---


def test_field_accuracy_all_correct():
    records = [record(q(), ExtractionResult.ok(q())) for _ in range(3)]
    summary = evaluate_records(records)
    assert all(v == 1.0 for v in summary.field_accuracy.values())


def test_field_accuracy_invalid_output_counts_all_fields_wrong():
    records = [record(q(), ExtractionResult.fail(JsonParseError("x")))]
    summary = evaluate_records(records)
    assert all(v == 0.0 for v in summary.field_accuracy.values())


def test_field_accuracy_isolates_operation_error():
    records = [record(q(), ExtractionResult.ok(q(operation="sum")))]
    summary = evaluate_records(records)
    assert summary.field_accuracy["operation"] == 0.0
    assert summary.field_accuracy["target"] == 1.0
    assert summary.field_accuracy["filter_field"] == 1.0


def test_field_accuracy_filter_field_vs_value_distinction():
    """Alan adi dogru ama deger yanlissa yalnizca filter_value dusmeli."""
    expected = q(filters=[{"field": "company", "operator": "eq", "value": "X"}])
    predicted = q(filters=[{"field": "company", "operator": "eq", "value": "Y"}])
    summary = evaluate_records([record(expected, ExtractionResult.ok(predicted))])
    assert summary.field_accuracy["filter_field"] == 1.0
    assert summary.field_accuracy["filter_operator"] == 1.0
    assert summary.field_accuracy["filter_value"] == 0.0


def test_field_accuracy_group_by_order_independent():
    expected = q(group_by=["product", "city"])
    predicted = q(group_by=["city", "product"])
    summary = evaluate_records([record(expected, ExtractionResult.ok(predicted))])
    assert summary.field_accuracy["group_by"] == 1.0


def test_field_accuracy_denominator_is_total_examples():
    """Basarisiz ornekler de field-level paydaya dahil olmalidir."""
    records = [
        record(q(), ExtractionResult.ok(q())),
        record(q(), ExtractionResult.fail(JsonParseError("x"))),
    ]
    summary = evaluate_records(records)
    assert summary.field_accuracy["operation"] == 0.5


# --- latency / token istatistikleri ---


def test_latency_stats_empty():
    stats = latency_stats([])
    assert stats["count"] == 0


def test_latency_stats_basic():
    records = [record(q(), ExtractionResult.ok(q()), latency_ms=ms) for ms in [10, 20, 30, 40, 100]]
    stats = latency_stats(records)
    assert stats["count"] == 5
    assert stats["mean_ms"] == pytest.approx(40.0)
    assert stats["median_ms"] == pytest.approx(30.0)
    assert stats["min_ms"] == 10
    assert stats["max_ms"] == 100
    assert 40 <= stats["p95_ms"] <= 100


def test_latency_stats_ignores_missing_values():
    records = [
        record(q(), ExtractionResult.ok(q()), latency_ms=10),
        record(q(), ExtractionResult.ok(q()), latency_ms=None),
    ]
    assert latency_stats(records)["count"] == 1


def test_token_stats_basic():
    records = [
        record(q(), ExtractionResult.ok(q()), prompt_tokens=20, generated_tokens=10),
        record(q(), ExtractionResult.ok(q()), prompt_tokens=30, generated_tokens=20),
    ]
    stats = token_stats(records)
    assert stats["avg_prompt_tokens"] == 25.0
    assert stats["avg_generated_tokens"] == 15.0


def test_token_stats_empty():
    stats = token_stats([])
    assert stats == {"avg_prompt_tokens": 0.0, "avg_generated_tokens": 0.0}


# --- V2: status/rejection/family/noise (CLAUDE_V2_MIGRATION.md) ---

from training.dataset import ExampleMetadata  # noqa: E402
from training.evaluate import (  # noqa: E402
    EXTRACTION_FAILED_LABEL,
    RejectionMetrics,
    evaluate_rejection,
    group_by_family,
    group_by_noise,
    group_by_status,
    status_confusion_matrix,
)


def non_valid_q(status="ambiguous", clarification="netlestirin"):
    return QueryRequest.model_validate(
        {
            "operation": None,
            "target": None,
            "filters": [],
            "group_by": [],
            "order_by": None,
            "limit": None,
            "status": status,
            "clarification": clarification,
        }
    )


def rec_with_meta(expected, result, **meta_kwargs):
    metadata = ExampleMetadata(**meta_kwargs) if meta_kwargs else None
    return EvaluationRecord(input="dummy", expected=expected, result=result, metadata=metadata)


# --- predicted_status_label ---


def test_predicted_status_label_reflects_model_output():
    rec = record(q(), ExtractionResult.ok(q()))
    assert rec.predicted_status_label == "valid"


def test_predicted_status_label_for_non_valid_prediction():
    predicted = non_valid_q(status="conflicting")
    rec = record(non_valid_q(status="conflicting"), ExtractionResult.ok(predicted))
    assert rec.predicted_status_label == "conflicting"


def test_predicted_status_label_is_extraction_failed_on_json_error():
    rec = record(q(), ExtractionResult.fail(JsonParseError("x")))
    assert rec.predicted_status_label == EXTRACTION_FAILED_LABEL


def test_predicted_status_label_is_extraction_failed_on_schema_error():
    rec = record(q(), ExtractionResult.fail(QueryValidationError("x")))
    assert rec.predicted_status_label == EXTRACTION_FAILED_LABEL


# --- to_dict (scripts/benchmark.py --dump_predictions ciktisi bunu kullanir) ---


def test_to_dict_reflects_successful_exact_match():
    rec = record(q(), ExtractionResult.ok(q()))
    payload = rec.to_dict()
    assert payload["input"] == "dummy"
    assert payload["predicted"] == payload["expected"]
    assert payload["exact_match"] is True
    assert payload["error"] is None


def test_to_dict_reflects_mismatch():
    rec = record(q(), ExtractionResult.ok(q(target="products")))
    payload = rec.to_dict()
    assert payload["exact_match"] is False


def test_to_dict_on_extraction_failure_has_no_prediction():
    rec = record(q(), ExtractionResult.fail(JsonParseError("kirik")))
    payload = rec.to_dict()
    assert payload["predicted"] is None
    assert payload["exact_match"] is False
    assert payload["error"]["type"] == "JSON_PARSE_ERROR"


# --- status field-level accuracy ---


def test_field_accuracy_includes_status():
    summary = evaluate_records([record(q(), ExtractionResult.ok(q()))])
    assert summary.field_accuracy["status"] == 1.0


def test_field_accuracy_status_mismatch():
    predicted = non_valid_q(status="ambiguous")
    summary = evaluate_records([record(q(), ExtractionResult.ok(predicted))])
    assert summary.field_accuracy["status"] == 0.0


# --- evaluate_rejection: precision/recall ---


def test_rejection_true_positive_correct_rejection():
    expected = non_valid_q("ambiguous")
    predicted = non_valid_q("ambiguous")
    metrics = evaluate_rejection([record(expected, ExtractionResult.ok(predicted))])
    assert metrics.true_positive == 1
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0


def test_rejection_false_positive_over_rejection():
    """Beklenen valid ama model gereksiz yere reddetti."""
    metrics = evaluate_rejection([record(q(), ExtractionResult.ok(non_valid_q("ambiguous")))])
    assert metrics.false_positive == 1
    assert metrics.precision == 0.0


def test_rejection_false_negative_missed_rejection_via_wrong_valid_answer():
    """Beklenen non-valid ama model 'valid' bir sorgu uretti (tehlikeli senaryo)."""
    expected = non_valid_q("conflicting")
    metrics = evaluate_rejection([record(expected, ExtractionResult.ok(q()))])
    assert metrics.false_negative == 1
    assert metrics.recall == 0.0


def test_rejection_false_negative_missed_rejection_via_extraction_failure():
    """Beklenen non-valid ama extraction tamamen basarisiz oldu -- 'valid'
    sayilmaz ama yine de reddetme sinyali VERILEMEDIGI icin kacirilmis sayilir."""
    expected = non_valid_q("conflicting")
    metrics = evaluate_rejection([record(expected, ExtractionResult.fail(JsonParseError("x")))])
    assert metrics.false_negative == 1


def test_rejection_true_negative_correct_extraction():
    metrics = evaluate_rejection([record(q(), ExtractionResult.ok(q()))])
    assert metrics.true_negative == 1


def test_rejection_precision_recall_zero_denominator():
    metrics = RejectionMetrics()
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0


def test_rejection_metrics_to_dict_shape():
    metrics = evaluate_rejection([record(q(), ExtractionResult.ok(q()))])
    payload = metrics.to_dict()
    assert set(payload) == {
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "extraction_failed_on_valid",
        "precision",
        "recall",
    }


def test_rejection_extraction_failed_on_valid_is_tracked_separately():
    """Beklenen valid ama extraction basarisiz -- ne FP ne TN, ayri sayaç."""
    metrics = evaluate_rejection([record(q(), ExtractionResult.fail(JsonParseError("x")))])
    assert metrics.extraction_failed_on_valid == 1
    assert metrics.false_positive == 0
    assert metrics.true_negative == 0
    assert metrics.precision == 0.0  # 0/0 -> guvenli varsayilan


# --- status_confusion_matrix ---


def test_confusion_matrix_correct_diagonal():
    records_ = [
        record(q(), ExtractionResult.ok(q())),
        record(non_valid_q("ambiguous"), ExtractionResult.ok(non_valid_q("ambiguous"))),
    ]
    matrix = status_confusion_matrix(records_)
    assert matrix["valid"] == {"valid": 1}
    assert matrix["ambiguous"] == {"ambiguous": 1}


def test_confusion_matrix_extraction_failure_bucket():
    matrix = status_confusion_matrix([record(q(), ExtractionResult.fail(JsonParseError("x")))])
    assert matrix["valid"] == {EXTRACTION_FAILED_LABEL: 1}


def test_confusion_matrix_off_diagonal_mismatch():
    expected = non_valid_q("conflicting")
    predicted = non_valid_q("ambiguous")
    matrix = status_confusion_matrix([record(expected, ExtractionResult.ok(predicted))])
    assert matrix["conflicting"] == {"ambiguous": 1}


# --- group_by_status / family / noise ---


def test_group_by_status_splits_correctly():
    records_ = [
        record(q(), ExtractionResult.ok(q())),
        record(q(), ExtractionResult.ok(q(target="wrong"))),
        record(non_valid_q("ambiguous"), ExtractionResult.ok(non_valid_q("ambiguous"))),
    ]
    groups = group_by_status(records_)
    assert set(groups) == {"valid", "ambiguous"}
    assert groups["valid"].total == 2
    assert groups["valid"].exact_match_rate == 0.5
    assert groups["ambiguous"].total == 1
    assert groups["ambiguous"].exact_match_rate == 1.0


def test_group_by_family_groups_and_excludes_missing_metadata():
    records_ = [
        rec_with_meta(q(), ExtractionResult.ok(q()), family="threshold"),
        rec_with_meta(q(), ExtractionResult.ok(q()), family="threshold"),
        rec_with_meta(q(), ExtractionResult.ok(q()), family="negation"),
        record(q(), ExtractionResult.ok(q())),  # metadata yok -> disarida kalir
    ]
    groups = group_by_family(records_)
    assert set(groups) == {"threshold", "negation"}
    assert groups["threshold"].total == 2
    assert groups["negation"].total == 1


def test_group_by_noise_groups_correctly():
    records_ = [
        rec_with_meta(q(), ExtractionResult.ok(q()), noise="typo"),
        rec_with_meta(q(), ExtractionResult.ok(q(target="wrong")), noise="clean"),
    ]
    groups = group_by_noise(records_)
    assert groups["typo"].exact_match_rate == 1.0
    assert groups["clean"].exact_match_rate == 0.0


def test_group_summaries_empty_input_returns_empty_dict():
    assert group_by_status([]) == {}
