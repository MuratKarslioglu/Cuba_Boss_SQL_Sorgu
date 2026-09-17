"""Filtre celiski tespiti testleri (app.schemas.semantics).

Bu senaryolarin en onemlisi gercek bir model calistirmasinda gozlemlendi:
"ABC ile XYZ firmalarini karsilastir" istegi icin model ayni alanda iki
farkli 'eq' degeri uretti -- semaya uyuyordu ama SQL'e cevrilince hicbir
satirla eslesmezdi.
"""

import pytest
from pydantic import ValidationError

from app.schemas.query import FilterCondition, QueryRequest
from app.schemas.semantics import find_filter_contradictions


def fc(field: str, operator: str, value) -> FilterCondition:
    return FilterCondition.model_validate({"field": field, "operator": operator, "value": value})


def q(filters: list[dict], **overrides) -> dict:
    payload = {
        "operation": "count",
        "target": "sales",
        "filters": filters,
        "group_by": [],
        "order_by": None,
        "limit": None,
    }
    payload.update(overrides)
    return payload


# --- find_filter_contradictions: dogrudan birim testleri ---


def test_no_contradiction_for_empty_filters():
    assert find_filter_contradictions([]) == []


def test_no_contradiction_for_different_fields():
    filters = [fc("company", "eq", "ABC"), fc("product", "eq", "Y")]
    assert find_filter_contradictions(filters) == []


def test_conflicting_eq_values_on_same_field():
    """Gercek dunyadan gozlemlenen senaryo: ABC/XYZ karsilastirma sorgusu."""
    filters = [fc("company", "eq", "XYZ"), fc("company", "eq", "ABC")]
    issues = find_filter_contradictions(filters)
    assert len(issues) == 1
    assert issues[0].field == "company"
    assert "celisen" in issues[0].message


def test_repeated_identical_eq_is_not_a_contradiction():
    filters = [fc("company", "eq", "ABC"), fc("company", "eq", "ABC")]
    assert find_filter_contradictions(filters) == []


def test_eq_and_neq_same_value_conflict():
    filters = [fc("status", "eq", "aktif"), fc("status", "neq", "aktif")]
    issues = find_filter_contradictions(filters)
    assert len(issues) == 1
    assert "neq" in issues[0].message


def test_eq_and_neq_different_value_is_fine():
    filters = [fc("status", "eq", "aktif"), fc("status", "neq", "pasif")]
    assert find_filter_contradictions(filters) == []


def test_multiple_neq_different_values_is_fine():
    """field != A AND field != B tatmin edilebilir bir kisittir."""
    filters = [fc("status", "neq", "aktif"), fc("status", "neq", "pasif")]
    assert find_filter_contradictions(filters) == []


def test_eq_not_in_allowed_in_list():
    filters = [fc("city", "eq", "Ankara"), fc("city", "in", ["Istanbul", "Izmir"])]
    issues = find_filter_contradictions(filters)
    assert len(issues) == 1
    assert "in" in issues[0].message


def test_eq_inside_in_list_is_fine():
    filters = [fc("city", "eq", "Ankara"), fc("city", "in", ["Ankara", "Izmir"])]
    assert find_filter_contradictions(filters) == []


# --- sayisal/tarih araligi celiskileri ---


def test_numeric_range_contradiction():
    filters = [fc("quantity", "gt", 100), fc("quantity", "lt", 50)]
    issues = find_filter_contradictions(filters)
    assert len(issues) == 1
    assert "bos aralik" in issues[0].message


def test_numeric_range_touching_strict_bounds_is_contradiction():
    """gt 100 ve lt 100 -- (100,100) araligi bostur."""
    filters = [fc("quantity", "gt", 100), fc("quantity", "lt", 100)]
    assert len(find_filter_contradictions(filters)) == 1


def test_numeric_range_touching_inclusive_bounds_is_valid():
    """gte 100 ve lte 100 -- tam olarak 100 degeri gecerlidir."""
    filters = [fc("quantity", "gte", 100), fc("quantity", "lte", 100)]
    assert find_filter_contradictions(filters) == []


def test_valid_numeric_range_is_fine():
    filters = [fc("quantity", "gte", 10), fc("quantity", "lte", 100)]
    assert find_filter_contradictions(filters) == []


def test_date_range_contradiction():
    filters = [fc("date", "gte", "2026-06-01"), fc("date", "lte", "2026-01-01")]
    issues = find_filter_contradictions(filters)
    assert len(issues) == 1


def test_valid_date_range_is_fine():
    filters = [fc("date", "gte", "2026-01-01"), fc("date", "lte", "2026-06-30")]
    assert find_filter_contradictions(filters) == []


def test_between_conflicting_with_gt():
    """between [10,20] ile gt 30 ayni alanda celisir."""
    filters = [fc("quantity", "between", [10, 20]), fc("quantity", "gt", 30)]
    assert len(find_filter_contradictions(filters)) == 1


def test_between_compatible_with_gt():
    filters = [fc("quantity", "between", [10, 100]), fc("quantity", "gt", 5)]
    assert find_filter_contradictions(filters) == []


def test_eq_outside_range_is_contradiction():
    filters = [fc("quantity", "eq", 5), fc("quantity", "gt", 100)]
    issues = find_filter_contradictions(filters)
    assert len(issues) == 1


def test_eq_inside_range_is_fine():
    filters = [fc("quantity", "eq", 50), fc("quantity", "gt", 10), fc("quantity", "lt", 100)]
    assert find_filter_contradictions(filters) == []


def test_eq_at_exclusive_boundary_is_contradiction():
    """eq 100 ve gt 100 -- 100, (100, +inf) araliginda degildir."""
    filters = [fc("quantity", "eq", 100), fc("quantity", "gt", 100)]
    assert len(find_filter_contradictions(filters)) == 1


def test_eq_at_inclusive_boundary_is_fine():
    filters = [fc("quantity", "eq", 100), fc("quantity", "gte", 100)]
    assert find_filter_contradictions(filters) == []


# --- guvenli taraf: karsilastirilamayan turler icin sessiz kal ---


def test_mixed_numeric_and_date_bounds_are_not_compared():
    """Farkli turdeki sinirlar (sayi vs tarih) guvenle karsilastirilamaz;
    yanlis pozitif uretmemek icin hicbir celiski raporlanmaz."""
    filters = [fc("mixed_field", "gt", 100), fc("mixed_field", "lt", "2026-01-01")]
    assert find_filter_contradictions(filters) == []


def test_non_comparable_string_values_are_not_compared():
    filters = [fc("company", "gt", "ABC"), fc("company", "lt", "XYZ")]
    assert find_filter_contradictions(filters) == []


def test_contains_operator_has_no_contradiction_rule():
    filters = [fc("customer", "contains", "AŞ"), fc("customer", "eq", "Coca Cola AŞ")]
    assert find_filter_contradictions(filters) == []


# --- QueryRequest entegrasyonu: gercekten reddediyor mu ---


def test_query_request_rejects_conflicting_company_filters():
    """Gercek model ciktisi: 'ABC ile XYZ firmalarini karsilastir'."""
    payload = q(
        [
            {"field": "company", "operator": "eq", "value": "XYZ"},
            {"field": "company", "operator": "eq", "value": "ABC"},
        ]
    )
    with pytest.raises(ValidationError, match="celisiyor"):
        QueryRequest.model_validate(payload)


def test_query_request_accepts_valid_multi_field_filters():
    payload = q(
        [
            {"field": "company", "operator": "eq", "value": "ABC"},
            {"field": "product", "operator": "eq", "value": "Y"},
        ]
    )
    assert QueryRequest.model_validate(payload) is not None


def test_query_request_accepts_valid_range_via_two_conditions():
    payload = q(
        [
            {"field": "quantity", "operator": "gte", "value": 10},
            {"field": "quantity", "operator": "lte", "value": 100},
        ]
    )
    assert QueryRequest.model_validate(payload) is not None


def test_query_request_rejects_empty_range():
    payload = q(
        [
            {"field": "quantity", "operator": "gt", "value": 100},
            {"field": "quantity", "operator": "lt", "value": 50},
        ]
    )
    with pytest.raises(ValidationError, match="celisiyor"):
        QueryRequest.model_validate(payload)


def test_query_request_with_no_filters_is_unaffected():
    assert QueryRequest.model_validate(q([])) is not None
