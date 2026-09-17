"""QueryRequest semasi testleri (spec §24)."""

import json

import pytest
from pydantic import ValidationError

from app.schemas.enums import FilterOperator, OrderDirection, QueryOperation
from app.schemas.query import (
    CANONICAL_KEY_ORDER,
    FilterCondition,
    OrderBy,
    QueryRequest,
    canonicalize_dict,
    to_canonical_json,
)

VALID = {
    "operation": "count",
    "target": "sales",
    "filters": [{"field": "company", "operator": "eq", "value": "X"}],
    "group_by": [],
    "order_by": None,
    "limit": None,
}


def test_valid_query_request():
    query = QueryRequest.model_validate(VALID)
    assert query.operation is QueryOperation.COUNT
    assert query.target == "sales"
    assert query.filters[0].operator is FilterOperator.EQ


def test_defaults_are_independent():
    a = QueryRequest(operation=QueryOperation.SELECT, target="sales")
    b = QueryRequest(operation=QueryOperation.SELECT, target="sales")
    a.filters.append(FilterCondition(field="company", operator=FilterOperator.EQ, value="X"))
    assert b.filters == []


@pytest.mark.parametrize("bad", ["Count", "greater_than", "SELECT", "fetch", ""])
def test_invalid_operation_rejected(bad):
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({**VALID, "operation": bad})


@pytest.mark.parametrize("bad", ["equals", "=", ">", "after", "before", "greater_than"])
def test_invalid_operator_rejected(bad):
    with pytest.raises(ValidationError):
        QueryRequest.model_validate(
            {**VALID, "filters": [{"field": "company", "operator": bad, "value": "X"}]}
        )


@pytest.mark.parametrize("bad", ["ascending", "DESC", "down", ""])
def test_invalid_order_direction_rejected(bad):
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({**VALID, "order_by": {"field": "sales", "direction": bad}})


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_non_positive_limit_rejected(bad):
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({**VALID, "limit": bad})


def test_positive_limit_accepted():
    assert QueryRequest.model_validate({**VALID, "limit": 5}).limit == 5


@pytest.mark.parametrize(
    "value",
    ["2026-01-01", ["2026-01-01"], ["a", "b", "c"], []],
)
def test_between_requires_exactly_two_values(value):
    with pytest.raises(ValidationError):
        FilterCondition.model_validate({"field": "date", "operator": "between", "value": value})


def test_between_with_two_values_accepted():
    condition = FilterCondition.model_validate(
        {"field": "date", "operator": "between", "value": ["2026-01-01", "2026-06-30"]}
    )
    assert condition.value == ["2026-01-01", "2026-06-30"]


@pytest.mark.parametrize("value", ["X", 5, []])
def test_in_requires_non_empty_list(value):
    with pytest.raises(ValidationError):
        FilterCondition.model_validate({"field": "city", "operator": "in", "value": value})


def test_scalar_operator_rejects_list():
    with pytest.raises(ValidationError):
        FilterCondition.model_validate({"field": "company", "operator": "eq", "value": ["X", "Y"]})


def test_null_filter_value_rejected():
    with pytest.raises(ValidationError):
        FilterCondition.model_validate({"field": "company", "operator": "eq", "value": None})


def test_empty_target_rejected():
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({**VALID, "target": "   "})


def test_empty_filter_field_rejected():
    with pytest.raises(ValidationError):
        FilterCondition.model_validate({"field": "", "operator": "eq", "value": "X"})


def test_extra_keys_forbidden_on_query():
    """Spec §16/§37: model uydurma anahtar uretirse sessizce yutulmaz."""
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({**VALID, "having": []})


def test_extra_keys_forbidden_on_filter():
    with pytest.raises(ValidationError):
        FilterCondition.model_validate(
            {"field": "company", "operator": "eq", "value": "X", "negate": True}
        )


def test_extra_keys_forbidden_on_order_by():
    with pytest.raises(ValidationError):
        OrderBy.model_validate({"field": "sales", "direction": "desc", "nulls": "last"})


def test_missing_required_field_rejected():
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({"target": "sales"})


# --- Kanonik seri hale getirme (spec §9, §14) ---


def test_canonical_json_key_order():
    rendered = to_canonical_json(QueryRequest.model_validate(VALID))
    assert list(json.loads(rendered).keys()) == list(CANONICAL_KEY_ORDER)


def test_canonical_json_has_no_whitespace():
    rendered = to_canonical_json(QueryRequest.model_validate(VALID))
    assert " " not in rendered.replace('"X"', "")
    assert ", " not in rendered
    assert ": " not in rendered


def test_canonical_json_preserves_turkish_characters():
    query = QueryRequest.model_validate(
        {**VALID, "filters": [{"field": "city", "operator": "eq", "value": "İstanbul"}]}
    )
    rendered = to_canonical_json(query)
    assert "İstanbul" in rendered
    assert "\\u" not in rendered


def test_canonical_json_matches_spec_example():
    """Spec §14'teki egitim satiriyla karakter karakter esitlik."""
    expected = (
        '{"operation":"count","target":"sales","filters":'
        '[{"field":"company","operator":"eq","value":"X"},'
        '{"field":"product","operator":"eq","value":"Y"}],'
        '"group_by":[],"order_by":null,"limit":null,'
        '"status":"valid","clarification":null}'
    )
    actual = canonicalize_dict(
        {
            "operation": "count",
            "target": "sales",
            "filters": [
                {"field": "company", "operator": "eq", "value": "X"},
                {"field": "product", "operator": "eq", "value": "Y"},
            ],
            "group_by": [],
            "order_by": None,
            "limit": None,
        }
    )
    assert actual == expected


def test_canonical_json_round_trip():
    query = QueryRequest.model_validate(
        {
            "operation": "sum",
            "target": "sales",
            "filters": [{"field": "company", "operator": "eq", "value": "ABC"}],
            "group_by": ["product"],
            "order_by": {"field": "sales", "direction": "desc"},
            "limit": 5,
        }
    )
    rendered = to_canonical_json(query)
    assert to_canonical_json(QueryRequest.model_validate(json.loads(rendered))) == rendered


def test_enum_serializes_as_plain_string():
    query = QueryRequest.model_validate(VALID)
    payload = query.model_dump(mode="json")
    assert payload["operation"] == "count"
    assert payload["filters"][0]["operator"] == "eq"


def test_order_direction_values():
    assert {d.value for d in OrderDirection} == {"asc", "desc"}


# --- V2: status/clarification sozlesmesi (CLAUDE_V2_MIGRATION.md) ---

from app.schemas.enums import QueryStatus  # noqa: E402


def _valid_payload(**overrides):
    payload = {**VALID}
    payload.update(overrides)
    return payload


def test_default_status_is_valid_for_v1_style_payload():
    """Geriye donuk uyumluluk: status/clarification hic verilmezse VALID varsayilir."""
    query = QueryRequest.model_validate(VALID)
    assert query.status is QueryStatus.VALID
    assert query.clarification is None


def test_explicit_valid_status_requires_operation_and_target():
    payload = _valid_payload(operation=None, target=None, status="valid")
    with pytest.raises(ValidationError, match="operation ve target zorunludur"):
        QueryRequest.model_validate(payload)


def test_valid_status_rejects_non_null_clarification():
    payload = _valid_payload(status="valid", clarification="bu bir sorun")
    with pytest.raises(ValidationError, match="clarification None olmalidir"):
        QueryRequest.model_validate(payload)


@pytest.mark.parametrize("status", ["ambiguous", "conflicting", "insufficient_information"])
def test_non_valid_status_requires_null_operation_and_target(status):
    payload = _valid_payload(status=status, clarification="netlestirin")
    with pytest.raises(ValidationError, match="None olmalidir"):
        QueryRequest.model_validate(payload)


@pytest.mark.parametrize("status", ["ambiguous", "conflicting", "insufficient_information"])
def test_non_valid_status_requires_clarification(status):
    payload = {
        "operation": None,
        "target": None,
        "filters": [],
        "group_by": [],
        "order_by": None,
        "limit": None,
        "status": status,
        "clarification": None,
    }
    with pytest.raises(ValidationError, match="clarification zorunludur"):
        QueryRequest.model_validate(payload)


@pytest.mark.parametrize("status", ["ambiguous", "conflicting", "insufficient_information"])
def test_non_valid_status_accepted_with_null_operation_target(status):
    payload = {
        "operation": None,
        "target": None,
        "filters": [],
        "group_by": [],
        "order_by": None,
        "limit": None,
        "status": status,
        "clarification": "netlestirin",
    }
    query = QueryRequest.model_validate(payload)
    assert query.status.value == status
    assert query.operation is None
    assert query.target is None


def test_invalid_status_value_rejected():
    payload = _valid_payload(status="rejected")
    with pytest.raises(ValidationError):
        QueryRequest.model_validate(payload)


def test_canonical_json_includes_status_and_clarification_at_end():
    rendered = to_canonical_json(QueryRequest.model_validate(VALID))
    keys = list(json.loads(rendered).keys())
    assert keys[-2:] == ["status", "clarification"]


def test_canonical_json_for_non_valid_status():
    payload = {
        "operation": None,
        "target": None,
        "filters": [],
        "group_by": [],
        "order_by": None,
        "limit": None,
        "status": "ambiguous",
        "clarification": "Hangi sirket icin?",
    }
    rendered = to_canonical_json(QueryRequest.model_validate(payload))
    assert rendered == (
        '{"operation":null,"target":null,"filters":[],"group_by":[],'
        '"order_by":null,"limit":null,"status":"ambiguous","clarification":"Hangi sirket icin?"}'
    )
