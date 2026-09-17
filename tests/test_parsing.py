"""JSON izolasyon ve dogrulama testleri (spec §16, §24).

Bu modul torch gerektirmez; testler her ortamda calisir.
"""

import pytest

from app.errors import ErrorType
from app.inference.parsing import (
    ExtractionResult,
    JsonIsolationError,
    JsonParseError,
    QueryValidationError,
    isolate_json,
    parse_query_json,
)
from app.schemas.query import QueryRequest

VALID_JSON = (
    '{"operation":"count","target":"sales","filters":'
    '[{"field":"company","operator":"eq","value":"X"}],'
    '"group_by":[],"order_by":null,"limit":null}'
)


# --- isolate_json ---


def test_isolates_bare_json():
    assert isolate_json(VALID_JSON) == VALID_JSON


def test_isolates_json_with_surrounding_text():
    """Spec §16/§24: 'extra surrounding text'."""
    wrapped = f"Elbette, iste sonuc:\n{VALID_JSON}\nUmarim yardimci olmustur."
    assert isolate_json(wrapped) == VALID_JSON


def test_isolates_json_with_leading_whitespace_and_code_fence():
    wrapped = f"```json\n{VALID_JSON}\n```"
    assert isolate_json(wrapped) == VALID_JSON


def test_handles_nested_objects_correctly():
    nested = (
        '{"operation":"sum","target":"sales","filters":[],'
        '"group_by":["product"],"order_by":{"field":"sales","direction":"desc"},"limit":5}'
    )
    assert isolate_json(f"onek {nested} sonek") == nested


def test_brace_inside_string_value_does_not_confuse_scanner():
    tricky = (
        '{"operation":"count","target":"sales","filters":[],'
        '"group_by":[],"order_by":null,"limit":null,"note":"a } b"}'
    )
    assert isolate_json(tricky) == tricky


def test_escaped_quote_inside_string_does_not_end_string_early():
    tricky = '{"a":"iceren \\" tirnak ve } suslu parantez","b":1}'
    assert isolate_json(tricky) == tricky


def test_no_json_object_raises():
    with pytest.raises(JsonIsolationError):
        isolate_json("burada hicbir json yok")


def test_unbalanced_braces_raises():
    with pytest.raises(JsonIsolationError, match="kapanmadi"):
        isolate_json('{"operation":"count"')


# --- parse_query_json ---


def test_parses_valid_output():
    query = parse_query_json(VALID_JSON)
    assert isinstance(query, QueryRequest)
    assert query.operation.value == "count"


def test_parses_output_with_surrounding_text():
    query = parse_query_json(f"Sonuc: {VALID_JSON}")
    assert query.target == "sales"


def test_malformed_json_raises_json_parse_error():
    with pytest.raises(JsonParseError):
        parse_query_json('{"operation": "count", "target": }')


def test_array_only_output_raises_isolation_error():
    """`isolate_json` yalnizca `{` ile baslayan nesneleri tanir; salt bir
    dizi iceren cikti icin JSON nesnesi bulunamaz."""
    with pytest.raises(JsonIsolationError):
        parse_query_json("[1, 2, 3]")


def test_schema_violation_raises_query_validation_error():
    bad = (
        '{"operation":"fetch","target":"sales","filters":[],'
        '"group_by":[],"order_by":null,"limit":null}'
    )
    with pytest.raises(QueryValidationError):
        parse_query_json(bad)


def test_missing_required_field_raises_query_validation_error():
    with pytest.raises(QueryValidationError):
        parse_query_json('{"target":"sales"}')


def test_error_types_match_spec():
    assert JsonIsolationError("x").error_type is ErrorType.JSON_PARSE_ERROR
    assert JsonParseError("x").error_type is ErrorType.JSON_PARSE_ERROR
    assert QueryValidationError("x").error_type is ErrorType.VALIDATION_ERROR


# --- ExtractionResult ---


def test_extraction_result_ok_matches_spec_shape():
    query = parse_query_json(VALID_JSON)
    result = ExtractionResult.ok(query)
    payload = result.model_dump(mode="json")
    assert payload["success"] is True
    assert payload["error"] is None
    assert payload["data"]["operation"] == "count"


def test_extraction_result_fail_matches_spec_shape():
    result = ExtractionResult.fail(QueryValidationError("semaya uymuyor"))
    payload = result.model_dump(mode="json")
    assert payload["success"] is False
    assert payload["data"] is None
    assert payload["error"] == {"type": "VALIDATION_ERROR", "message": "semaya uymuyor"}


def test_extraction_result_rejects_extra_keys():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExtractionResult(success=True, data=None, error=None, extra_field="x")
