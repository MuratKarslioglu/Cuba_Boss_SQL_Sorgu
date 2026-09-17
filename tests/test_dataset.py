"""Dataset yukleme ve butunluk testleri (spec §10, §21, §24)."""

from training.dataset import (
    RawExample,
    collect_field_names,
    distribution_report,
    find_duplicate_inputs,
    find_metadata_split_mismatch,
    find_split_overlap,
    load_examples,
)

from .conftest import example


def test_loads_valid_file(write_jsonl):
    path = write_jsonl("train.jsonl", [example("X firmasının satışları"), example("Y ürünü")])
    result = load_examples(path)
    assert result.ok
    assert len(result.examples) == 2
    assert result.line_numbers == [1, 2]


def test_blank_lines_are_skipped(write_jsonl):
    path = write_jsonl("train.jsonl", [example("ilk"), "", "   ", example("ikinci")])
    result = load_examples(path)
    assert result.ok
    assert result.line_numbers == [1, 4]


def test_invalid_json_line_reported_with_line_number(write_jsonl):
    path = write_jsonl("train.jsonl", [example("ilk"), "{bozuk json", example("ucuncu")])
    result = load_examples(path)
    codes = {(i.code, i.line) for i in result.issues}
    assert ("INVALID_JSON", 2) in codes
    assert len(result.examples) == 2


def test_non_object_line_reported(write_jsonl):
    path = write_jsonl("train.jsonl", [example("ilk"), "[1, 2, 3]"])
    result = load_examples(path)
    assert any(i.code == "NOT_AN_OBJECT" and i.line == 2 for i in result.issues)


def test_missing_input_reported(write_jsonl):
    row = example("x")
    del row["input"]
    path = write_jsonl("train.jsonl", [row])
    result = load_examples(path)
    assert any(i.code == "MISSING_KEY" and "input" in i.message for i in result.issues)


def test_missing_output_reported(write_jsonl):
    row = example("x")
    del row["output"]
    path = write_jsonl("train.jsonl", [row])
    result = load_examples(path)
    assert any(i.code == "MISSING_KEY" and "output" in i.message for i in result.issues)


def test_empty_input_reported(write_jsonl):
    path = write_jsonl("train.jsonl", [example("   ")])
    result = load_examples(path)
    assert any(i.code == "SCHEMA_ERROR" for i in result.issues)


def test_schema_mismatch_reported(write_jsonl):
    path = write_jsonl("train.jsonl", [example("x", operation="fetch")])
    result = load_examples(path)
    issue = next(i for i in result.issues if i.code == "SCHEMA_ERROR")
    assert issue.line == 1
    assert "operation" in issue.message


def test_between_violation_reported(write_jsonl):
    row = example("x", filters=[{"field": "date", "operator": "between", "value": ["2026-01-01"]}])
    path = write_jsonl("train.jsonl", [row])
    result = load_examples(path)
    assert any(i.code == "SCHEMA_ERROR" for i in result.issues)


def test_extra_top_level_key_reported(write_jsonl):
    row = example("x")
    row["notes"] = "ek alan"
    path = write_jsonl("train.jsonl", [row])
    result = load_examples(path)
    assert any(i.code == "SCHEMA_ERROR" for i in result.issues)


def test_missing_file_reported(tmp_path):
    result = load_examples(tmp_path / "yok.jsonl")
    assert any(i.code == "FILE_NOT_FOUND" for i in result.issues)


def test_empty_file_reported(write_jsonl):
    path = write_jsonl("train.jsonl", [""])
    result = load_examples(path)
    assert any(i.code == "EMPTY_FILE" for i in result.issues)


# --- Tekrar ve sizinti ---


def test_exact_duplicate_detected(write_jsonl):
    text = "X firmasının satışları"
    path = write_jsonl("train.jsonl", [example(text), example("başka"), example(text)])
    issues = find_duplicate_inputs(load_examples(path))
    assert [(i.code, i.line) for i in issues] == [("DUPLICATE_INPUT", 3)]


def test_near_duplicate_detected(write_jsonl):
    path = write_jsonl(
        "train.jsonl",
        [example("X firmasının satışları."), example("x   FİRMASININ satışları")],
    )
    issues = find_duplicate_inputs(load_examples(path))
    assert [(i.code, i.line) for i in issues] == [("NEAR_DUPLICATE_INPUT", 2)]


def test_no_false_duplicate(write_jsonl):
    path = write_jsonl("train.jsonl", [example("X firması"), example("Y firması")])
    assert find_duplicate_inputs(load_examples(path)) == []


def test_split_leakage_detected(write_jsonl):
    text = "X firmasının satışları"
    train = load_examples(write_jsonl("train.jsonl", [example(text)]))
    test = load_examples(write_jsonl("test.jsonl", [example("farklı"), example(text)]))
    issues = find_split_overlap(train, test)
    assert [(i.code, i.line) for i in issues] == [("SPLIT_LEAKAGE", 2)]


def test_split_near_leakage_detected(write_jsonl):
    train = load_examples(write_jsonl("train.jsonl", [example("X firmasının satışları")]))
    test = load_examples(write_jsonl("test.jsonl", [example("x firmasının   SATIŞLARI.")]))
    issues = find_split_overlap(train, test)
    assert [i.code for i in issues] == ["SPLIT_NEAR_LEAKAGE"]


def test_clean_splits_have_no_overlap(write_jsonl):
    train = load_examples(write_jsonl("train.jsonl", [example("X firması")]))
    test = load_examples(write_jsonl("test.jsonl", [example("Y firması")]))
    assert find_split_overlap(train, test) == []


# --- Raporlama ---


def test_distribution_report(write_jsonl):
    rows = [
        example("bir"),
        example(
            "iki",
            operation="sum",
            target="sales_amount",
            filters=[{"field": "date", "operator": "between", "value": ["a", "b"]}],
            group_by=["product"],
            order_by={"field": "sales", "direction": "desc"},
            limit=5,
        ),
    ]
    report = distribution_report(load_examples(write_jsonl("train.jsonl", rows)))
    assert report["operation"] == {"count": 1, "sum": 1}
    assert report["filter_operator"] == {"between": 1, "eq": 1}
    assert report["filter_count"] == {"1": 2}
    assert report["feature"] == {"group_by": 1, "limit": 1, "order_by": 1}


def test_collect_field_names_includes_group_by_and_order_by(write_jsonl):
    rows = [
        example(
            "bir",
            group_by=["product"],
            order_by={"field": "sales", "direction": "desc"},
        )
    ]
    names = collect_field_names(load_examples(write_jsonl("train.jsonl", rows)))
    assert set(names) == {"company", "product", "sales"}


def test_raw_example_parses_nested_output():
    parsed = RawExample.model_validate(example("X firması"))
    assert parsed.output.target == "sales"


# --- id / metadata (uretici tarafindan eklenen izlenebilirlik alanlari) ---


def test_id_and_metadata_are_optional(write_jsonl):
    path = write_jsonl("train.jsonl", [example("X firması")])
    result = load_examples(path)
    assert result.ok
    assert result.examples[0].id is None
    assert result.examples[0].metadata is None


def test_id_and_metadata_are_accepted_when_present(write_jsonl):
    row = example("X firması")
    row["id"] = "train-L1-0001-abcd"
    row["metadata"] = {"level": "L1", "split": "train", "tags": ["simple"]}
    path = write_jsonl("train.jsonl", [row])
    result = load_examples(path)
    assert result.ok
    assert result.examples[0].id == "train-L1-0001-abcd"
    assert result.examples[0].metadata.level == "L1"
    assert result.examples[0].metadata.tags == ["simple"]


def test_metadata_source_is_accepted(write_jsonl):
    row = example("X firması")
    row["metadata"] = {"split": "train", "source": "turspider"}
    path = write_jsonl("train.jsonl", [row])
    result = load_examples(path)
    assert result.ok
    assert result.examples[0].metadata.source == "turspider"


def test_metadata_source_defaults_to_none(write_jsonl):
    row = example("X firması")
    row["metadata"] = {"split": "train", "tags": []}
    path = write_jsonl("train.jsonl", [row])
    result = load_examples(path)
    assert result.ok
    assert result.examples[0].metadata.source is None


def test_unknown_top_level_key_still_rejected(write_jsonl):
    """extra="forbid" gercek yazim hatalarini yakalamaya devam eder."""
    row = example("X firması")
    row["souce"] = "typo"
    path = write_jsonl("train.jsonl", [row])
    result = load_examples(path)
    assert any(i.code == "SCHEMA_ERROR" for i in result.issues)


def test_distribution_report_includes_level_when_metadata_present(write_jsonl):
    row = example("X firması")
    row["metadata"] = {"level": "L1", "split": "train", "tags": []}
    path = write_jsonl("train.jsonl", [row])
    report = distribution_report(load_examples(path))
    assert report["level"] == {"L1": 1}


def test_distribution_report_omits_level_without_metadata(write_jsonl):
    path = write_jsonl("train.jsonl", [example("X firması")])
    report = distribution_report(load_examples(path))
    assert "level" not in report


def test_metadata_split_mismatch_detected(write_jsonl):
    row = example("X firması")
    row["metadata"] = {"level": "L1", "split": "validation", "tags": []}
    path = write_jsonl("train.jsonl", [row])
    result = load_examples(path)
    issues = find_metadata_split_mismatch(result, expected_split="train")
    assert [(i.code, i.line) for i in issues] == [("SPLIT_METADATA_MISMATCH", 1)]


def test_metadata_split_match_is_clean(write_jsonl):
    row = example("X firması")
    row["metadata"] = {"level": "L1", "split": "train", "tags": []}
    path = write_jsonl("train.jsonl", [row])
    result = load_examples(path)
    assert find_metadata_split_mismatch(result, expected_split="train") == []


def test_metadata_split_mismatch_skipped_when_absent(write_jsonl):
    path = write_jsonl("train.jsonl", [example("X firması")])
    result = load_examples(path)
    assert find_metadata_split_mismatch(result, expected_split="train") == []


def test_metadata_split_challenge_test_accepted_for_test_flag(write_jsonl):
    """V2: `--test challenge_test.jsonl` mesru bir kullanimdir."""
    row = example("X firması")
    row["metadata"] = {"split": "challenge_test"}
    path = write_jsonl("challenge_test.jsonl", [row])
    result = load_examples(path)
    assert find_metadata_split_mismatch(result, expected_split="test") == []


def test_metadata_split_challenge_test_still_rejected_for_train_flag(write_jsonl):
    """Ayni istisna `train`/`validation` icin GECERLI degildir."""
    row = example("X firması")
    row["metadata"] = {"split": "challenge_test"}
    path = write_jsonl("train.jsonl", [row])
    result = load_examples(path)
    issues = find_metadata_split_mismatch(result, expected_split="train")
    assert len(issues) == 1
