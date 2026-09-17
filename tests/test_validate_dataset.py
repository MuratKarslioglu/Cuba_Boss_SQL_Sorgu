"""validate_dataset.py CLI testleri (spec §21).

Script gercek bir alt surecte calistirilir; cikis kodu sozlesmesi burada
kilitlenir (hata varsa 0 disi kod).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import example

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "validate_dataset.py"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_clean_dataset_exits_zero(write_jsonl):
    path = write_jsonl(
        "train.jsonl",
        [example("X firmasının satışları"), example("Y firmasının satışları")],
    )
    proc = run("--train", str(path))
    assert proc.returncode == 0, proc.stdout
    assert "BASARILI" in proc.stdout


def test_broken_json_exits_nonzero(write_jsonl):
    path = write_jsonl("train.jsonl", [example("ilk"), "{bozuk"])
    proc = run("--train", str(path))
    assert proc.returncode == 1
    assert "INVALID_JSON" in proc.stdout


def test_schema_error_exits_nonzero(write_jsonl):
    path = write_jsonl("train.jsonl", [example("x", operation="fetch")])
    proc = run("--train", str(path))
    assert proc.returncode == 1
    assert "SCHEMA_ERROR" in proc.stdout


def test_duplicate_exits_nonzero(write_jsonl):
    path = write_jsonl("train.jsonl", [example("aynı"), example("aynı")])
    proc = run("--train", str(path))
    assert proc.returncode == 1
    assert "DUPLICATE_INPUT" in proc.stdout


def test_split_leakage_exits_nonzero(write_jsonl):
    train = write_jsonl("train.jsonl", [example("X firmasının satışları")])
    test = write_jsonl("test.jsonl", [example("X firmasının satışları")])
    proc = run("--train", str(train), "--test", str(test))
    assert proc.returncode == 1
    assert "SPLIT_LEAKAGE" in proc.stdout


def test_metadata_split_mismatch_exits_nonzero(write_jsonl):
    row = example("X firmasının satışları")
    row["metadata"] = {"level": "L1", "split": "test", "tags": []}
    path = write_jsonl("train.jsonl", [row])
    proc = run("--train", str(path))
    assert proc.returncode == 1
    assert "SPLIT_METADATA_MISMATCH" in proc.stdout


def test_unknown_field_is_warning_by_default(write_jsonl):
    row = example("x", filters=[{"field": "tedarikci", "operator": "eq", "value": "A"}])
    path = write_jsonl("train.jsonl", [row])
    proc = run("--train", str(path))
    assert proc.returncode == 0
    assert "UNKNOWN_FIELD_WARNING" in proc.stdout


def test_unknown_field_is_error_in_strict_mode(write_jsonl):
    row = example("x", filters=[{"field": "tedarikci", "operator": "eq", "value": "A"}])
    path = write_jsonl("train.jsonl", [row])
    proc = run("--train", str(path), "--strict-vocabulary")
    assert proc.returncode == 1
    assert "UNKNOWN_FIELD" in proc.stdout


def test_unknown_target_reported(write_jsonl):
    path = write_jsonl("train.jsonl", [example("x", target="fatura")])
    proc = run("--train", str(path), "--strict-vocabulary")
    assert proc.returncode == 1
    assert "UNKNOWN_TARGET" in proc.stdout


def test_json_output_is_machine_readable(write_jsonl):
    path = write_jsonl("train.jsonl", [example("X firması")])
    proc = run("--train", str(path), "--json")
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["total_examples"] == 1
    assert payload["splits"]["train"]["distribution"]["operation"] == {"count": 1}


def test_json_output_lists_errors(write_jsonl):
    path = write_jsonl("train.jsonl", [example("x", limit=0)])
    proc = run("--train", str(path), "--json")
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert payload["errors"][0]["code"] == "SCHEMA_ERROR"
    assert payload["errors"][0]["line"] == 1


def test_missing_split_argument_errors():
    proc = run()
    assert proc.returncode == 2
    assert "en az bir split" in proc.stderr


def test_help_works():
    proc = run("--help")
    assert proc.returncode == 0
    assert "--strict-vocabulary" in proc.stdout


@pytest.mark.parametrize("flag", ["--train", "--validation", "--test"])
def test_each_split_flag_accepted(write_jsonl, flag):
    path = write_jsonl("data.jsonl", [example("X firması")])
    assert run(flag, str(path)).returncode == 0
