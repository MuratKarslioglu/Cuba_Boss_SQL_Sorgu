"""benchmark.py CLI testleri (spec §23, §24).

Gercek model yuklenmez; `app.inference.model_loader` ve `app.inference.extractor`
mock'lanir. Saf yardimci fonksiyonlar (`sanitize_model_name`, `build_report`,
`peak_memory_mb`) ayrica torch olmadan test edilir.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="Faz 3+ testleri icin tam requirements.txt gerekir")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import benchmark  # noqa: E402

from app.errors import AppError  # noqa: E402
from app.inference.extractor import ExtractionOutcome, GenerationTiming  # noqa: E402
from app.inference.parsing import ExtractionResult  # noqa: E402
from app.schemas.query import QueryRequest  # noqa: E402
from training.evaluate import EvaluationRecord  # noqa: E402

from .conftest import example  # noqa: E402


def q(**overrides) -> QueryRequest:
    payload = {
        "operation": "count",
        "target": "sales",
        "filters": [],
        "group_by": [],
        "order_by": None,
        "limit": None,
    }
    payload.update(overrides)
    return QueryRequest.model_validate(payload)


# --- sanitize_model_name ---


def test_sanitize_model_name_replaces_slash():
    assert benchmark.sanitize_model_name("org/model-name") == "org__model-name"


def test_sanitize_model_name_replaces_spaces():
    assert benchmark.sanitize_model_name("my model") == "my_model"


def test_sanitize_model_name_strips_whitespace():
    assert benchmark.sanitize_model_name("  org/model  ") == "org__model"


# --- peak_memory_mb ---


def test_peak_memory_mb_returns_positive_number_or_none():
    value = benchmark.peak_memory_mb()
    assert value is None or value > 0


# --- build_report ---


def _fake_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_name="org/model",
        adapter_path=None,
        allow_retry=False,
        prompt_format="plain",
        test_file=Path("test.jsonl"),
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_report_shape():
    records = [
        EvaluationRecord(
            input="x",
            expected=q(),
            result=ExtractionResult.ok(q()),
            latency_ms=10.0,
            prompt_tokens=20,
            generated_tokens=5,
        )
    ]
    report = benchmark.build_report(
        _fake_args(),
        records,
        model_load_seconds=1.5,
        resolved_device="cpu",
        resolved_dtype="torch.float32",
    )

    assert report["model_name"] == "org/model"
    assert report["device"] == "cpu"
    assert report["dtype"] == "torch.float32"
    assert report["model_load_seconds"] == 1.5
    assert report["test_examples"] == 1
    assert report["exact_match_rate"] == 1.0
    assert report["latency"]["count"] == 1
    assert report["tokens"]["avg_prompt_tokens"] == 20.0
    assert "generated_at" in report


def test_build_report_reflects_retry_flag():
    records = [EvaluationRecord(input="x", expected=q(), result=ExtractionResult.ok(q()))]
    report = benchmark.build_report(
        _fake_args(allow_retry=True), records, 0.0, "cpu", "torch.float32"
    )
    assert report["retry_enabled"] is True


# --- CLI: dataset dogrulama basarisiz olursa ---


def test_main_fails_on_invalid_test_file(write_jsonl, tmp_path, capsys):
    bad_path = write_jsonl("test.jsonl", [example("x", operation="fetch")])
    exit_code = benchmark.main(
        [
            "--model_name",
            "org/model",
            "--test_file",
            str(bad_path),
            "--output_dir",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 1
    assert "dogrulamadan gecemedi" in capsys.readouterr().err


def test_main_fails_on_empty_limit(write_jsonl, tmp_path):
    good_path = write_jsonl("test.jsonl", [example("x")])
    exit_code = benchmark.main(
        [
            "--model_name",
            "org/model",
            "--test_file",
            str(good_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--limit",
            "0",
        ]
    )
    assert exit_code == 1


# --- CLI: basarili kosu (model/extract mock'lanir) ---


def _fake_loaded():
    loaded = MagicMock()
    loaded.tokenizer.chat_template = None
    loaded.device = "cpu"
    loaded.dtype = "torch.float32"
    return loaded


@patch("app.inference.extractor.extract")
@patch("app.inference.model_loader.load_base_model")
def test_main_writes_report_file(mock_load, mock_extract, write_jsonl, tmp_path):
    mock_load.return_value = _fake_loaded()
    mock_extract.return_value = ExtractionOutcome(
        result=ExtractionResult.ok(q()),
        timing=GenerationTiming(latency_ms=5.0, prompt_tokens=10, generated_tokens=3),
        retried=False,
    )

    test_path = write_jsonl(
        "test.jsonl", [example("X firması", filters=[]), example("Y firması", filters=[])]
    )
    output_dir = tmp_path / "benchmarks"

    exit_code = benchmark.main(
        [
            "--model_name",
            "org/tiny-model",
            "--test_file",
            str(test_path),
            "--output_dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    written = list(output_dir.glob("*.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["test_examples"] == 2
    assert payload["exact_match_rate"] == 1.0
    assert payload["tokens"]["avg_prompt_tokens"] == 10.0


@patch("app.inference.extractor.extract")
@patch("app.inference.model_loader.load_base_model")
def test_main_respects_limit(mock_load, mock_extract, write_jsonl, tmp_path):
    mock_load.return_value = _fake_loaded()
    mock_extract.return_value = ExtractionOutcome(
        result=ExtractionResult.ok(q()),
        timing=GenerationTiming(latency_ms=1.0, prompt_tokens=1, generated_tokens=1),
        retried=False,
    )
    test_path = write_jsonl(
        "test.jsonl", [example("A firması"), example("B firması"), example("C firması")]
    )
    benchmark.main(
        [
            "--model_name",
            "org/tiny-model",
            "--test_file",
            str(test_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--limit",
            "2",
        ]
    )
    assert mock_extract.call_count == 2


# --- dump_predictions ---


def test_dump_predictions_writes_input_expected_predicted(tmp_path):
    records = [
        EvaluationRecord(
            input="X firmasının satışları", expected=q(), result=ExtractionResult.ok(q())
        ),
        EvaluationRecord(
            input="bozuk cumle",
            expected=q(),
            result=ExtractionResult.fail(AppError("kirik")),
        ),
    ]
    out_path = tmp_path / "predictions.jsonl"
    benchmark.dump_predictions(records, out_path)

    lines = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["input"] == "X firmasının satışları"
    assert lines[0]["predicted"] == lines[0]["expected"]
    assert lines[0]["exact_match"] is True
    assert lines[1]["predicted"] is None
    assert lines[1]["error"] is not None


@patch("app.inference.extractor.extract")
@patch("app.inference.model_loader.load_base_model")
def test_main_writes_dump_predictions_when_requested(
    mock_load, mock_extract, write_jsonl, tmp_path
):
    mock_load.return_value = _fake_loaded()
    mock_extract.return_value = ExtractionOutcome(
        result=ExtractionResult.ok(q()),
        timing=GenerationTiming(latency_ms=1.0, prompt_tokens=1, generated_tokens=1),
        retried=False,
    )
    test_path = write_jsonl("test.jsonl", [example("X firması", filters=[])])
    dump_path = tmp_path / "preds.jsonl"

    exit_code = benchmark.main(
        [
            "--model_name",
            "org/tiny-model",
            "--test_file",
            str(test_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--dump_predictions",
            str(dump_path),
        ]
    )
    assert exit_code == 0
    assert dump_path.exists()
    line = json.loads(dump_path.read_text(encoding="utf-8").splitlines()[0])
    assert line["input"] == "X firması"


def test_cli_help_runs_as_subprocess_without_torch_import_error():
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "benchmark.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    assert "--test_file" in proc.stdout


# --- V2: rapor icinde rejection/confusion/group-by kirilimlari ---


def test_build_report_includes_v2_breakdowns():
    records = [
        EvaluationRecord(
            input="x",
            expected=q(),
            result=ExtractionResult.ok(q()),
            latency_ms=1.0,
            prompt_tokens=1,
            generated_tokens=1,
        )
    ]
    report = benchmark.build_report(_fake_args(), records, 0.0, "cpu", "torch.float32")
    assert "rejection" in report
    assert set(report["rejection"]) >= {"precision", "recall", "true_positive"}
    assert "status_confusion_matrix" in report
    assert report["status_confusion_matrix"] == {"valid": {"valid": 1}}
    assert "by_status" in report and "valid" in report["by_status"]
    assert report["by_family"] == {}  # metadata yok
    assert report["by_noise"] == {}


@patch("app.inference.extractor.extract")
@patch("app.inference.model_loader.load_base_model")
def test_main_report_includes_metadata_breakdown(mock_load, mock_extract, write_jsonl, tmp_path):
    mock_load.return_value = _fake_loaded()
    mock_extract.return_value = ExtractionOutcome(
        result=ExtractionResult.ok(q()),
        timing=GenerationTiming(latency_ms=1.0, prompt_tokens=1, generated_tokens=1),
        retried=False,
    )
    row = example("X firması", filters=[])
    row["metadata"] = {"split": "test", "family": "simple", "noise": "clean"}
    test_path = write_jsonl("test.jsonl", [row])

    exit_code = benchmark.main(
        [
            "--model_name",
            "org/tiny-model",
            "--test_file",
            str(test_path),
            "--output_dir",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 0
    written = list((tmp_path / "out").glob("*.json"))
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert "simple" in payload["by_family"]
    assert "clean" in payload["by_noise"]
