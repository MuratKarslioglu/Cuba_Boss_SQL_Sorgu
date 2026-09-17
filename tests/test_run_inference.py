"""run_inference.py CLI testleri (spec §20, §24).

Gercek model yuklenmez; `app.inference.model_loader` ve `app.inference.extractor`
mock'lanir. Bu, CLI'nin arguman/formatlama/exit-code sozlesmesini gercek bir
LLM olmadan dogrulamak icindir.
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="Faz 3+ testleri icin tam requirements.txt gerekir")

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_inference  # noqa: E402

from app.inference.extractor import ExtractionOutcome, GenerationTiming  # noqa: E402
from app.inference.parsing import ExtractionResult, QueryValidationError  # noqa: E402
from app.schemas.query import QueryRequest  # noqa: E402

_ZERO_TIMING = GenerationTiming(latency_ms=1.0, prompt_tokens=1, generated_tokens=1)


def _fake_loaded():
    loaded = MagicMock()
    loaded.tokenizer.chat_template = None
    loaded.model = MagicMock()
    return loaded


def test_help_does_not_require_model_name(capsys):
    with pytest.raises(SystemExit) as exc:
        run_inference.build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    assert "--text" in capsys.readouterr().out


def test_missing_model_name_errors_cleanly(monkeypatch, capsys):
    monkeypatch.delenv("BASE_MODEL", raising=False)
    exit_code = run_inference.main(["--text", "X firmasının satışları"])
    assert exit_code == 1
    assert "model adi verilmedi" in capsys.readouterr().err


@patch("app.inference.extractor.extract")
@patch("app.inference.model_loader.load_base_model")
def test_successful_extraction_prints_canonical_json(mock_load, mock_extract, capsys):
    mock_load.return_value = _fake_loaded()
    query = QueryRequest.model_validate(
        {
            "operation": "count",
            "target": "sales",
            "filters": [],
            "group_by": [],
            "order_by": None,
            "limit": None,
        }
    )
    mock_extract.return_value = ExtractionOutcome(
        result=ExtractionResult.ok(query), timing=_ZERO_TIMING, retried=False
    )

    exit_code = run_inference.main(
        [
            "--text",
            "X firmasının satışları",
            "--model_name",
            "org/model",
            "--adapter_path",
            "/does/not/exist",
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert '"operation":"count"' in out
    assert "Valid: true" in out
    assert "Latency:" in out


@patch("app.inference.extractor.extract")
@patch("app.inference.model_loader.load_base_model")
def test_failed_extraction_exits_nonzero(mock_load, mock_extract, capsys):
    mock_load.return_value = _fake_loaded()
    mock_extract.return_value = ExtractionOutcome(
        result=ExtractionResult.fail(QueryValidationError("semaya uymuyor")),
        timing=_ZERO_TIMING,
        retried=False,
    )

    exit_code = run_inference.main(
        ["--text", "X", "--model_name", "org/model", "--adapter_path", "/does/not/exist"]
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "Valid: false" in out
    assert "VALIDATION_ERROR" in out


def test_cli_help_runs_as_subprocess_without_torch_import_error():
    """CLI'nin gercek surecte de calistigini dogrular (import hatasi yok)."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_inference.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0
    assert "--text" in proc.stdout
