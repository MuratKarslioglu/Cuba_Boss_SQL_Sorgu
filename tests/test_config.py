"""Konfigurasyon testleri (spec §25).

Ayrica onemli bir mimari degismez burada kilitlenir: `app.config` torch
import etmez, boylece sema/dataset katmani agir ML bagimliliklari olmadan
calisir ve test edilir.
"""

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Gercek .env dosyasinin testleri etkilemesini onler."""
    for name in (
        "BASE_MODEL",
        "ADAPTER_PATH",
        "MAX_NEW_TOKENS",
        "DEVICE",
        "MODEL_DTYPE",
        "HOST",
        "PORT",
        "LOG_PROMPTS",
        "LOG_LEVEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def test_defaults_match_env_example():
    settings = get_settings()
    assert settings.adapter_path == "./models/latest"
    assert settings.max_new_tokens == 192
    assert settings.device == "auto"
    assert settings.model_dtype == "auto"
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000


def test_prompt_logging_is_off_by_default():
    """Spec §26: prompt icerigi varsayilan olarak loglanmaz."""
    assert get_settings().log_prompts is False


def test_base_model_is_not_hardcoded():
    """Spec §13: model kaynak kodda sabitlenmez."""
    assert get_settings().base_model == ""


def test_reads_environment(monkeypatch):
    monkeypatch.setenv("BASE_MODEL", "org/model-0.5b")
    monkeypatch.setenv("DEVICE", "mps")
    monkeypatch.setenv("MAX_NEW_TOKENS", "256")
    monkeypatch.setenv("LOG_PROMPTS", "true")
    settings = get_settings()
    assert settings.base_model == "org/model-0.5b"
    assert settings.device == "mps"
    assert settings.max_new_tokens == 256
    assert settings.log_prompts is True


def test_reads_dotenv_file(tmp_path):
    (tmp_path / ".env").write_text("BASE_MODEL=from-dotenv\nPORT=9000\n", encoding="utf-8")
    settings = Settings()
    assert settings.base_model == "from-dotenv"
    assert settings.port == 9000


@pytest.mark.parametrize("bad", ["gpu", "cuda:0", "metal", ""])
def test_invalid_device_rejected(monkeypatch, bad):
    monkeypatch.setenv("DEVICE", bad)
    with pytest.raises(ValidationError):
        get_settings()


@pytest.mark.parametrize("bad", ["int8", "fp16", ""])
def test_invalid_dtype_rejected(monkeypatch, bad):
    monkeypatch.setenv("MODEL_DTYPE", bad)
    with pytest.raises(ValidationError):
        get_settings()


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_non_positive_max_new_tokens_rejected(monkeypatch, bad):
    monkeypatch.setenv("MAX_NEW_TOKENS", bad)
    with pytest.raises(ValidationError):
        get_settings()


@pytest.mark.parametrize("bad", ["0", "70000"])
def test_invalid_port_rejected(monkeypatch, bad):
    monkeypatch.setenv("PORT", bad)
    with pytest.raises(ValidationError):
        get_settings()


def test_unknown_env_vars_ignored(monkeypatch):
    monkeypatch.setenv("SOME_UNRELATED_VAR", "x")
    assert get_settings().device == "auto"


def test_config_does_not_import_torch():
    """Faz 1+2 torch kurulmadan calisabilmelidir."""
    code = "import app.config, sys; print('torch' in sys.modules)"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    assert proc.stdout.strip() == "False"


def test_schema_layer_does_not_import_torch():
    code = (
        "import app.schemas, training.dataset, training.formatting, sys; "
        "print('torch' in sys.modules)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    assert proc.stdout.strip() == "False"
