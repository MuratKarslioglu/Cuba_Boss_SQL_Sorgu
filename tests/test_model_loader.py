"""Model yukleyici testleri (spec §24, §30).

Cihaz/dtype cozumleme mantigi torch.cuda/torch.backends.mps sahteleyerek
(monkeypatch) test edilir -- gercek bir model indirmeye gerek yoktur.

Gercek bir model yukleme + LoRA uygulama testi, agi ve HuggingFace Hub
erisimini gerektirdigi icin opsiyoneldir ve varsayilan olarak atlanir
(spec §24: "Keep one optional integration test for actual model inference").
Calistirmak icin:

    RUN_MODEL_INTEGRATION_TESTS=1 pytest tests/test_model_loader.py
"""

import os

import pytest

torch = pytest.importorskip("torch", reason="Faz 3 testleri icin tam requirements.txt gerekir")

from app.inference.model_loader import (  # noqa: E402
    ModelLoadError,
    resolve_device,
    resolve_dtype,
)

TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


# --- resolve_device ---


def test_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_auto_falls_back_to_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == "mps"


def test_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_forced_cpu_always_allowed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cpu") == "cpu"


def test_forced_cuda_fails_clearly_when_unavailable(monkeypatch):
    """Spec §30: sessizce baska cihaza dusmek yerine acik hata."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ModelLoadError, match="CUDA"):
        resolve_device("cuda")


def test_forced_mps_fails_clearly_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ModelLoadError, match="MPS"):
        resolve_device("mps")


# --- resolve_dtype ---


@pytest.mark.parametrize(
    "preference,expected",
    [("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)],
)
def test_explicit_dtype_always_honored(preference, expected):
    assert resolve_dtype(preference, device="cuda") == expected
    assert resolve_dtype(preference, device="cpu") == expected


def test_auto_dtype_on_cpu_is_float32():
    assert resolve_dtype("auto", device="cpu") == torch.float32


def test_auto_dtype_on_mps_is_float16():
    """MPS'te bf16 destegi donanim/surume bagli oldugundan float16 guvenli varsayilandir."""
    assert resolve_dtype("auto", device="mps") == torch.float16


def test_auto_dtype_on_cuda_prefers_bf16_when_supported(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_dtype("auto", device="cuda") == torch.bfloat16


def test_auto_dtype_on_cuda_falls_back_to_fp16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert resolve_dtype("auto", device="cuda") == torch.float16


# --- adapter yukleme hatalari ---


def test_load_adapter_missing_path_raises(tmp_path):
    from app.inference.model_loader import load_adapter

    with pytest.raises(ModelLoadError, match="bulunamadi"):
        load_adapter(base_model=None, adapter_path=tmp_path / "yok")


def test_load_base_model_rejects_empty_name():
    from app.inference.model_loader import load_base_model

    with pytest.raises(ModelLoadError, match="bos olamaz"):
        load_base_model("")


# --- opsiyonel entegrasyon: gercek (kucuk) model indirme + LoRA ---


@pytest.mark.skipif(
    not os.environ.get("RUN_MODEL_INTEGRATION_TESTS"),
    reason="ag erisimi ve HuggingFace Hub gerektirir; RUN_MODEL_INTEGRATION_TESTS=1 ile acilir",
)
def test_load_base_model_and_apply_lora_end_to_end():
    from app.inference.model_loader import apply_lora, build_lora_config, load_base_model

    loaded = load_base_model(TINY_MODEL, device="cpu", dtype="float32")
    assert loaded.device == "cpu"
    assert loaded.tokenizer.pad_token is not None

    peft_model = apply_lora(
        loaded.model, build_lora_config(r=4, alpha=8, target_modules=["c_attn"])
    )
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    assert trainable > 0
