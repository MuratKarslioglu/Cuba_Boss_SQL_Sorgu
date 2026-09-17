"""Model/tokenizer yukleme ve cihaz secimi (spec §13, §30).

`app.config` bilerek torch'tan bagimsiz tutulmustur (Faz 1+2'nin agir ML
bagimliliklari olmadan calisabilmesi icin); torch'a bagimli tum mantik
buradadir.

Model kaynak kodda sabitlenmez (spec §13): `load_base_model` her zaman bir
`model_name` parametresi alir, cagiran taraf bunu `Settings.base_model`
uzerinden gecirir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import DevicePreference, DtypePreference
from app.errors import AppError, ErrorType

if TYPE_CHECKING:
    from peft import PeftModel as _PeftModel
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class ModelLoadError(AppError):
    error_type = ErrorType.MODEL_LOAD_ERROR


def resolve_device(preference: DevicePreference = "auto") -> str:
    """Cihaz secimi (spec §30).

    `preference` "auto" degilse zorlanir; istenen cihaz mevcut degilse
    sessizce baska bir cihaza dusmek yerine acik bir hata verilir
    (spec §30: "fail with a clear error").
    """
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise ModelLoadError("DEVICE=cuda istendi ama CUDA kullanilamiyor")
        return "cuda"
    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise ModelLoadError("DEVICE=mps istendi ama MPS (Apple Silicon) kullanilamiyor")
        return "mps"
    if preference == "cpu":
        return "cpu"

    # auto: spec §30'daki zincirin birebir uygulanmasi
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(preference: DtypePreference, device: str) -> torch.dtype:
    """Veri tipi secimi.

    `auto`: CPU'da float32 (kararlilik), CUDA'da bf16 destekleniyorsa bf16,
    aksi halde float16. MPS'te float16 kullanilir; bf16 destegi PyTorch/
    donanim surumune gore degisir ve daha az test edilmistir.
    """
    if preference == "float32":
        return torch.float32
    if preference == "float16":
        return torch.float16
    if preference == "bfloat16":
        return torch.bfloat16

    if device == "cpu":
        return torch.float32
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: str
    dtype: torch.dtype
    model_name: str


def load_base_model(
    model_name: str,
    device: DevicePreference = "auto",
    dtype: DtypePreference = "auto",
) -> LoadedModel:
    """Temel modeli ve tokenizer'i yukler (LoRA uygulanmamis haliyle)."""
    if not model_name:
        raise ModelLoadError("model adi bos olamaz (BASE_MODEL ayarlanmamis olabilir)")

    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)

    logger.info(
        "model yukleniyor: %s (device=%s, dtype=%s)", model_name, resolved_device, resolved_dtype
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=resolved_dtype)
    except Exception as exc:
        raise ModelLoadError(f"model yuklenemedi: {model_name!r}: {exc}") from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(resolved_device)

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
        dtype=resolved_dtype,
        model_name=model_name,
    )


def build_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: str | list[str] = "all-linear",
) -> LoraConfig:
    """Spec §29'daki ornek varsayilanlar; hicbiri optimal kabul edilmez,
    cagiran taraf CLI'dan degistirir."""
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )


def apply_lora(model: PreTrainedModel, config: LoraConfig) -> _PeftModel:
    return get_peft_model(model, config)


def load_adapter(base_model: PreTrainedModel, adapter_path: str | Path) -> _PeftModel:
    """Egitilmis bir LoRA adapter'ini temel modele uygular (cikarim icin)."""
    path = Path(adapter_path)
    if not path.exists():
        raise ModelLoadError(f"adapter yolu bulunamadi: {path}")
    try:
        return PeftModel.from_pretrained(base_model, str(path))
    except Exception as exc:
        raise ModelLoadError(f"adapter yuklenemedi: {path}: {exc}") from exc
