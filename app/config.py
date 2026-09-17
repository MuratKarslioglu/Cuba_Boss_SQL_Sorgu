"""Merkezi konfigurasyon (spec §25).

Onemli: bu modul torch import ETMEZ. `DEVICE` burada yalnizca bir tercih
string'i olarak tutulur; gercek cihaz cozumlemesi (spec §30'daki
cuda -> mps -> cpu zinciri) Faz 3'te model yukleyiciye aittir.
Bu sayede sema ve dataset katmanlari torch kurulmadan calisir ve test edilir.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DevicePreference = Literal["auto", "cuda", "mps", "cpu"]
DtypePreference = Literal["auto", "float32", "float16", "bfloat16"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_` onekli alan adlariyla cakismayi onler.
        protected_namespaces=(),
    )

    # Spec §13: model kaynak kodda sabitlenmez.
    base_model: str = Field(default="", alias="BASE_MODEL")
    adapter_path: str = Field(default="./models/latest", alias="ADAPTER_PATH")

    # Spec §15: dusuk tutulur, olculerek ayarlanir.
    max_new_tokens: int = Field(default=192, alias="MAX_NEW_TOKENS", gt=0)

    device: DevicePreference = Field(default="auto", alias="DEVICE")
    model_dtype: DtypePreference = Field(default="auto", alias="MODEL_DTYPE")

    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=8000, alias="PORT", gt=0, lt=65536)

    # Spec §26: prompt icerigi varsayilan olarak loglanmaz.
    log_prompts: bool = Field(default=False, alias="LOG_PROMPTS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


def get_settings() -> Settings:
    """Ayarlari okur. Test edilebilir kalmasi icin cache'lenmez."""
    return Settings()
