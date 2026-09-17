"""Uygulama hata tipleri (spec §32).

API yanitlari Python traceback'i sizdirmaz; traceback yalnizca gelistirici
loglarina gider. Bu enum, hem CLI hem de ileride FastAPI katmani icin
ortak hata sozlugudur.
"""

from __future__ import annotations

from enum import Enum


class ErrorType(str, Enum):
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"
    MODEL_INFERENCE_ERROR = "MODEL_INFERENCE_ERROR"
    JSON_PARSE_ERROR = "JSON_PARSE_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    DATASET_ERROR = "DATASET_ERROR"


class AppError(Exception):
    """Disariya guvenli bir mesaj tasiyan taban hata."""

    error_type: ErrorType = ErrorType.VALIDATION_ERROR

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def to_payload(self) -> dict[str, str]:
        return {"type": self.error_type.value, "message": self.message}


class DatasetError(AppError):
    error_type = ErrorType.DATASET_ERROR
