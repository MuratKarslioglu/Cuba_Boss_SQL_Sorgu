"""Model ciktisindan JSON izole etme ve dogrulama (spec §16).

Bu modul torch'tan BAGIMSIZDIR: `extractor.py`'nin uretim (generation)
kismindan ayri tutulur, boylece model yuklemeden saf metin -> JSON ->
Pydantic hattini test etmek mumkun olur. Ayni sebeple Faz 5'in
degerlendirme kodu da bu modulu torch olmadan yeniden kullanabilir.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, ValidationError

from app.errors import AppError, ErrorType
from app.schemas.query import QueryRequest


class JsonIsolationError(AppError):
    """Metinde dengeli bir JSON nesnesi bulunamadi."""

    error_type = ErrorType.JSON_PARSE_ERROR


class JsonParseError(AppError):
    """Izole edilen metin gecerli JSON degil."""

    error_type = ErrorType.JSON_PARSE_ERROR


class QueryValidationError(AppError):
    """JSON gecerli ama QueryRequest semasina uymuyor."""

    error_type = ErrorType.VALIDATION_ERROR


def isolate_json(text: str) -> str:
    """Metindeki ilk dengeli JSON nesnesini bulur (spec §16: 'extra surrounding text').

    Naif `text[text.find("{"):text.rfind("}")+1]` yaklasimi ic ice nesnelerde
    (filters, order_by) ve string degerlerin icindeki suslu parantezlerde
    yanlis sonuc verir. Bunun yerine string durumunu ve parantez derinligini
    takip eden bir tarayici kullanilir.
    """
    start = text.find("{")
    if start == -1:
        raise JsonIsolationError("model ciktisinda JSON nesnesi bulunamadi")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise JsonIsolationError("JSON nesnesi kapanmadi (dengesiz suslu parantez)")


def parse_query_json(raw_text: str) -> QueryRequest:
    """Ham model ciktisindan QueryRequest'e: izole et -> json.loads -> Pydantic.

    Spec §16: hicbir asamada gecersiz cikti sessizce kabul edilmez; her
    basarisizlik turu ayri, acik bir hata olarak yukari tasinir.

    Not: `isolate_json` her zaman `{` ile baslayan, dengeli bir dilim
    dondurdugu icin `json.loads` sonucu her zaman bir dict'tir; ayrica bir
    "nesne mi" kontrolune gerek yoktur.
    """
    isolated = isolate_json(raw_text)

    try:
        payload = json.loads(isolated)
    except json.JSONDecodeError as exc:
        raise JsonParseError(f"gecersiz JSON: {exc.msg}") from exc

    try:
        return QueryRequest.model_validate(payload)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()
        )
        raise QueryValidationError(f"QueryRequest semasina uymuyor: {details}") from exc


class ExtractionErrorInfo(BaseModel):
    """spec §16'daki hata payload'u: {"type", "message"}."""

    model_config = ConfigDict(extra="forbid")

    type: str
    message: str


class ExtractionResult(BaseModel):
    """spec §16'daki sonuc zarfi: {"success", "data", "error"}."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    data: QueryRequest | None = None
    error: ExtractionErrorInfo | None = None

    @classmethod
    def ok(cls, data: QueryRequest) -> ExtractionResult:
        return cls(success=True, data=data, error=None)

    @classmethod
    def fail(cls, error: AppError) -> ExtractionResult:
        return cls(success=False, data=None, error=ExtractionErrorInfo(**error.to_payload()))
