"""Deterministik cikarim: model uretimi + JSON izolasyon/dogrulama (spec §15-17).

Uretim ayarlari spec §15'in gerektirdigi gibi tamamen deterministiktir:
`do_sample=False`, tek beam (`num_beams=1`). `temperature`/`top_p` hic
GECIRILMEZ (None gecirmek yerine) -- bu parametreler yalnizca ornekleme
(sampling) acikken anlamlidir ve do_sample=False iken transformers'in
gereksiz uyari basmasini onler; net sonuc spec'in istedigi ile aynidir.

JSON izolasyon/dogrulama mantigi `app.inference.parsing`'e ayrilmistir
(torch'tan bagimsiz, ayri test edilebilir).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.errors import AppError
from app.inference.parsing import ExtractionResult, parse_query_json

if TYPE_CHECKING:
    from app.inference.model_loader import LoadedModel
    from training.formatting import PromptFormatter

logger = logging.getLogger(__name__)


@dataclass
class GenerationTiming:
    """Benchmark scripti (Faz 5) icin ham zamanlama/token bilgisi."""

    latency_ms: float
    prompt_tokens: int
    generated_tokens: int


def _generate_from_prompt(
    loaded: LoadedModel, prompt: str, max_new_tokens: int
) -> tuple[str, GenerationTiming]:
    """Verilen tam prompt'tan ham tamamlama metnini uretir."""
    inputs = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.device)
    prompt_len = inputs["input_ids"].shape[-1]

    start = time.perf_counter()
    output_ids = loaded.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=loaded.tokenizer.pad_token_id,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    generated_ids = output_ids[0][prompt_len:]
    raw_text = loaded.tokenizer.decode(generated_ids, skip_special_tokens=True)

    timing = GenerationTiming(
        latency_ms=latency_ms,
        prompt_tokens=prompt_len,
        generated_tokens=int(generated_ids.shape[-1]),
    )
    return raw_text, timing


def generate_raw(
    loaded: LoadedModel,
    formatter: PromptFormatter,
    text: str,
    max_new_tokens: int,
) -> tuple[str, GenerationTiming]:
    """Modelden ham (JSON izole edilmemis) tamamlama metnini uretir."""
    return _generate_from_prompt(loaded, formatter.format_inference(text), max_new_tokens)


def _build_repair_prompt(original_prompt: str, bad_output: str) -> str:
    """Spec §17: 'kucuk' bir onarim istemi -- buyuk bir onarim promptuna
    guvenilmez. Modelin kendi hatali ciktisini gorup duzeltmesi istenir."""
    return (
        f"{original_prompt}{bad_output}\n\n"
        "Yukaridaki cikti gecerli degildi. Sadece tek, gecerli bir JSON "
        "nesnesi don.\n"
    )


@dataclass
class ExtractionOutcome:
    """`extract()`in tam sonucu: dogrulanmis veri + benchmark icin zamanlama.

    `timing`, tekrar denendiyse (spec §17) her iki uretim cagrisinin
    TOPLAM gecikmesini ve TOPLAM uretilen token sayisini tasir (gercek
    maliyeti yansitmasi icin); `prompt_tokens` ise SON denemenin degeridir
    (onarim istemi orijinalden uzun oldugundan en anlamli deger budur).
    """

    result: ExtractionResult
    timing: GenerationTiming
    retried: bool


def extract(
    text: str,
    loaded: LoadedModel,
    formatter: PromptFormatter,
    max_new_tokens: int = 192,
    allow_retry: bool = True,
) -> ExtractionOutcome:
    """Spec §16'nin tam akisi + spec §17'nin en fazla bir tekrar politikasi.

    Basarisiz cikti asla sessizce onarilmaz (spec §37): ilk deneme
    basarisiz olursa en fazla bir kez, kucuk bir onarim istemiyle tekrar
    denenir; o da basarisiz olursa acik bir hata donulur.
    """
    if not text or not text.strip():
        from app.errors import ErrorType

        empty_result = ExtractionResult(
            success=False,
            error={"type": ErrorType.INVALID_INPUT.value, "message": "girdi metni bos olamaz"},
        )
        return ExtractionOutcome(
            result=empty_result,
            timing=GenerationTiming(latency_ms=0.0, prompt_tokens=0, generated_tokens=0),
            retried=False,
        )

    raw_text, timing = generate_raw(loaded, formatter, text, max_new_tokens)

    try:
        result = ExtractionResult.ok(parse_query_json(raw_text))
        return ExtractionOutcome(result=result, timing=timing, retried=False)
    except AppError as first_error:
        if not allow_retry:
            return ExtractionOutcome(
                result=ExtractionResult.fail(first_error), timing=timing, retried=False
            )

        logger.info(
            "ilk deneme basarisiz [%s], tek seferlik onarim deneniyor",
            first_error.error_type.value,
        )
        repair_prompt = _build_repair_prompt(formatter.format_inference(text), raw_text)
        retry_text, retry_timing = _generate_from_prompt(loaded, repair_prompt, max_new_tokens)
        total_timing = GenerationTiming(
            latency_ms=timing.latency_ms + retry_timing.latency_ms,
            prompt_tokens=retry_timing.prompt_tokens,
            generated_tokens=timing.generated_tokens + retry_timing.generated_tokens,
        )

        try:
            result = ExtractionResult.ok(parse_query_json(retry_text))
            return ExtractionOutcome(result=result, timing=total_timing, retried=True)
        except AppError as second_error:
            return ExtractionOutcome(
                result=ExtractionResult.fail(second_error), timing=total_timing, retried=True
            )
