"""Egitim/cikarim prompt formatlama soyutlamasi (spec §14).

Kritik degismez:

    format_train(ex) == format_inference(ex.input) + canonical_json(ex.output) + eos

Bu esitlik, completion-only loss masking'in (yalnizca cevap tokenlari
uzerinden loss) dogru calismasi icin gereklidir ve testle kilitlenmistir.

`transformers` modul seviyesinde import EDILMEZ; chat formatter tokenizer'i
duck-typed olarak alir. Bu sayede Faz 1+2 torch'suz calisir.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.schemas.query import QueryRequest, to_canonical_json
from training.dataset import RawExample

# Spec §14: prompt cok kucuk tutulur, buyuk sistem promptundan kacinilir.
PLAIN_INPUT_HEADER = "### Input\n"
PLAIN_OUTPUT_HEADER = "### Output\n"

DEFAULT_SYSTEM_PROMPT = "Turkce sorguyu JSON'a cevir."


@runtime_checkable
class PromptFormatter(Protocol):
    """Egitim ve cikarim prompt'unu ureten arayuz."""

    name: str
    # TRL'in DataCollatorForCompletionOnlyLM'i icin cevap baslangic isareti.
    response_template: str

    def format_inference(self, text: str) -> str:
        """Modelin devamini uretecegi onek."""

    def format_train(self, example: RawExample) -> str:
        """Onek + hedef kanonik JSON."""


class _BaseFormatter:
    name = "base"
    response_template = ""

    def __init__(self, eos_token: str = "") -> None:
        self.eos_token = eos_token

    def format_inference(self, text: str) -> str:  # pragma: no cover - alt siniflar uygular
        raise NotImplementedError

    def format_target(self, query: QueryRequest) -> str:
        return to_canonical_json(query)

    def format_train(self, example: RawExample) -> str:
        prefix = self.format_inference(example.input)
        return prefix + self.format_target(example.output) + self.eos_token


class PlainPromptFormatter(_BaseFormatter):
    """Spec §14'teki duz metin formati.

    ### Input
    <turkce metin>

    ### Output
    <kanonik json>
    """

    name = "plain"
    response_template = PLAIN_OUTPUT_HEADER

    def format_inference(self, text: str) -> str:
        return f"{PLAIN_INPUT_HEADER}{text}\n\n{PLAIN_OUTPUT_HEADER}"


class ChatPromptFormatter(_BaseFormatter):
    """Tokenizer'in kendi chat template'ini kullanir (spec §14).

    ChatML elle taklit EDILMEZ. `tokenizer` yalnizca `apply_chat_template`
    metoduna sahip olmak zorundadir.
    """

    name = "chat"

    def __init__(
        self,
        tokenizer: object,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        eos_token: str = "",
    ) -> None:
        super().__init__(eos_token=eos_token)
        if not hasattr(tokenizer, "apply_chat_template"):
            raise TypeError("tokenizer 'apply_chat_template' metoduna sahip olmalidir")
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.response_template = ""

    def format_inference(self, text: str) -> str:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": text})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


_REGISTRY = {
    PlainPromptFormatter.name: PlainPromptFormatter,
    ChatPromptFormatter.name: ChatPromptFormatter,
}


def available_formatters() -> list[str]:
    return sorted(_REGISTRY)


def get_formatter(name: str, **kwargs: object) -> PromptFormatter:
    """Isimle formatter uretir (spec §14: format kolayca degisebilmeli)."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"bilinmeyen formatter: {name!r}; secenekler: {available_formatters()}"
        ) from None
    return factory(**kwargs)  # type: ignore[arg-type]
