"""Prompt formatlama testleri (spec §14, §24).

Gercek model/tokenizer yuklenmez; chat template sahte bir nesneyle test edilir.
"""

import pytest

from app.schemas.query import to_canonical_json
from training.dataset import RawExample
from training.formatting import (
    PLAIN_OUTPUT_HEADER,
    ChatPromptFormatter,
    PlainPromptFormatter,
    PromptFormatter,
    available_formatters,
    get_formatter,
)

from .conftest import example

TEXT = "X firmasındaki Y ürününün satış sayısını getir."


@pytest.fixture
def sample() -> RawExample:
    return RawExample.model_validate(
        example(
            TEXT,
            filters=[
                {"field": "company", "operator": "eq", "value": "X"},
                {"field": "product", "operator": "eq", "value": "Y"},
            ],
        )
    )


class FakeTokenizer:
    """`apply_chat_template` sozlesmesini taklit eder."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        rendered = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        if add_generation_prompt:
            rendered += "<|assistant|>"
        return rendered


# --- Plain formatter ---


def test_plain_inference_prompt(sample):
    formatter = PlainPromptFormatter()
    assert formatter.format_inference(TEXT) == f"### Input\n{TEXT}\n\n### Output\n"


def test_plain_train_matches_spec_example(sample):
    formatter = PlainPromptFormatter()
    expected = (
        "### Input\n"
        "X firmasındaki Y ürününün satış sayısını getir.\n"
        "\n"
        "### Output\n"
        '{"operation":"count","target":"sales","filters":'
        '[{"field":"company","operator":"eq","value":"X"},'
        '{"field":"product","operator":"eq","value":"Y"}],'
        '"group_by":[],"order_by":null,"limit":null,'
        '"status":"valid","clarification":null}'
    )
    assert formatter.format_train(sample) == expected


def test_plain_response_template_appears_once_in_prompt(sample):
    formatter = PlainPromptFormatter()
    assert formatter.format_train(sample).count(PLAIN_OUTPUT_HEADER) == 1


# --- Kritik degismez: prefix esitligi ---


@pytest.mark.parametrize("eos", ["", "<|endoftext|>"])
def test_plain_prefix_invariant(sample, eos):
    """format_train == format_inference + canonical_json + eos (completion-only loss icin)."""
    formatter = PlainPromptFormatter(eos_token=eos)
    assert formatter.format_train(sample) == (
        formatter.format_inference(sample.input) + to_canonical_json(sample.output) + eos
    )


@pytest.mark.parametrize("eos", ["", "<|im_end|>"])
def test_chat_prefix_invariant(sample, eos):
    formatter = ChatPromptFormatter(tokenizer=FakeTokenizer(), eos_token=eos)
    assert formatter.format_train(sample) == (
        formatter.format_inference(sample.input) + to_canonical_json(sample.output) + eos
    )


# --- Chat formatter ---


def test_chat_uses_tokenizer_template(sample):
    formatter = ChatPromptFormatter(tokenizer=FakeTokenizer(), system_prompt=None)
    assert formatter.format_inference(TEXT) == f"<|user|>{TEXT}<|end|><|assistant|>"


def test_chat_includes_system_prompt():
    formatter = ChatPromptFormatter(tokenizer=FakeTokenizer(), system_prompt="SYS")
    assert formatter.format_inference(TEXT).startswith("<|system|>SYS<|end|>")


def test_chat_rejects_object_without_template():
    with pytest.raises(TypeError):
        ChatPromptFormatter(tokenizer=object())


# --- Registry ---


def test_registry_lists_both_formatters():
    assert available_formatters() == ["chat", "plain"]


def test_get_formatter_returns_plain():
    formatter = get_formatter("plain")
    assert isinstance(formatter, PlainPromptFormatter)
    assert isinstance(formatter, PromptFormatter)


def test_get_formatter_returns_chat():
    formatter = get_formatter("chat", tokenizer=FakeTokenizer())
    assert isinstance(formatter, ChatPromptFormatter)


def test_get_formatter_unknown_name():
    with pytest.raises(ValueError, match="bilinmeyen formatter"):
        get_formatter("chatml")
