"""Cikarim orkestrasyonu testleri (spec §16, §17, §24).

`generate_raw`/model.generate() sahte (mock) bir model+tokenizer ile
test edilir; gercek bir LLM yuklenmez (spec §24: "Mock the model where
possible"). Gercek, kucuk bir modelle uctan uca calisan opsiyonel bir
entegrasyon testi dosyanin sonundadir.
"""

import os

import pytest

torch = pytest.importorskip("torch", reason="Faz 3+ testleri icin tam requirements.txt gerekir")

from app.inference.extractor import extract  # noqa: E402
from training.formatting import PlainPromptFormatter  # noqa: E402

VALID_JSON = (
    '{"operation":"count","target":"sales","filters":'
    '[{"field":"company","operator":"eq","value":"X"}],'
    '"group_by":[],"order_by":null,"limit":null}'
)


class FakeTokenizerOutput(dict):
    """`tokenizer(text, return_tensors="pt")`in dondurdugu BatchEncoding'i taklit eder."""

    def to(self, device):  # noqa: ARG002 - imza uyumu icin
        return self


class FakeTokenizer:
    """Prompt'u tek bir "token" (kendisi) sayan basit bir sahte tokenizer."""

    pad_token_id = 0

    def __call__(self, text, return_tensors="pt"):  # noqa: ARG002
        # input_ids uzunlugu = "prompt uzunlugu"; tensor yerine liste yeterli
        # (kodumuz yalnizca .shape[-1] ve tokenizer(...).to(device) kullaniyor).
        return FakeTokenizerOutput(input_ids=_FakeTensor(1))

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return ids.text


class _FakeTensor:
    """`.shape[-1]` ve indeksleme icin minimal bir sahte tensor."""

    def __init__(self, length, text: str = ""):
        self._length = length
        self.text = text

    @property
    def shape(self):
        return (1, self._length)

    def __getitem__(self, key):
        # output_ids[0][prompt_len:] -> "uretilen" metni tasiyan bir nesne don.
        return self


class FakeModel:
    """`generate()` cagrisinda sirayla verilen ciktilardan birini doner."""

    def __init__(self, outputs: list[str]):
        self._outputs = list(outputs)
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        text = self._outputs.pop(0)
        # output_ids[0][prompt_len:] -> _FakeTensor(taskes text)
        return [_FakeTensor(1, text=text)]


class FakeLoaded:
    def __init__(self, outputs: list[str]):
        self.model = FakeModel(outputs)
        self.tokenizer = FakeTokenizer()
        self.device = "cpu"


@pytest.fixture
def formatter():
    return PlainPromptFormatter()


# --- basarili ilk deneme ---


def test_extract_succeeds_on_first_try(formatter):
    loaded = FakeLoaded([VALID_JSON])
    outcome = extract("X firmasının satışları", loaded, formatter, max_new_tokens=32)
    assert outcome.result.success is True
    assert outcome.result.data.operation.value == "count"
    assert outcome.retried is False
    assert len(loaded.model.calls) == 1


def test_extract_uses_deterministic_generation_config(formatter):
    """Spec §15: do_sample=False, num_beams=1; temperature/top_p hic gecirilmez."""
    loaded = FakeLoaded([VALID_JSON])
    extract("X firmasının satışları", loaded, formatter, max_new_tokens=32)
    call_kwargs = loaded.model.calls[0]
    assert call_kwargs["do_sample"] is False
    assert call_kwargs["num_beams"] == 1
    assert "temperature" not in call_kwargs
    assert "top_p" not in call_kwargs


# --- basarisizlik + tek seferlik onarim (spec §17) ---


def test_extract_retries_once_on_failure_then_succeeds(formatter):
    loaded = FakeLoaded(["bozuk cikti, json yok", VALID_JSON])
    outcome = extract("X firmasının satışları", loaded, formatter, allow_retry=True)
    assert outcome.result.success is True
    assert outcome.retried is True
    assert len(loaded.model.calls) == 2


def test_extract_fails_after_retry_also_fails(formatter):
    loaded = FakeLoaded(["bozuk 1", "bozuk 2"])
    outcome = extract("X firmasının satışları", loaded, formatter, allow_retry=True)
    assert outcome.result.success is False
    assert outcome.result.error.type == "JSON_PARSE_ERROR"
    assert len(loaded.model.calls) == 2


def test_extract_does_not_retry_when_disabled(formatter):
    """Spec §17: sifir-retry benchmark kolu icin tekrar kapatilabilmelidir."""
    loaded = FakeLoaded(["bozuk cikti"])
    outcome = extract("X firmasının satışları", loaded, formatter, allow_retry=False)
    assert outcome.result.success is False
    assert outcome.retried is False
    assert len(loaded.model.calls) == 1


def test_extract_reports_validation_error_type(formatter):
    invalid_operation = VALID_JSON.replace('"count"', '"fetch"')
    loaded = FakeLoaded([invalid_operation, invalid_operation])
    outcome = extract("X", loaded, formatter, allow_retry=True)
    assert outcome.result.success is False
    assert outcome.result.error.type == "VALIDATION_ERROR"


def test_extract_timing_sums_across_retry_attempts(formatter):
    """Spec §23: tekrar denendiginde toplam maliyet (gecikme+token) raporlanmalidir."""
    loaded = FakeLoaded(["bozuk cikti, json yok", VALID_JSON])
    outcome = extract("X firmasının satışları", loaded, formatter, allow_retry=True)
    # Her sahte cagri 1 "token" uretiyor; iki cagri toplamda 2 olmalidir.
    assert outcome.timing.generated_tokens == 2


def test_extract_timing_is_single_attempt_without_retry(formatter):
    loaded = FakeLoaded([VALID_JSON])
    outcome = extract("X firmasının satışları", loaded, formatter, allow_retry=True)
    assert outcome.timing.generated_tokens == 1


# --- girdi dogrulama ---


def test_extract_rejects_empty_input(formatter):
    loaded = FakeLoaded([])
    outcome = extract("   ", loaded, formatter)
    assert outcome.result.success is False
    assert outcome.result.error.type == "INVALID_INPUT"
    assert len(loaded.model.calls) == 0  # model hic cagrilmamali


# --- opsiyonel entegrasyon: gercek (kucuk) modelle uctan uca ---


@pytest.mark.skipif(
    not os.environ.get("RUN_MODEL_INTEGRATION_TESTS"),
    reason="ag erisimi ve HuggingFace Hub gerektirir; RUN_MODEL_INTEGRATION_TESTS=1 ile acilir",
)
def test_real_tiny_model_produces_controlled_failure():
    """Egitilmemis (random) bir model gecerli JSON uretemez; pipeline bunu
    sessizce kabul etmek yerine kontrollu bir hata olarak raporlamalidir."""
    from app.inference.model_loader import load_base_model

    loaded = load_base_model("hf-internal-testing/tiny-random-gpt2", device="cpu", dtype="float32")
    outcome = extract(
        "X firmasındaki satış sayısını getir.",
        loaded,
        PlainPromptFormatter(),
        max_new_tokens=16,
        allow_retry=True,
    )
    assert outcome.result.success is False
    assert outcome.result.error.type in {"JSON_PARSE_ERROR", "VALIDATION_ERROR"}
    assert outcome.timing.latency_ms > 0
