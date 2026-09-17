"""Egitim scripti testleri (spec §24, §29).

Gercek bir model/torch egitimi burada CALISTIRILMAZ; yalnizca CLI ayristirma
ve saf donusum fonksiyonlari test edilir. Uctan uca gercek bir egitim
adiminin calistigi ayrica manuel olarak dogrulanmistir (bkz. proje notlari);
bunu her `pytest` calistirmasinda tekrarlamak agi ve dakikalar suren bir
modelo indirmeyi gerektirir.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from training.dataset import load_examples
from training.formatting import get_formatter
from training.train import (
    TrainingDataError,
    _reject_challenge_test,
    parse_args,
    resolve_prompt_format,
    to_prompt_completion_pairs,
)

from .conftest import example

# --- parse_args ---


def test_parse_args_requires_core_arguments():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_applies_spec_defaults():
    args = parse_args(
        [
            "--model_name",
            "org/tiny-model",
            "--train_file",
            "train.jsonl",
            "--validation_file",
            "validation.jsonl",
            "--output_dir",
            "models/run_001",
        ]
    )
    # Spec §29'daki ornek varsayilanlar.
    assert args.epochs == 3
    assert args.learning_rate == 2e-4
    assert args.batch_size == 4
    assert args.gradient_accumulation_steps == 4
    assert args.lora_r == 16
    assert args.lora_alpha == 32
    assert args.lora_dropout == 0.05
    assert args.seed == 42
    assert args.device == "auto"
    assert args.prompt_format == "auto"


def test_parse_args_overrides_hyperparameters():
    args = parse_args(
        [
            "--model_name",
            "org/tiny-model",
            "--train_file",
            "train.jsonl",
            "--validation_file",
            "validation.jsonl",
            "--output_dir",
            "models/run_001",
            "--epochs",
            "5",
            "--learning_rate",
            "1e-3",
            "--lora_r",
            "8",
        ]
    )
    assert args.epochs == 5
    assert args.learning_rate == 1e-3
    assert args.lora_r == 8


def test_parse_args_rejects_unknown_device():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model_name",
                "m",
                "--train_file",
                "t",
                "--validation_file",
                "v",
                "--output_dir",
                "o",
                "--device",
                "tpu",
            ]
        )


def test_importing_train_module_does_not_load_trl():
    """`--help` gibi hafif yollarin trl/torch'u zorunlu kilmamasi gerekir.

    trl/torch import'lari `run_training` icine gec birakilmistir; bu test
    modul import edildiginde bunlarin tetiklenmedigini ayri bir surecte
    dogrular (bu surecte torch zaten kurulu olsa bile).
    """
    code = "import training.train, sys; print('trl' in sys.modules)"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert proc.stdout.strip() == "False"


# --- resolve_prompt_format ---


class _WithChatTemplate:
    chat_template = "{{ messages }}"


class _WithoutChatTemplate:
    chat_template = None


def test_resolve_prompt_format_auto_picks_chat_when_available():
    assert resolve_prompt_format("auto", _WithChatTemplate()) == "chat"


def test_resolve_prompt_format_auto_falls_back_to_plain():
    assert resolve_prompt_format("auto", _WithoutChatTemplate()) == "plain"


def test_resolve_prompt_format_explicit_choice_wins():
    assert resolve_prompt_format("plain", _WithChatTemplate()) == "plain"
    assert resolve_prompt_format("chat", _WithoutChatTemplate()) == "chat"


# --- to_prompt_completion_pairs ---


def test_prompt_completion_pairs_match_formatter(write_jsonl):
    path = write_jsonl(
        "train.jsonl",
        [example("X firmasının satışları"), example("Y firmasının satışları")],
    )
    result = load_examples(path)
    formatter = get_formatter("plain")

    pairs = to_prompt_completion_pairs(result, formatter)

    assert len(pairs) == 2
    for pair, ex in zip(pairs, result.examples, strict=True):
        assert set(pair) == {"prompt", "completion"}
        assert pair["prompt"] == formatter.format_inference(ex.input)
        assert pair["completion"] == formatter.format_train(ex)[len(pair["prompt"]) :]


def test_prompt_completion_pairs_preserve_turkish_characters(write_jsonl):
    path = write_jsonl(
        "train.jsonl",
        [example("İstanbul şubesindeki satışları göster")],
    )
    result = load_examples(path)
    pairs = to_prompt_completion_pairs(result, get_formatter("plain"))
    assert "İstanbul" in pairs[0]["prompt"]


def test_prompt_completion_pairs_empty_for_no_examples(write_jsonl):
    path = write_jsonl("train.jsonl", [])
    result = load_examples(path)
    assert to_prompt_completion_pairs(result, get_formatter("plain")) == []


# --- V2: challenge_test guard (CLAUDE_V2_MIGRATION.md: "Never train on challenge_test.jsonl") ---


def test_rejects_challenge_test_by_filename(write_jsonl):
    path = write_jsonl("challenge_test.jsonl", [example("X firması")])
    result = load_examples(path)
    with pytest.raises(TrainingDataError, match="challenge test"):
        _reject_challenge_test(result, "train_file")


def test_rejects_challenge_test_by_metadata_split(write_jsonl):
    row = example("X firması")
    row["metadata"] = {"split": "challenge_test"}
    path = write_jsonl("renamed.jsonl", [row])
    result = load_examples(path)
    with pytest.raises(TrainingDataError, match="challenge_test"):
        _reject_challenge_test(result, "train_file")


def test_accepts_normal_train_file(write_jsonl):
    path = write_jsonl("train.jsonl", [example("X firması")])
    result = load_examples(path)
    _reject_challenge_test(result, "train_file")  # raise etmemeli


def test_accepts_file_with_unrelated_metadata_split(write_jsonl):
    row = example("X firması")
    row["metadata"] = {"split": "train"}
    path = write_jsonl("train.jsonl", [row])
    result = load_examples(path)
    _reject_challenge_test(result, "train_file")  # raise etmemeli
