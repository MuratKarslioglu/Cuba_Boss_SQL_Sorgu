#!/usr/bin/env python3
"""run_003 icin tek komutluk egitim + benchmark orkestratoru.

`training/train.py` ve `scripts/benchmark.py` zaten test edilmis `main(argv)`
giris noktalarini sirayla cagirir -- egitim/degerlendirme mantigi burada
TEKRAR YAZILMAZ. Bu betik yalnizca bu belirli run icin sabit degerlerle
orkestrasyon yapar; parametrik hale getirmek (argparse vb.) su an icin
istenmeyen bir genelleme.

Kullanim:
    python main.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import benchmark  # noqa: E402

from training import train  # noqa: E402

RUN_NAME = "run_003"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_FILE = "datasets/external/train_synthetic_plus_external.jsonl"
VALIDATION_FILE = "datasets/sql_json_dataset_v2_professional/validation.jsonl"
MODEL_OUTPUT_DIR = f"models/{RUN_NAME}"

# models/run_002/training_config.json ile birebir ayni -- adil karsilastirma
# icin sadece TRAIN_FILE degisiyor (12k sentetik -> 21k sentetik+gercek).
EPOCHS = "3"
LEARNING_RATE = "2e-4"
BATCH_SIZE = "4"
GRADIENT_ACCUMULATION_STEPS = "4"
LORA_R = "16"
LORA_ALPHA = "32"
LORA_DROPOUT = "0.05"
PROMPT_FORMAT = "chat"

BENCHMARK_TEST_FILES = {
    "test": "datasets/sql_json_dataset_v2_professional/test.jsonl",
    "challenge_test": "datasets/sql_json_dataset_v2_professional/challenge_test.jsonl",
    "tur2sql_ood": "datasets/ood_eval/tur2sql_ood.jsonl",
    "turkish_real_nl": "datasets/ood_eval/turkish_real_nl.jsonl",
    "turspider_ood": "datasets/ood_eval/turspider_ood.jsonl",
}


def run_training() -> tuple[int, float]:
    argv = [
        "--model_name", MODEL_NAME,
        "--train_file", TRAIN_FILE,
        "--validation_file", VALIDATION_FILE,
        "--output_dir", MODEL_OUTPUT_DIR,
        "--epochs", EPOCHS,
        "--learning_rate", LEARNING_RATE,
        "--batch_size", BATCH_SIZE,
        "--gradient_accumulation_steps", GRADIENT_ACCUMULATION_STEPS,
        "--lora_r", LORA_R,
        "--lora_alpha", LORA_ALPHA,
        "--lora_dropout", LORA_DROPOUT,
        "--prompt_format", PROMPT_FORMAT,
    ]  # fmt: skip
    start = time.perf_counter()
    exit_code = train.main(argv)
    return exit_code, time.perf_counter() - start


def run_benchmarks() -> dict[str, dict[str, object]]:
    """Her test dosyasi icin ayri output_dir kullanir -- ayni gun/model icin
    benchmark.py'nin `<model>_<YYYYMMDD>.json` dosya adi cakismasin diye."""
    results: dict[str, dict[str, object]] = {}
    for label, test_file in BENCHMARK_TEST_FILES.items():
        output_dir = f"benchmarks/{RUN_NAME}/{label}"
        argv = [
            "--model_name", MODEL_NAME,
            "--adapter_path", MODEL_OUTPUT_DIR,
            "--test_file", test_file,
            "--output_dir", output_dir,
            "--dump_predictions", f"{output_dir}/predictions.jsonl",
        ]  # fmt: skip
        start = time.perf_counter()
        try:
            exit_code = benchmark.main(argv)
        except Exception as exc:  # bir test setinin patlamasi digerlerini durdurmasin
            print(f"HATA: {label} benchmark'i patladi: {exc}", file=sys.stderr)
            exit_code = 1
        results[label] = {
            "test_file": test_file,
            "output_dir": output_dir,
            "exit_code": exit_code,
            "duration_seconds": time.perf_counter() - start,
        }
    return results


def print_summary(training: dict[str, object], benchmarks: dict[str, dict[str, object]]) -> None:
    print("\n" + "=" * 70)
    print(f"PIPELINE OZETI -- {RUN_NAME}")
    print("=" * 70)
    status = "OK" if training["exit_code"] == 0 else "BASARISIZ"
    print(f"[egitim] {status}  ({training['duration_seconds']:.0f}s)  -> {MODEL_OUTPUT_DIR}")
    for label, info in benchmarks.items():
        status = "OK" if info["exit_code"] == 0 else "BASARISIZ"
        print(f"[{label}] {status}  ({info['duration_seconds']:.0f}s)  -> {info['output_dir']}")


def main() -> int:
    training_exit_code, training_seconds = run_training()
    training_summary = {"exit_code": training_exit_code, "duration_seconds": training_seconds}

    if training_exit_code != 0:
        print("HATA: egitim basarisiz oldu, benchmark'lara gecilmiyor.", file=sys.stderr)
        print_summary(training_summary, {})
        return 1

    benchmark_summaries = run_benchmarks()

    summary_path = Path(f"benchmarks/{RUN_NAME}/pipeline_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"training": training_summary, "benchmarks": benchmark_summaries}, indent=2),
        encoding="utf-8",
    )

    print_summary(training_summary, benchmark_summaries)
    print(f"\nOzet JSON: {summary_path}")

    any_benchmark_failed = any(info["exit_code"] != 0 for info in benchmark_summaries.values())
    return 1 if any_benchmark_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
