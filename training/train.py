#!/usr/bin/env python3
"""LoRA/SFT egitim scripti (spec §12, §29).

Sorumluluk ayrimi:
    dataset yukleme/dogrulama  -> training.dataset
    prompt formatlama          -> training.formatting
    model/cihaz/LoRA           -> app.inference.model_loader
    bu dosya                   -> yalnizca CLI + orkestrasyon

Model adi ve hiperparametreler kaynak kod degistirilmeden CLI'dan
degistirilebilir (spec §13, §29). Asagidaki varsayilanlar spec'teki
ornek degerlerdir, optimal kabul edilmez.

`trl`/`torch` import'lari fonksiyon icine gec birakilmistir; boylece bu
modul import edildiginde (ornegin CLI --help icin) agir ML bagimliliklari
zorunlu olmaz ve saf yardimci fonksiyonlar torch'suz test edilebilir.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Script dogrudan calistirildiginda repo kokunu import yoluna ekle.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.errors import AppError, ErrorType  # noqa: E402
from app.schemas.query import to_canonical_json  # noqa: E402
from training.dataset import LoadResult, load_examples  # noqa: E402
from training.formatting import PromptFormatter, get_formatter  # noqa: E402

logger = logging.getLogger(__name__)

# Spec §31: deterministik davranis icin sabit tohum.
DEFAULT_SEED = 42


class TrainingDataError(AppError):
    error_type = ErrorType.DATASET_ERROR


def set_determinism(seed: int) -> None:
    """Spec §31: mumkun oldugunca deterministik davranis.

    Not: bu, farkli donanim/backend'lerde matematiksel olarak birebir ayni
    sonucu garanti etmez (spec §31); yalnizca rastgeleligi azaltir.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _require_valid(result: LoadResult, label: str) -> LoadResult:
    """Gecersiz dataset'le sessizce devam etmez (spec §37)."""
    if not result.ok:
        details = "\n".join(f"  {issue.format()}" for issue in result.issues)
        raise TrainingDataError(f"{label} dogrulamadan gecemedi:\n{details}")
    return result


def _reject_challenge_test(result: LoadResult, label: str) -> None:
    """CLAUDE_V2_MIGRATION.md: 'Never train on challenge_test.jsonl'.

    Iki bagimsiz sinyalle kontrol edilir -- dosya adi VE (varsa)
    `metadata.split` -- boylece dosya yeniden adlandirilsa bile ornekler
    kendi kokenini tasidigi surece koruma calismaya devam eder.
    """
    if "challenge_test" in result.path.name.lower():
        raise TrainingDataError(
            f"{label} bir challenge test dosyasina isaret ediyor gibi gorunuyor "
            f"({result.path.name}); challenge test setleri asla egitimde kullanilmaz."
        )
    for example in result.examples:
        if example.metadata is not None and example.metadata.split == "challenge_test":
            raise TrainingDataError(
                f"{label} icinde metadata.split='challenge_test' olan ornekler var; "
                "challenge test setleri asla egitimde kullanilmaz."
            )


def to_prompt_completion_pairs(
    result: LoadResult, formatter: PromptFormatter
) -> list[dict[str, str]]:
    """RawExample listesini TRL'in prompt/completion formatina cevirir.

    Modern TRL surumleri "prompt"/"completion" kolonlarini otomatik tanir
    ve completion-only loss'u (yalnizca cevap tokenlari uzerinden) kendisi
    uygular; elle response_template esletirmeye gerek kalmaz.
    """
    return [
        {
            "prompt": formatter.format_inference(example.input),
            "completion": to_canonical_json(example.output),
        }
        for example in result.examples
    ]


def build_dataset(result: LoadResult, formatter: PromptFormatter) -> Any:
    from datasets import Dataset

    return Dataset.from_list(to_prompt_completion_pairs(result, formatter))


def resolve_prompt_format(name: str, tokenizer: object) -> str:
    """`auto`: tokenizer'in chat template'i varsa chat, yoksa plain (spec §14)."""
    if name != "auto":
        return name
    return "chat" if getattr(tokenizer, "chat_template", None) else "plain"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turkce -> JSON cikarici icin LoRA/SFT egitimi (spec §12, §29)."
    )
    parser.add_argument(
        "--model_name", required=True, help="HuggingFace model id veya yerel model yolu"
    )
    parser.add_argument("--train_file", required=True, type=Path)
    parser.add_argument("--validation_file", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)

    # Spec §29'daki ornek degerler; optimal kabul edilmez.
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument(
        "--prompt_format",
        choices=["auto", "plain", "chat"],
        default="auto",
        help="auto: tokenizer'in chat template'i varsa chat, yoksa plain (spec §14)",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="egitimde prompt+completion icin maksimum token uzunlugu "
        "(uretimdeki MAX_NEW_TOKENS'tan ayri bir kavramdir)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", choices=["no", "epoch", "steps"], default="epoch")
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None)

    return parser.parse_args(argv)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Egitimi calistirir; adapter/tokenizer/config/eval ozetini kaydeder."""
    # Gec import: torch/trl yalnizca gercekten egitim calistirilirken yuklenir.
    import torch
    from trl import SFTConfig, SFTTrainer

    from app.inference.model_loader import build_lora_config, load_base_model

    set_determinism(args.seed)

    train_result = _require_valid(load_examples(args.train_file), "train_file")
    val_result = _require_valid(load_examples(args.validation_file), "validation_file")
    _reject_challenge_test(train_result, "train_file")
    _reject_challenge_test(val_result, "validation_file")

    loaded = load_base_model(args.model_name, device=args.device, dtype=args.dtype)

    prompt_format = resolve_prompt_format(args.prompt_format, loaded.tokenizer)
    formatter_kwargs = {"tokenizer": loaded.tokenizer} if prompt_format == "chat" else {}
    formatter = get_formatter(prompt_format, **formatter_kwargs)

    train_dataset = build_dataset(train_result, formatter)
    eval_dataset = build_dataset(val_result, formatter)

    lora_config = build_lora_config(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        seed=args.seed,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy="epoch",
        report_to="none",
        bf16=(loaded.dtype == torch.bfloat16 and loaded.device == "cuda"),
        fp16=(loaded.dtype == torch.float16 and loaded.device == "cuda"),
    )

    trainer = SFTTrainer(
        model=loaded.model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=loaded.tokenizer,
        peft_config=lora_config,
    )

    resume = str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    trainer.train(resume_from_checkpoint=resume)

    eval_metrics = trainer.evaluate()

    # Spec §29: adapter, tokenizer metadata, training config, evaluation summary.
    trainer.save_model(str(output_dir))
    loaded.tokenizer.save_pretrained(str(output_dir))

    config_payload = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "resolved_device": loaded.device,
        "resolved_dtype": str(loaded.dtype),
        "prompt_format": prompt_format,
        "train_examples": len(train_result.examples),
        "validation_examples": len(val_result.examples),
        "trained_at": datetime.now(UTC).isoformat(),
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "eval_summary.json").write_text(
        json.dumps(eval_metrics, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    return {"config": config_payload, "eval": eval_metrics}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        summary = run_training(args)
    except AppError as exc:
        logger.error("egitim basarisiz [%s]: %s", exc.error_type.value, exc.message)
        return 1

    logger.info("egitim tamamlandi -> %s", args.output_dir)
    print(json.dumps(summary["eval"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
