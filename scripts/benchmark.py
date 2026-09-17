#!/usr/bin/env python3
"""Performans ve dogruluk benchmarki (spec §23).

    python scripts/benchmark.py \\
      --model_name <HF_MODEL_ID> \\
      --adapter_path ./models/latest \\
      --test_file datasets/text_to_sql_json_dataset_v1_1/test.jsonl

Held-out test seti uzerinde calisir. Olculenler: model yukleme suresi,
ortalama/medyan/p95 gecikme, prompt/uretilen token sayisi, mumkunse tepe
bellek, valid JSON orani, exact match, semantic match (spec §23).

Sonuclar `<output_dir>/<model>_<YYYYMMDD>.json`'a yazilir.

Spec §17: sifir-retry ve tek-retry ayri ayri benchmarklanmalidir. Varsayilan
`--allow_retry` KAPALIDIR (ilk-deneme dogrulugu spec'in birincil metrigidir);
tek-retry kolunu olcmek icin `--allow_retry` ile ayrica calistirin.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

# Script dogrudan calistirildiginda repo kokunu import yoluna ekle.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.dataset import load_examples  # noqa: E402
from training.evaluate import (  # noqa: E402
    EvaluationRecord,
    EvaluationSummary,
    evaluate_records,
    evaluate_rejection,
    group_by_family,
    group_by_noise,
    group_by_status,
    latency_stats,
    status_confusion_matrix,
    token_stats,
)


def _summaries_to_dict(groups: dict[str, EvaluationSummary]) -> dict[str, dict[str, object]]:
    return {key: summary.to_dict() for key, summary in groups.items()}


def dump_predictions(records: list[EvaluationRecord], path: Path) -> None:
    """Her kayit icin {input, expected, predicted} JSONL'ini yazar (kullanicilara
    "su cumleyi verdik, model su JSON'u uretti" gostermek icin)."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def sanitize_model_name(name: str) -> str:
    """Dosya adi olarak guvenli bir slug uretir (spec §23 ornegi: model_name_YYYYMMDD.json)."""
    return name.strip().replace("/", "__").replace(" ", "_")


def peak_memory_mb() -> float | None:
    """Mumkunse tepe bellek kullanimini MB olarak doner (spec §23: 'where practical').

    `resource` modulu Windows'ta yoktur; bu durumda None doner (spec §37:
    sessizce yanlis bir deger uretmek yerine acikca eksik birakilir).
    `ru_maxrss` Linux'ta KB, macOS/BSD'de byte birimindedir.
    """
    try:
        import resource
    except ImportError:
        return None

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 if platform.system() == "Linux" else 1024 * 1024
    return raw / divisor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test seti uzerinde performans/dogruluk benchmarki (spec §23)."
    )
    parser.add_argument("--model_name", required=True, help="HuggingFace model id veya yerel yol")
    parser.add_argument("--adapter_path", default=None, type=Path)
    parser.add_argument("--test_file", required=True, type=Path)
    parser.add_argument("--output_dir", default=Path("benchmarks"), type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto"
    )
    parser.add_argument("--prompt_format", choices=["auto", "plain", "chat"], default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument(
        "--allow_retry",
        action="store_true",
        help="spec §17'nin tek-retry kolunu olcer (varsayilan: sifir-retry / ilk-deneme)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="hizli deneme icin test setinden en fazla N ornek"
    )
    parser.add_argument(
        "--dump_predictions",
        type=Path,
        default=None,
        help="her ornek icin {input, expected, predicted} JSONL'ini bu yola yaz",
    )
    return parser


def build_report(
    args: argparse.Namespace,
    records: list[EvaluationRecord],
    model_load_seconds: float,
    resolved_device: str,
    resolved_dtype: str,
) -> dict[str, object]:
    """Saf rapor birlestirme (torch gerektirmez, ayri test edilebilir).

    V2 (DATASET_CARD.md): status/rejection/aile/gurultu bazinda kirilim da
    rapora dahil edilir. Bir dataset'te `status`/`metadata.family`/
    `metadata.noise` yoksa (V1 dataset) ilgili boyutlar bos sozluk olarak
    kalir -- hicbir sey uydurulmaz.
    """
    summary = evaluate_records(records)
    return {
        "model_name": args.model_name,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "device": resolved_device,
        "dtype": resolved_dtype,
        "retry_enabled": args.allow_retry,
        "prompt_format": args.prompt_format,
        "test_file": str(args.test_file),
        "test_examples": len(records),
        "model_load_seconds": model_load_seconds,
        "latency": latency_stats(records),
        "tokens": token_stats(records),
        "peak_memory_mb": peak_memory_mb(),
        "generated_at": datetime.now(UTC).isoformat(),
        **summary.to_dict(),
        "rejection": evaluate_rejection(records).to_dict(),
        "status_confusion_matrix": status_confusion_matrix(records),
        "by_status": _summaries_to_dict(group_by_status(records)),
        "by_family": _summaries_to_dict(group_by_family(records)),
        "by_noise": _summaries_to_dict(group_by_noise(records)),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    test_result = load_examples(args.test_file)
    if not test_result.ok:
        details = "\n".join(f"  {issue.format()}" for issue in test_result.issues)
        print(f"HATA: test dosyasi dogrulamadan gecemedi:\n{details}", file=sys.stderr)
        return 1

    examples = test_result.examples
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        print("HATA: benchmark icin ornek kalmadi (bos dosya veya --limit 0)", file=sys.stderr)
        return 1

    # Gec import: --help ve dogrulama hatalari torch'u zorunlu kilmaz.
    from app.inference.extractor import extract
    from app.inference.model_loader import load_adapter, load_base_model
    from training.formatting import get_formatter
    from training.train import resolve_prompt_format

    load_start = time.perf_counter()
    loaded = load_base_model(args.model_name, device=args.device, dtype=args.dtype)
    if args.adapter_path and Path(args.adapter_path).exists():
        loaded.model = load_adapter(loaded.model, args.adapter_path)
    model_load_seconds = time.perf_counter() - load_start

    prompt_format = resolve_prompt_format(args.prompt_format, loaded.tokenizer)
    formatter_kwargs = {"tokenizer": loaded.tokenizer} if prompt_format == "chat" else {}
    formatter = get_formatter(prompt_format, **formatter_kwargs)

    records: list[EvaluationRecord] = []
    for example in examples:
        outcome = extract(
            example.input,
            loaded,
            formatter,
            max_new_tokens=args.max_new_tokens,
            allow_retry=args.allow_retry,
        )
        records.append(
            EvaluationRecord(
                input=example.input,
                expected=example.output,
                result=outcome.result,
                latency_ms=outcome.timing.latency_ms,
                prompt_tokens=outcome.timing.prompt_tokens,
                generated_tokens=outcome.timing.generated_tokens,
                metadata=example.metadata,
            )
        )

    report = build_report(args, records, model_load_seconds, loaded.device, str(loaded.dtype))

    if args.dump_predictions is not None:
        args.dump_predictions.parent.mkdir(parents=True, exist_ok=True)
        dump_predictions(records, args.dump_predictions)
        print(f"Ornek bazli ciktilar yazildi: {args.dump_predictions}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        args.output_dir / f"{sanitize_model_name(args.model_name)}_{date.today():%Y%m%d}.json"
    )
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSonuclar yazildi: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
