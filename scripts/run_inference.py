#!/usr/bin/env python3
"""Yerel CLI cikarimi (spec §20).

    python scripts/run_inference.py --text "X firmasındaki Y ürününün satış sayısını getir."

Model/adapter yolu argumanlarla verilmezse `.env` uzerinden `Settings`e
duser (spec §25); boylece model kaynak kodda sabitlenmez (spec §13).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Script dogrudan calistirildiginda repo kokunu import yoluna ekle.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.errors import AppError  # noqa: E402
from app.schemas.query import to_canonical_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Turkce -> JSON cikarim CLI'i (spec §20).")
    parser.add_argument("--text", required=True, help="Turkce dogal dil sorgusu")
    parser.add_argument("--model_name", default=settings.base_model or None)
    parser.add_argument("--adapter_path", default=settings.adapter_path)
    parser.add_argument("--max_new_tokens", type=int, default=settings.max_new_tokens)
    parser.add_argument("--device", default=settings.device)
    parser.add_argument("--dtype", default=settings.model_dtype)
    parser.add_argument("--prompt_format", choices=["auto", "plain", "chat"], default="auto")
    parser.add_argument(
        "--no_retry",
        action="store_true",
        help="spec §17'deki tek seferlik onarimi devre disi birak",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    if not args.model_name:
        print(
            "HATA: model adi verilmedi. --model_name gecirin veya BASE_MODEL'i .env'de ayarlayin.",
            file=sys.stderr,
        )
        return 1

    # Gec import: --help ve arguman hatalari torch'u zorunlu kilmaz.
    from app.inference.extractor import extract
    from app.inference.model_loader import load_adapter, load_base_model
    from training.formatting import get_formatter
    from training.train import resolve_prompt_format

    loaded = load_base_model(args.model_name, device=args.device, dtype=args.dtype)

    adapter_path = Path(args.adapter_path)
    if adapter_path.exists():
        loaded.model = load_adapter(loaded.model, adapter_path)

    prompt_format = resolve_prompt_format(args.prompt_format, loaded.tokenizer)
    formatter_kwargs = {"tokenizer": loaded.tokenizer} if prompt_format == "chat" else {}
    formatter = get_formatter(prompt_format, **formatter_kwargs)

    start = time.perf_counter()
    try:
        outcome = extract(
            args.text,
            loaded,
            formatter,
            max_new_tokens=args.max_new_tokens,
            allow_retry=not args.no_retry,
        )
    except AppError as exc:
        print(f"HATA [{exc.error_type.value}]: {exc.message}", file=sys.stderr)
        return 1
    result = outcome.result
    latency_ms = (time.perf_counter() - start) * 1000

    print("Input:")
    print(args.text)
    print()
    print("Output:")
    if result.success:
        print(to_canonical_json(result.data))
    else:
        print(json.dumps(result.error.model_dump(mode="json"), ensure_ascii=False))
    print()
    print(f"Valid: {str(result.success).lower()}")
    print(f"Latency: {latency_ms:.0f} ms")

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
