#!/usr/bin/env python3
"""Dataset butunluk dogrulayicisi (spec §21).

Kontroller:
  - her satir gecerli JSON
  - `input` ve `output` alanlari mevcut
  - `output` QueryRequest semasina uyuyor (enum'lar, between/in, limit, yon)
  - bos input yok
  - dosya ici tekrar eden input yok
  - train/validation/test arasi sizinti yok
  - (opsiyonel) alan adlari V1 sozlugune uyuyor

Sorun bulunursa 0 disi kod ile cikar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Script dogrudan calistirildiginda repo kokunu import yoluna ekle.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas.vocabulary import KNOWN_FIELDS, KNOWN_TARGETS  # noqa: E402
from training.dataset import (  # noqa: E402
    DatasetIssue,
    LoadResult,
    collect_field_names,
    distribution_report,
    find_duplicate_inputs,
    find_metadata_split_mismatch,
    find_split_overlap,
    load_examples,
)

SPLIT_ORDER = ("train", "validation", "test")


def check_not_challenge_test(result: LoadResult, split: str) -> list[DatasetIssue]:
    """CLAUDE_V2_MIGRATION.md: 'Never train on challenge_test.jsonl'.

    `--train`/`--validation` olarak yanlislikla bir challenge test dosyasi
    verilmesini erken yakalar. `--test` icin bu kisitlama uygulanmaz --
    challenge_test zaten bir degerlendirme dosyasidir, sadece egitimde
    kullanilmamalidir.
    """
    if split == "test":
        return []

    issues: list[DatasetIssue] = []
    if "challenge_test" in result.path.name.lower():
        issues.append(
            DatasetIssue(
                str(result.path),
                None,
                "CHALLENGE_TEST_MISUSE",
                f"'{split}' split'i bir challenge test dosyasina isaret ediyor gibi "
                "gorunuyor; challenge test setleri asla egitimde/dogrulamada kullanilmaz",
            )
        )
        return issues

    for example, line_no in zip(result.examples, result.line_numbers, strict=True):
        if example.metadata is not None and example.metadata.split == "challenge_test":
            issues.append(
                DatasetIssue(
                    str(result.path),
                    line_no,
                    "CHALLENGE_TEST_MISUSE",
                    f"'{split}' split'inde metadata.split='challenge_test' olan bir "
                    "ornek var; challenge test setleri asla egitimde/dogrulamada kullanilmaz",
                )
            )
            break  # bir kez raporlamak yeterli
    return issues


def check_vocabulary(result: LoadResult, strict: bool) -> list[DatasetIssue]:
    """V1 sozlugune uymayan alan/hedef adlari (spec §5).

    Varsayilanda uyaridir: bu isimler gecicidir ve isverenin gercek semasiyla
    degisecektir. `--strict-vocabulary` ile hataya donusur.
    """
    code = "UNKNOWN_FIELD" if strict else "UNKNOWN_FIELD_WARNING"
    issues: list[DatasetIssue] = []

    for name, lines in sorted(collect_field_names(result).items()):
        if name not in KNOWN_FIELDS:
            issues.append(
                DatasetIssue(
                    str(result.path),
                    lines[0],
                    code,
                    f"alan adi V1 sozlugunde yok: {name!r} ({len(lines)} kez)",
                )
            )

    target_code = "UNKNOWN_TARGET" if strict else "UNKNOWN_TARGET_WARNING"
    seen_targets: dict[str, int] = {}
    for example, line_no in zip(result.examples, result.line_numbers, strict=True):
        # V2: status != valid iken target None'dur (kontrol edilecek bir sey yok).
        if example.output.target is not None:
            seen_targets.setdefault(example.output.target, line_no)
    for target, line_no in sorted(seen_targets.items()):
        if target not in KNOWN_TARGETS:
            issues.append(
                DatasetIssue(
                    str(result.path),
                    line_no,
                    target_code,
                    f"hedef V1 sozlugunde yok: {target!r}",
                )
            )

    return issues


def build_report(
    paths: dict[str, Path], strict_vocabulary: bool
) -> tuple[dict[str, object], list[DatasetIssue], list[DatasetIssue]]:
    """Tum kontrolleri calistirir; (rapor, hatalar, uyarilar) doner."""
    results: dict[str, LoadResult] = {}
    errors: list[DatasetIssue] = []
    warnings: list[DatasetIssue] = []

    for split, path in paths.items():
        result = load_examples(path)
        results[split] = result
        errors.extend(result.issues)

        # Tam ayni girdi (spec §21) her zaman hatadir. Ayni split icindeki
        # NEAR_DUPLICATE (buyuk/kucuk harf, noktalama farki) buyuk organik
        # dataset'lerde kacinilmazdir ve sizinti anlamina gelmez (ayni
        # split'te kaliyor) -- bu yuzden uyari olarak raporlanir. Splitler
        # ARASI near-duplicate (SPLIT_NEAR_LEAKAGE) hala hatadir; asagida.
        for issue in find_duplicate_inputs(result):
            (warnings if issue.code == "NEAR_DUPLICATE_INPUT" else errors).append(issue)

        errors.extend(find_metadata_split_mismatch(result, split))
        errors.extend(check_not_challenge_test(result, split))

        for issue in check_vocabulary(result, strict_vocabulary):
            (errors if strict_vocabulary else warnings).append(issue)

    # Sizinti kontrolu: her split cifti bir kez.
    ordered = [s for s in SPLIT_ORDER if s in results]
    for i, left_split in enumerate(ordered):
        for right_split in ordered[i + 1 :]:
            errors.extend(find_split_overlap(results[left_split], results[right_split]))

    report: dict[str, object] = {
        "splits": {
            split: {
                "path": str(result.path),
                "examples": len(result.examples),
                "distribution": distribution_report(result),
            }
            for split, result in results.items()
        },
        "total_examples": sum(len(r.examples) for r in results.values()),
    }
    return report, errors, warnings


def print_human_report(
    report: dict[str, object], errors: list[DatasetIssue], warnings: list[DatasetIssue]
) -> None:
    splits: dict[str, dict] = report["splits"]  # type: ignore[assignment]

    print("=" * 70)
    print("DATASET DOGRULAMA")
    print("=" * 70)

    for split, info in splits.items():
        print(f"\n[{split}]  {info['path']}")
        print(f"  ornek sayisi: {info['examples']}")
        for section, counts in info["distribution"].items():
            if counts:
                rendered = ", ".join(f"{k}={v}" for k, v in counts.items())
                print(f"  {section:16s} {rendered}")

    print(f"\ntoplam ornek: {report['total_examples']}")

    if warnings:
        print(f"\n--- UYARI ({len(warnings)}) ---")
        for issue in warnings:
            print(f"  {issue.format()}")

    if errors:
        print(f"\n--- HATA ({len(errors)}) ---")
        for issue in errors:
            print(f"  {issue.format()}")
        print("\nSONUC: BASARISIZ")
    else:
        print("\nSONUC: BASARILI")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="JSONL dataset split'lerini QueryRequest semasina gore dogrular."
    )
    parser.add_argument("--train", type=Path, help="train.jsonl yolu")
    parser.add_argument("--validation", type=Path, help="validation.jsonl yolu")
    parser.add_argument("--test", type=Path, help="test.jsonl yolu")
    parser.add_argument(
        "--strict-vocabulary",
        action="store_true",
        help="V1 sozlugu disindaki alan/hedef adlarini hata say (varsayilan: uyari)",
    )
    parser.add_argument("--json", action="store_true", help="raporu JSON olarak yazdir")
    args = parser.parse_args(argv)

    paths = {
        split: getattr(args, split) for split in SPLIT_ORDER if getattr(args, split) is not None
    }
    if not paths:
        parser.error("en az bir split verilmelidir (--train / --validation / --test)")

    report, errors, warnings = build_report(paths, args.strict_vocabulary)

    if args.json:
        payload = {
            **report,
            "errors": [vars(i) for i in errors],
            "warnings": [vars(i) for i in warnings],
            "ok": not errors,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_human_report(report, errors, warnings)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
