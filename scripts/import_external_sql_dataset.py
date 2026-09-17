#!/usr/bin/env python3
"""TURSpider ve TUR2SQL'i proje JSON semamiza donusturur.

Veri edinme plani Bolum B: `datasets/external/tur2sql/` ve
`datasets/external/turspider/` altina indirilen ham verilerden, semamizla
UYUMLU (tek tablo, basit filtre/agregasyon) satirlari cikarir.

Iki farkli kaynak, iki farkli donusum stratejisi gerektirir:

  TUR2SQL  - zaten WikiSQL bicimde (tek tablo, `sql.agg`/`sql.conds`
             sozlugu ile yapilandirilmis) -- SQL string PARSE EDILMEZ,
             dogrudan alan esleme yapilir. Yuksek guven.
  TURSpider - ham SQL string'i icerir (Spider tarzi, coklu-domain, join'ler
             yaygin) -- sqlglot ile parse edilip yalnizca tek-tablo/basit
             sorgular (join, subquery, HAVING, UNION, window, OR YOK)
             donusturulur. Dusuk-orta guven.

Cikti iki turludur:
  - egitim adayi (kaynagin kendi TRAIN split'inden)      -> --train-out-dir
  - OOD degerlendirme (kaynagin kendi DEV/TEST split'inden) -> --ood-out-dir
    Bu dosyalar asla egitimde kullanilmaz; yalniz
    `scripts/benchmark.py --test_file` ile okunur.

ONEMLI: Donusturulen etiketler KORU KORUNE guvenilmemelidir. Egitime
eklemeden once (plan D.6-D.8): `scripts/validate_dataset.py` ile dogrula,
mevcut split'lerle sizinti kontrolu yap, ve bir ornek altkumeyi elle
incele.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlglot  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from sqlglot import exp  # noqa: E402

from app.schemas.query import QueryRequest, to_canonical_json  # noqa: E402

# --- TUR2SQL: WikiSQL agg/cond kod tablolari -------------------------------
# Standart WikiSQL sozlesmesi; tur2sql/data/train.jsonl uzerinde ornekler
# (agg=0->"SELECT baslik", agg=1->"SELECT max(numara)", agg=3->"SELECT
# count(numara)") ile dogrulandi. agg=2/4/5 (min/sum/avg) ve cond op=1/2
# (>/<) bu dataset'te gozlemlenmedi ama WikiSQL standardina gore eklendi.
_AGG_TO_OPERATION = {0: "select", 1: "max", 2: "min", 3: "count", 4: "sum", 5: "average"}
_COND_OP_TO_OPERATOR = {0: "eq", 1: "gt", 2: "lt"}

# --- TURSpider: sqlglot AST -> semamiz esleme tablolari --------------------
_AGG_EXPR_TO_OPERATION = {
    exp.Sum: "sum",
    exp.Avg: "average",
    exp.Min: "min",
    exp.Max: "max",
    exp.Count: "count",
}
_COMPARISON_TO_OPERATOR = {
    exp.EQ: "eq",
    exp.NEQ: "neq",
    exp.GT: "gt",
    exp.GTE: "gte",
    exp.LT: "lt",
    exp.LTE: "lte",
}


class ConversionStats:
    def __init__(self) -> None:
        self.read = 0
        self.incompatible = 0
        self.schema_invalid = 0
        self.converted = 0

    def report(self, label: str) -> str:
        return (
            f"{label}: {self.read} okundu, {self.converted} donusturuldu, "
            f"{self.incompatible} uyumsuz (atlandi), "
            f"{self.schema_invalid} QueryRequest dogrulamasinda basarisiz"
        )


# --- TUR2SQL -----------------------------------------------------------------


def load_tur2sql_tables(tables_path: Path) -> dict[str, list[str]]:
    tables: dict[str, list[str]] = {}
    with tables_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tables[row["id"]] = row["header"]
    return tables


def convert_tur2sql_row(row: dict[str, Any], tables: dict[str, list[str]]) -> dict[str, Any] | None:
    headers = tables.get(row.get("table_id"))
    if not headers:
        return None
    sql = row.get("sql") or {}
    agg = sql.get("agg")
    sel = sql.get("sel")
    operation = _AGG_TO_OPERATION.get(agg)
    if operation is None or sel is None or not (0 <= sel < len(headers)):
        return None
    target = headers[sel]

    filters: list[dict[str, Any]] = []
    for cond in sql.get("conds", []):
        if len(cond) != 3:
            return None
        col_idx, op_idx, value = cond
        operator = _COND_OP_TO_OPERATOR.get(op_idx)
        if operator is None or not (0 <= col_idx < len(headers)) or value is None:
            return None
        filters.append({"field": headers[col_idx], "operator": operator, "value": value})

    return {
        "operation": operation,
        "target": target,
        "filters": filters,
        "group_by": [],
        "order_by": None,
        "limit": None,
        "status": "valid",
        "clarification": None,
    }


def process_tur2sql_file(
    jsonl_path: Path, tables: dict[str, list[str]], stats: ConversionStats
) -> list[dict[str, Any]]:
    results = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats.read += 1
            row = json.loads(line)
            question = row.get("question", "").strip()
            if not question:
                stats.incompatible += 1
                continue
            converted = convert_tur2sql_row(row, tables)
            if converted is None:
                stats.incompatible += 1
                continue
            validated = _validate(converted, stats)
            if validated is not None:
                results.append({"input": question, "output": validated})
    return results


# --- TURSpider -----------------------------------------------------------------


def _literal_value(node: exp.Expression | None) -> Any:
    if node is None or not isinstance(node, exp.Literal):
        return None
    if node.is_int:
        return int(node.this)
    if node.is_number:
        return float(node.this)
    return node.this


def _classify_where(where: exp.Where | None) -> list[dict[str, Any]] | None:
    if where is None:
        return []
    return _classify_condition(where.this)


def _classify_condition(node: exp.Expression) -> list[dict[str, Any]] | None:
    if isinstance(node, exp.And):
        left = _classify_condition(node.left)
        right = _classify_condition(node.right)
        if left is None or right is None:
            return None
        return left + right
    if isinstance(node, exp.Paren):
        return _classify_condition(node.this)

    left = node.this if hasattr(node, "this") else None
    if not isinstance(left, exp.Column):
        return None
    field = left.name

    if type(node) in _COMPARISON_TO_OPERATOR:
        value = _literal_value(node.expression)
        if value is None:
            return None
        return [{"field": field, "operator": _COMPARISON_TO_OPERATOR[type(node)], "value": value}]

    if isinstance(node, exp.Between):
        low = _literal_value(node.args.get("low"))
        high = _literal_value(node.args.get("high"))
        if low is None or high is None:
            return None
        return [{"field": field, "operator": "between", "value": [low, high]}]

    if isinstance(node, exp.In):
        items = node.args.get("expressions") or []
        values = [_literal_value(i) for i in items]
        if not values or any(v is None for v in values):
            return None
        return [{"field": field, "operator": "in", "value": values}]

    if isinstance(node, exp.Like):
        pattern = _literal_value(node.expression)
        if pattern is None:
            return None
        return [{"field": field, "operator": "contains", "value": str(pattern).strip("%")}]

    return None  # OR, subquery, fonksiyon karsilastirmasi vb. desteklenmiyor


def _agg_target(agg_node: exp.Expression, table_name: str) -> str | None:
    inner = agg_node.this
    if isinstance(inner, exp.Star):
        return table_name
    if isinstance(inner, exp.Column):
        return inner.name
    return None


def _classify_select_list(
    select_exprs: list[exp.Expression], table_name: str, has_group_by: bool
) -> tuple[str, str, list[str]] | None:
    agg_exprs = [e for e in select_exprs if type(e) in _AGG_EXPR_TO_OPERATION]
    plain_cols = [e for e in select_exprs if isinstance(e, exp.Column)]
    if len(agg_exprs) + len(plain_cols) != len(select_exprs):
        return None  # SELECT *, aritmetik ifade, alt-sorgu vb. desteklenmiyor

    if has_group_by:
        if len(agg_exprs) != 1:
            return None
        target = _agg_target(agg_exprs[0], table_name)
        if not target:
            return None
        return _AGG_EXPR_TO_OPERATION[type(agg_exprs[0])], target, [c.name for c in plain_cols]

    if len(agg_exprs) == 1 and not plain_cols:
        target = _agg_target(agg_exprs[0], table_name)
        if not target:
            return None
        return _AGG_EXPR_TO_OPERATION[type(agg_exprs[0])], target, []

    if len(plain_cols) == 1 and not agg_exprs:
        return "select", plain_cols[0].name, []

    return None


def _classify_group_cols(group: exp.Group) -> list[str] | None:
    cols = []
    for e in group.expressions:
        if not isinstance(e, exp.Column):
            return None
        cols.append(e.name)
    return cols


def _classify_order(order: exp.Order | None) -> dict[str, str] | None:
    if order is None:
        return None
    exprs = order.expressions
    if len(exprs) != 1 or not isinstance(exprs[0].this, exp.Column):
        return None
    direction = "desc" if exprs[0].args.get("desc") else "asc"
    return {"field": exprs[0].this.name, "direction": direction}


def _classify_limit(limit: exp.Limit | None) -> int | None:
    if limit is None:
        return None
    value = _literal_value(limit.expression)
    if not isinstance(value, int) or value <= 0:
        return None
    return value


def convert_turspider_row(sql: str) -> dict[str, Any] | None:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    if not isinstance(tree, exp.Select):
        return None
    if tree.find(exp.Join, exp.Subquery, exp.Window) is not None:
        return None
    if tree.args.get("having") is not None:
        return None
    where = tree.args.get("where")
    if where is not None and where.find(exp.Or) is not None:
        return None

    from_expr = tree.args.get("from")
    if from_expr is None:
        return None
    tables = list(from_expr.find_all(exp.Table))
    if len(tables) != 1:
        return None
    table_name = tables[0].name

    group_expr = tree.args.get("group")
    classified = _classify_select_list(tree.expressions, table_name, group_expr is not None)
    if classified is None:
        return None
    operation, target, group_by = classified
    if group_expr is not None:
        gcols = _classify_group_cols(group_expr)
        if gcols is None:
            return None
        group_by = gcols

    filters = _classify_where(where)
    if filters is None:
        return None

    return {
        "operation": operation,
        "target": target,
        "filters": filters,
        "group_by": group_by,
        "order_by": _classify_order(tree.args.get("order")),
        "limit": _classify_limit(tree.args.get("limit")),
        "status": "valid",
        "clarification": None,
    }


def process_turspider_file(
    csv_path: Path, hardness_filter: set[str] | None, stats: ConversionStats
) -> list[dict[str, Any]]:
    results = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            question = (row.get("question") or "").strip()
            sql = (row.get("query") or "").strip()
            hardness = (row.get("hardness") or "").strip()
            if not question or not sql:
                continue
            stats.read += 1
            if hardness_filter is not None and hardness not in hardness_filter:
                stats.incompatible += 1
                continue
            converted = convert_turspider_row(sql)
            if converted is None:
                stats.incompatible += 1
                continue
            validated = _validate(converted, stats)
            if validated is not None:
                results.append({"input": question, "output": validated})
    return results


# --- Ortak yardimcilar -------------------------------------------------------


def _validate(candidate: dict[str, Any], stats: ConversionStats) -> dict[str, Any] | None:
    try:
        model = QueryRequest.model_validate(candidate)
    except ValidationError:
        stats.schema_invalid += 1
        return None
    stats.converted += 1
    return json.loads(to_canonical_json(model))


def drop_leaked_ood_rows(
    train_rows: list[dict[str, Any]], ood_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Kaynagin kendi train/dev split'leri arasinda (nadiren) tekrar eden
    input'lari OOD tarafindan cikarir -- OOD setin 'egitimde hic
    gorulmedi' garantisini korumak icin (bkz. plan D.7)."""
    train_inputs = {row["input"] for row in train_rows}
    return [row for row in ood_rows if row["input"] not in train_inputs]


def write_jsonl(
    rows: list[dict[str, Any]], path: Path, source: str, split: str | None
) -> None:
    """Aynı `input`'a sahip satırları (birebir eşleşme) ilk görülenle sınırlar.

    Spider tarzı kaynaklarda aynı Türkçe soru metni farklı db_id'ler
    altında birden fazla kez geçebilir -- `validate_dataset.py`'nin
    DUPLICATE_INPUT kuralı bunu doğru şekilde reddeder, çünkü aynı girdiye
    iki farklı doğru çıktı, tek bir global model için belirsiz/çelişkili
    bir eğitim sinyalidir.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    seen_inputs: set[str] = set()
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if row["input"] in seen_inputs:
                continue
            seen_inputs.add(row["input"])
            metadata: dict[str, Any] = {"source": source, "family": "external_real", "noise": "real_world"}
            if split is not None:
                metadata["split"] = split
            record = {"input": row["input"], "output": row["output"], "metadata": metadata}
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
    if written != len(rows):
        print(f"  ({path.name}: {len(rows) - written} birebir tekrar eden input elendi)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tur2sql-dir", type=Path, default=Path("datasets/external/tur2sql"))
    parser.add_argument("--turspider-dir", type=Path, default=Path("datasets/external/turspider"))
    parser.add_argument("--train-out-dir", type=Path, default=Path("datasets/external"))
    parser.add_argument("--ood-out-dir", type=Path, default=Path("datasets/ood_eval"))
    parser.add_argument(
        "--turspider-hardness",
        default="",
        help="Virgulle ayrilmis hardness degerleri (easy,medium,hard,extra hard). "
        "Varsayilan bos = filtre yok. sqlglot siniflandiricisi zaten join/subquery/"
        "HAVING/UNION/window/OR iceren sorgulari hardness etiketinden bagimsiz "
        "olarak eliyor; deneysel olarak hardness on-filtresi olmadan ayni guvenlikte "
        "%%60 daha fazla ve daha cesitli (contains/neq/gte/lte, group_by/order_by/"
        "limit) ornek elde edildi -- bkz. import script'inin manuel inceleme notlari.",
    )
    parser.add_argument("--skip-tur2sql", action="store_true")
    parser.add_argument("--skip-turspider", action="store_true")
    args = parser.parse_args()

    hardness_filter = (
        {h.strip() for h in args.turspider_hardness.split(",") if h.strip()}
        if args.turspider_hardness
        else None
    )

    # Once tur2sql/turspider train adaylari YAZILDI, OOD dosyalari HER IKI
    # kaynagin train havuzuna karsi da sizinti kontrolunden gecirilir --
    # aksi halde farkli akademik kaynaklardaki tesaduf ortak sorular
    # (ör. "Kaç tane X var?") birbirinin OOD setine sizabilir.
    all_train_rows: list[dict[str, Any]] = []

    tur2sql_train_rows: list[dict[str, Any]] = []
    if not args.skip_tur2sql:
        tables = load_tur2sql_tables(args.tur2sql_dir / "data" / "data.tables.jsonl")

        train_stats = ConversionStats()
        tur2sql_train_rows = process_tur2sql_file(args.tur2sql_dir / "data" / "train.jsonl", tables, train_stats)
        write_jsonl(tur2sql_train_rows, args.train_out_dir / "tur2sql_converted_train.jsonl", "tur2sql", "train")
        print(train_stats.report("TUR2SQL train"))
        all_train_rows += tur2sql_train_rows

    turspider_train_rows: list[dict[str, Any]] = []
    if not args.skip_turspider:
        train_stats = ConversionStats()
        turspider_train_rows = process_turspider_file(
            args.turspider_dir / "data" / "train.csv", hardness_filter, train_stats
        )
        write_jsonl(
            turspider_train_rows, args.train_out_dir / "turspider_converted_train.jsonl", "turspider", "train"
        )
        print(train_stats.report("TURSpider train"))
        all_train_rows += turspider_train_rows

    if not args.skip_tur2sql:
        ood_stats = ConversionStats()
        ood_rows = []
        for name in ("dev.jsonl", "test.jsonl"):
            ood_rows += process_tur2sql_file(args.tur2sql_dir / "data" / name, tables, ood_stats)
        ood_rows = drop_leaked_ood_rows(all_train_rows, ood_rows)
        write_jsonl(ood_rows, args.ood_out_dir / "tur2sql_ood.jsonl", "tur2sql", None)
        print(ood_stats.report("TUR2SQL dev+test (OOD)"))

    if not args.skip_turspider:
        ood_stats = ConversionStats()
        ood_rows = process_turspider_file(
            args.turspider_dir / "data" / "dev.csv", hardness_filter, ood_stats
        )
        ood_rows = drop_leaked_ood_rows(all_train_rows, ood_rows)
        write_jsonl(ood_rows, args.ood_out_dir / "turspider_ood.jsonl", "turspider", None)
        print(ood_stats.report("TURSpider dev (OOD)"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
