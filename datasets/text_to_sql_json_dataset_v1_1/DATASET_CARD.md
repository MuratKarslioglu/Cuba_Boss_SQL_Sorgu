# Turkish Text → SQL-JSON Dataset V1.1

## Purpose

This dataset is designed for supervised fine-tuning of a small Turkish language model that converts natural-language business/sales requests into a strict JSON query representation.

It is intentionally **not** a direct text-to-SQL dataset.

## Size

- Total: 1000
- Train: 700
- Validation: 150
- Test: 150

## Complexity Levels

- L1: single-filter/simple aggregation
- L2: multiple filters
- L3: date and date-range semantics
- L4: grouping and aggregation
- L5: ordering + limit
- L6: mixed/advanced operators (`in`, `gt`, `contains`, `neq`) and compound requests

## JSON Contract

Every example contains:

```json
{
  "operation": "count|sum|average|min|max|select",
  "target": "string",
  "filters": [
    {
      "field": "string",
      "operator": "eq|neq|gt|gte|lt|lte|between|in|contains",
      "value": "any"
    }
  ],
  "group_by": [],
  "order_by": null,
  "limit": null
}
```

## Leakage Policy

Paraphrases generated from the same semantic signature are assigned to the same split.
No semantic signature is shared between train, validation, and test in this V1 build.

## Important Limitation

The entities and schema are synthetic and generic. They are placeholders for architecture validation and fine-tuning experiments.

When the employer supplies the real database schema and required JSON contract, this dataset should be replaced or adapted into V2.

## Files

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `all.jsonl`
- `manifest.json`

## Reproducibility

Random seed: 42

V1.1 balances advanced operator families more deliberately.
