# Turkish Text → SQL-JSON Extractor
## Project Architecture, Development Plan, and Implementation Specification

## 1. Project Goal

The goal of this project is to build a **small, local, deterministic language model system** that converts a Turkish natural-language request into a strict JSON representation containing the information required to later build an SQL query.

The first version of the project does **not** need to connect to a real database and does **not** need to execute SQL.

The initial scope is:

```text
Turkish Natural Language
        ↓
Small Fine-Tuned Language Model
        ↓
Strict Query JSON
        ↓
Schema Validation
```

Later, once the employer provides the real database schema and required JSON contract, the system will be customized.

The future production architecture is expected to become:

```text
User Prompt
    ↓
Text Normalization
    ↓
Fine-Tuned Small Language Model
    ↓
Structured Query JSON
    ↓
Schema Validation
    ↓
Entity Resolution
    ↓
Deterministic SQL Builder
    ↓
Database
```

The LLM must **not** be responsible for directly generating arbitrary SQL in the first version.

The model's task is only:

> Convert a Turkish natural-language request into a predefined structured JSON object.

---

# 2. Core Design Principles

The implementation must follow these principles.

## 2.1 Smallest Model Possible

The main objective is not maximum general-purpose intelligence.

The objective is:

- minimum model size
- minimum inference latency
- minimum VRAM/RAM consumption
- minimum prompt token usage
- high structured-output accuracy
- deterministic behavior

Start with a model around the **0.5B–1.5B parameter range**.

Only move to larger models if measurable evaluation results show that smaller models are insufficient.

---

## 2.2 Structured Extraction Instead of SQL Generation

Do not fine-tune the model to generate arbitrary SQL.

Bad architecture:

```text
User Prompt
    ↓
LLM
    ↓
SELECT ...
```

Preferred architecture:

```text
User Prompt
    ↓
LLM
    ↓
Structured JSON
    ↓
Deterministic SQL Builder
```

This significantly reduces:

- hallucination risk
- SQL syntax errors
- model complexity
- prompt size
- security risks
- database-specific coupling

---

# 3. V1 JSON Contract

Until the employer provides the final JSON schema, use the following intermediate representation.

Example:

```json
{
  "operation": "count",
  "target": "sales",
  "filters": [
    {
      "field": "company",
      "operator": "eq",
      "value": "X"
    },
    {
      "field": "product",
      "operator": "eq",
      "value": "Y"
    },
    {
      "field": "date",
      "operator": "lte",
      "value": "2026-08-31"
    }
  ],
  "group_by": [],
  "order_by": null,
  "limit": null
}
```

The schema must be implemented using Pydantic.

---

# 4. Allowed Vocabulary

The output space must remain deliberately restricted.

## 4.1 Operations

Initially support:

```text
select
count
sum
average
min
max
```

Represent these as an Enum.

Example:

```python
class QueryOperation(str, Enum):
    SELECT = "select"
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
```

---

## 4.2 Operators

Initially support:

```text
eq
neq
gt
gte
lt
lte
between
in
contains
```

Do not let the model invent alternate representations such as:

```text
greater_than
equals
=
>
after
before
```

Natural-language variation must always normalize into the canonical operators above.

---

# 5. Initial Semantic Fields

The synthetic V1 dataset can use a generic business/sales domain.

Example fields:

```text
company
product
customer
category
city
region
store
salesperson
date
quantity
sales_amount
unit_price
status
```

Example targets:

```text
sales
sales_amount
quantity
customer
product
order
```

These are temporary.

Later they will be replaced or mapped according to the employer's actual database schema.

---

# 6. Pydantic Models

Create strongly typed Pydantic models.

Recommended structure:

```python
from enum import Enum
from typing import Any
from pydantic import BaseModel


class QueryOperation(str, Enum):
    SELECT = "select"
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"


class FilterOperator(str, Enum):
    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    BETWEEN = "between"
    IN = "in"
    CONTAINS = "contains"


class FilterCondition(BaseModel):
    field: str
    operator: FilterOperator
    value: Any


class OrderDirection(str, Enum):
    ASC = "asc"
    DESC = "desc"


class OrderBy(BaseModel):
    field: str
    direction: OrderDirection


class QueryRequest(BaseModel):
    operation: QueryOperation
    target: str
    filters: list[FilterCondition] = []
    group_by: list[str] = []
    order_by: OrderBy | None = None
    limit: int | None = None
```

Do not over-engineer this schema.

It is intentionally a V1 abstraction.

---

# 7. Example Transformations

## Example 1

Input:

```text
X firmasındaki satışların sayısını getir.
```

Output:

```json
{
  "operation": "count",
  "target": "sales",
  "filters": [
    {
      "field": "company",
      "operator": "eq",
      "value": "X"
    }
  ],
  "group_by": [],
  "order_by": null,
  "limit": null
}
```

---

## Example 2

Input:

```text
ABC firmasının Y ürününün toplam satış tutarını göster.
```

Output:

```json
{
  "operation": "sum",
  "target": "sales_amount",
  "filters": [
    {
      "field": "company",
      "operator": "eq",
      "value": "ABC"
    },
    {
      "field": "product",
      "operator": "eq",
      "value": "Y"
    }
  ],
  "group_by": [],
  "order_by": null,
  "limit": null
}
```

---

## Example 3

Input:

```text
ABC firmasının 1 Ocak 2026 ile 30 Haziran 2026 arasındaki toplam satış tutarını getir.
```

Output:

```json
{
  "operation": "sum",
  "target": "sales_amount",
  "filters": [
    {
      "field": "company",
      "operator": "eq",
      "value": "ABC"
    },
    {
      "field": "date",
      "operator": "between",
      "value": [
        "2026-01-01",
        "2026-06-30"
      ]
    }
  ],
  "group_by": [],
  "order_by": null,
  "limit": null
}
```

---

## Example 4

Input:

```text
ABC firmasının en fazla satış yaptığı ilk 5 ürünü getir.
```

Output:

```json
{
  "operation": "sum",
  "target": "sales",
  "filters": [
    {
      "field": "company",
      "operator": "eq",
      "value": "ABC"
    }
  ],
  "group_by": [
    "product"
  ],
  "order_by": {
    "field": "sales",
    "direction": "desc"
  },
  "limit": 5
}
```

---

# 8. Repository Architecture

Create the repository approximately as follows:

```text
text-to-sql-json/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── app/
│   ├── __init__.py
│   │
│   ├── main.py
│   │
│   ├── config.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── enums.py
│   │   └── query.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   ├── extractor.py
│   │   └── prompt.py
│   │
│   ├── normalization/
│   │   ├── __init__.py
│   │   ├── text.py
│   │   └── dates.py
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   └── output_validator.py
│   │
│   └── api/
│       ├── __init__.py
│       └── routes.py
│
├── training/
│   ├── train.py
│   ├── config.py
│   ├── dataset.py
│   ├── formatting.py
│   └── evaluate.py
│
├── datasets/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
│
├── scripts/
│   ├── validate_dataset.py
│   ├── run_inference.py
│   └── benchmark.py
│
├── tests/
│   ├── test_schema.py
│   ├── test_validation.py
│   ├── test_normalization.py
│   └── test_inference.py
│
└── models/
    └── .gitkeep
```

Keep responsibilities separated.

Do not put training logic inside API files.

Do not put prompt formatting, Pydantic validation, and model loading in one monolithic file.

---

# 9. Dataset Format

Use JSONL.

Each line should contain one independent supervised example.

Recommended raw format:

```json
{
  "input": "X firmasındaki Y ürününün satış sayısını getir.",
  "output": {
    "operation": "count",
    "target": "sales",
    "filters": [
      {
        "field": "company",
        "operator": "eq",
        "value": "X"
      },
      {
        "field": "product",
        "operator": "eq",
        "value": "Y"
      }
    ],
    "group_by": [],
    "order_by": null,
    "limit": null
  }
}
```

Store JSON in canonical deterministic form when formatting for training.

For example:

```python
json.dumps(
    output,
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=False,
)
```

The model should see the same key ordering throughout the dataset.

Recommended ordering:

```text
operation
target
filters
group_by
order_by
limit
```

Consistency matters.

---

# 10. Dataset Split

Initial target:

```text
Total examples: approximately 1000

Train:       700
Validation:  150
Test:        150
```

Do not randomly duplicate paraphrases across train and test.

Semantically near-identical generated variants should remain within one split when possible.

Avoid data leakage.

---

# 11. Dataset Complexity Distribution

Dataset should contain multiple complexity classes.

Suggested distribution:

```text
L1: simple single-filter queries                 20%
L2: two or three filters                        25%
L3: date/date-range queries                     20%
L4: aggregation and grouping                    15%
L5: ordering and limit                          10%
L6: mixed complex combinations                  10%
```

Examples must include:

- formal Turkish
- conversational Turkish
- short prompts
- verbose prompts
- polite requests
- different word orders
- minor spelling variation
- optional filler phrases

Example paraphrases:

```text
X firmasının satış sayısını getir.
X firmasına ait kaç satış var?
X şirketinin toplam satış adedini göster.
X firmasında kaç satış gerçekleşmiş?
Mümkünse X firmasının satış sayısını getirir misin?
```

All semantically equivalent prompts should map to the same canonical JSON structure.

---

# 12. Training Strategy

Use supervised fine-tuning.

Preferred initial strategy:

```text
Base instruct model
    ↓
LoRA / QLoRA
    ↓
Task-specific adapter
```

Do not perform full fine-tuning unless later benchmarks justify it.

Recommended libraries:

```text
transformers
datasets
peft
trl
accelerate
bitsandbytes   # only where supported
torch
pydantic
fastapi
uvicorn
```

The code should detect available hardware.

Support at least:

```text
CUDA
MPS
CPU fallback
```

Do not hardcode CUDA-only assumptions.

---

# 13. Model Selection Strategy

Do not lock the architecture to one specific model.

The implementation should accept a model name/path through configuration.

Example:

```bash
BASE_MODEL=<huggingface-model-id>
```

The system should allow us to benchmark multiple small instruct models later.

Priority:

```text
0.5B class
    ↓
1B–1.5B class
    ↓
3B only if required
```

The final model choice will be made based on measured performance, not brand preference.

---

# 14. Training Prompt Format

Keep training prompts extremely small.

Avoid large system prompts.

One possible format:

```text
### Input
X firmasındaki Y ürününün satış sayısını getir.

### Output
{"operation":"count","target":"sales","filters":[{"field":"company","operator":"eq","value":"X"},{"field":"product","operator":"eq","value":"Y"}],"group_by":[],"order_by":null,"limit":null}
```

If the selected instruct model requires a chat template, use the tokenizer's native chat template.

Do not manually imitate ChatML unless required by that model.

The training pipeline should have a formatter abstraction so prompt formatting can easily change.

---

# 15. Inference Requirements

Inference must prioritize deterministic structured output.

Recommended generation configuration:

```python
do_sample = False
temperature = None
top_p = None
num_beams = 1
```

Do not use:

```python
temperature = 0.7
do_sample = True
```

for production extraction.

Use a low maximum number of generated tokens.

For the V1 schema:

```text
max_new_tokens ≈ 128–256
```

should usually be sufficient.

Measure this rather than assuming.

---

# 16. Output Parsing

The extractor must:

1. receive Turkish text
2. format the inference prompt
3. call the model
4. extract model output
5. isolate JSON
6. parse using `json.loads`
7. validate with Pydantic
8. return valid structured output or a controlled error

Never silently accept malformed JSON.

Example result object:

```json
{
  "success": true,
  "data": {
    "operation": "count",
    "target": "sales",
    "filters": [],
    "group_by": [],
    "order_by": null,
    "limit": null
  },
  "error": null
}
```

On failure:

```json
{
  "success": false,
  "data": null,
  "error": {
    "type": "VALIDATION_ERROR",
    "message": "Model output does not match QueryRequest schema."
  }
}
```

---

# 17. Retry Policy

For V1, implement at most one optional repair attempt.

However, do not immediately rely on a large repair prompt.

Preferred flow:

```text
Model Output
    ↓
Valid JSON?
 ┌──┴──┐
Yes    No
 │      │
return  optional minimal retry
```

Benchmark both:

- zero retry
- one retry

The key metric is first-pass correctness.

A system that only works through repeated retries is not considered successful.

---

# 18. Text Normalization

Do not aggressively modify user input.

Allowed lightweight preprocessing:

```text
trim whitespace
collapse repeated spaces
normalize obvious Unicode whitespace
optional Turkish date normalization helpers
```

Do not remove words merely because they look unnecessary.

Semantic interpretation belongs to the model.

Date normalization can initially remain in the dataset/model output.

Later, date normalization may be moved to deterministic code if useful.

---

# 19. REST API

Create a minimal FastAPI service.

Required endpoint:

```http
POST /extract
```

Request:

```json
{
  "text": "ABC firmasının en çok satış yaptığı 5 ürünü getir."
}
```

Response:

```json
{
  "success": true,
  "query": {
    "operation": "sum",
    "target": "sales",
    "filters": [
      {
        "field": "company",
        "operator": "eq",
        "value": "ABC"
      }
    ],
    "group_by": [
      "product"
    ],
    "order_by": {
      "field": "sales",
      "direction": "desc"
    },
    "limit": 5
  }
}
```

Also implement:

```http
GET /health
```

Example:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

# 20. CLI Inference

Before API testing, support local CLI inference.

Example:

```bash
python scripts/run_inference.py \
  --text "X firmasındaki Y ürününün satış sayısını getir."
```

Output only the extraction result and basic latency information.

Example:

```text
Input:
X firmasındaki Y ürününün satış sayısını getir.

Output:
{"operation":"count","target":"sales",...}

Valid: true
Latency: 82 ms
```

---

# 21. Dataset Validation Script

Create:

```text
scripts/validate_dataset.py
```

It must check:

- every line is valid JSON
- `input` exists
- `output` exists
- output conforms to `QueryRequest`
- no empty input
- no duplicate exact inputs
- allowed enums only
- `between` has exactly two values
- `in` receives a list
- limit is positive when present
- order direction is valid
- basic train/validation/test duplicate detection

Exit with non-zero status if validation fails.

---

# 22. Evaluation Metrics

Evaluation must be stricter than ordinary language-model evaluation.

Do not use BLEU or ROUGE as the primary metrics.

Required metrics:

## 22.1 Valid JSON Rate

```text
valid_json_outputs / total_outputs
```

Target should eventually approach:

```text
~100%
```

---

## 22.2 Schema Validity Rate

Percentage of outputs accepted by the Pydantic model.

---

## 22.3 Exact Match

Canonicalize expected and predicted JSON.

Then compare complete structures.

```python
prediction == expected
```

This is one of the most important metrics.

---

## 22.4 Field-Level Accuracy

Measure separately:

```text
operation accuracy
target accuracy
filter field accuracy
filter operator accuracy
filter value accuracy
group_by accuracy
order_by accuracy
limit accuracy
```

---

## 22.5 Semantic Match

Implement a structure-aware comparison.

For example, filters may be semantically identical despite order:

```json
[
  {"field":"company", ...},
  {"field":"product", ...}
]
```

versus:

```json
[
  {"field":"product", ...},
  {"field":"company", ...}
]
```

Therefore calculate both:

```text
Strict Exact Match
Semantic Structural Match
```

Do not replace strict exact match with semantic match.

Keep both.

---

# 23. Performance Benchmark

Create:

```text
scripts/benchmark.py
```

Measure:

- model load time
- average inference latency
- median inference latency
- p95 inference latency
- prompt token count
- generated token count
- peak memory where practical
- valid JSON rate
- exact match
- semantic match

Benchmark on the held-out test set.

Write results to:

```text
benchmarks/
```

Example:

```text
benchmarks/model_name_YYYYMMDD.json
```

---

# 24. Test Suite

Use `pytest`.

At minimum create tests for:

## Schema

```text
valid QueryRequest
invalid operation
invalid operator
negative limit
invalid order direction
```

## Validation

```text
valid model JSON
malformed JSON
extra surrounding text
missing required fields
wrong filter format
```

## Dataset

```text
duplicate input detection
invalid JSONL
schema mismatch
split leakage
```

## Inference

Mock the model where possible.

Do not make all unit tests require loading a real LLM.

Keep one optional integration test for actual model inference.

---

# 25. Configuration

Create a centralized config module.

Environment variables can include:

```text
BASE_MODEL
ADAPTER_PATH
MAX_NEW_TOKENS
DEVICE
MODEL_DTYPE
HOST
PORT
```

Example `.env.example`:

```env
BASE_MODEL=
ADAPTER_PATH=./models/latest
MAX_NEW_TOKENS=192
DEVICE=auto
MODEL_DTYPE=auto
HOST=127.0.0.1
PORT=8000
```

Never commit secrets.

No API key should be required for local inference after the model is downloaded.

---

# 26. Logging

Keep logging lightweight.

Log:

```text
startup
model loading
model path
device
inference latency
validation failures
unexpected exceptions
```

Do not log sensitive production prompt contents by default.

During local development, prompt logging may be enabled explicitly through configuration.

---

# 27. README Requirements

README must contain:

- project purpose
- architecture overview
- installation
- virtual environment setup
- dataset format
- dataset validation
- training command
- inference command
- API command
- test command
- benchmark command
- current limitations

Example local commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

macOS/Linux activation should be documented.

Optional Windows PowerShell equivalent:

```powershell
.venv\Scripts\Activate.ps1
```

---

# 28. Initial Development Order

Implement in the following sequence.

## Phase 1 — Skeleton

Create:

```text
repository structure
requirements
config
Pydantic schema
basic tests
```

Do not start training yet.

---

## Phase 2 — Dataset Infrastructure

Implement:

```text
dataset loader
dataset validator
training formatter
split validation
```

The actual dataset files will be supplied separately.

---

## Phase 3 — Model Infrastructure

Implement:

```text
model loader
device selection
LoRA/QLoRA training support
training script
checkpoint output
```

The training script must allow model name and hyperparameters to be changed without editing source code.

---

## Phase 4 — Inference

Implement:

```text
prompt formatter
local model inference
JSON parser
Pydantic validator
CLI
```

---

## Phase 5 — Evaluation

Implement:

```text
exact match
semantic match
field-level metrics
latency benchmark
test-set evaluation
```

---

## Phase 6 — API

Only after local inference works:

```text
FastAPI
/extract
/health
```

---

## Phase 7 — Optimization

After obtaining baseline results:

```text
reduce prompt size
reduce output length
compare models
adjust LoRA configuration
quantize if necessary
measure latency
measure memory
```

Optimization must be driven by benchmark data.

---

# 29. Training Script Requirements

The training script should support parameters such as:

```bash
python training/train.py \
  --model_name <MODEL> \
  --train_file datasets/train.jsonl \
  --validation_file datasets/validation.jsonl \
  --output_dir models/run_001 \
  --epochs 3 \
  --learning_rate 2e-4 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05
```

These are example defaults, not mandatory optimal values.

Do not assume the values above are optimal.

Make them configurable.

Use checkpointing.

Save:

```text
adapter
tokenizer metadata
training config
evaluation summary
```

---

# 30. Hardware Compatibility

The developer may run this project locally on different hardware.

Implement device detection:

```python
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

However, remember that some quantization libraries do not support MPS.

Therefore:

- keep quantization optional
- fail with a clear error
- do not assume bitsandbytes works everywhere

---

# 31. Determinism

Where possible:

```python
import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```

For inference:

```text
sampling disabled
single beam
fixed prompt format
fixed JSON key order
strict validation
```

Important:

Deterministic generation settings reduce randomness but do not guarantee mathematically identical behavior across every hardware/backend implementation.

The system-level contract should therefore depend on validation and constrained output design rather than only `temperature=0`.

---

# 32. Error Handling

Create explicit application errors.

Examples:

```text
MODEL_LOAD_ERROR
MODEL_INFERENCE_ERROR
JSON_PARSE_ERROR
VALIDATION_ERROR
INVALID_INPUT
```

API responses should not expose Python tracebacks.

Tracebacks belong in developer logs.

---

# 33. Security Constraints

Even though V1 does not execute SQL, design with future security in mind.

Future SQL generation must never directly concatenate model values into SQL strings.

Later use:

```text
parameterized queries
allowlisted fields
allowlisted tables
allowlisted operations
deterministic SQL builder
```

Never trust LLM-generated table names or raw SQL.

---

# 34. Things NOT to Build Yet

Do not implement these in V1 unless required for internal abstractions:

- real database integration
- direct SQL execution
- arbitrary text-to-SQL
- vector database
- RAG
- multi-agent architecture
- LangChain
- complex orchestration frameworks
- web UI
- authentication
- cloud deployment
- Kubernetes
- large-model serving infrastructure

The project should stay intentionally small.

---

# 35. Future Production Extension

Once the employer supplies:

```text
real prompt examples
actual database schema
required JSON structure
supported query types
business terminology
allowed filters
date semantics
entity IDs
```

the system will move to V2.

Expected V2 pipeline:

```text
User Prompt
    ↓
Normalizer
    ↓
Fine-Tuned Extractor
    ↓
Employer JSON Contract
    ↓
Pydantic / JSON Schema Validation
    ↓
Entity Resolver
    ↓
Deterministic Query Builder
    ↓
Parameterized SQL
    ↓
Database
```

Possible entity-resolution example:

User:

```text
Coca Cola satışlarını getir.
```

Extractor:

```json
{
  "company": "Coca Cola"
}
```

Resolver:

```text
"Coca Cola"
    ↓
company_id = 187
```

SQL builder uses:

```text
company_id = 187
```

The LLM should not be forced to memorize database primary keys.

---

# 36. Success Criteria for V1

V1 is successful when:

1. the dataset validator passes
2. the small model can be fine-tuned locally
3. CLI inference works
4. model outputs parseable JSON
5. Pydantic validation works
6. evaluation runs on unseen test data
7. exact-match and field-level metrics are reported
8. FastAPI can expose the extractor
9. inference is deterministic enough for repeated structured extraction
10. architecture can later accept the employer's real JSON schema without a full rewrite

The first objective is **correctness and clean architecture**.

Optimization follows after the baseline is measured.

---

# 37. Coding Rules for the Terminal Coding Agent

While implementing this project:

- keep modules small
- use type hints
- use Pydantic models
- write docstrings only where they add value
- avoid unnecessary abstractions
- avoid framework-heavy solutions
- do not introduce LangChain
- do not introduce agents
- do not add a database yet
- do not generate SQL yet unless creating a clearly isolated placeholder for future work
- write tests together with modules
- prefer standard Python and Hugging Face tooling
- maintain a clean dependency list
- do not hardcode one model throughout the source
- expose model selection through configuration
- keep training and inference code separate
- do not silently repair invalid model output
- report validation failures explicitly

After every major phase:

```bash
pytest
```

must pass before moving forward.

---

# 38. Expected First Implementation Deliverables

The coding agent should initially deliver:

```text
1. repository skeleton
2. requirements.txt
3. configuration system
4. Pydantic query schema
5. dataset loading utilities
6. dataset validation script
7. training data formatter
8. LoRA/SFT training script
9. local model loader
10. deterministic inference extractor
11. JSON validation layer
12. CLI inference script
13. evaluation module
14. benchmark script
15. FastAPI endpoints
16. pytest test suite
17. README
```

The agent must not create fake training results.

The dataset will be provided separately after the code skeleton is ready.

---

# 39. Immediate Task

Start by implementing **Phase 1 and Phase 2 only**.

That means:

```text
- initialize repository structure
- create config
- implement schemas/enums.py
- implement schemas/query.py
- implement dataset.py
- implement validate_dataset.py
- implement formatting.py
- create initial tests
- create README skeleton
- create requirements.txt
```

Do not start model training until the dataset is provided.

Once Phase 1 and Phase 2 are complete:

1. run tests
2. show the final repository tree
3. report all implemented files
4. report any assumptions
5. stop and wait for the dataset

This prevents the implementation from moving ahead with incorrect assumptions.

---

# 40. Summary

The system is a specialized Turkish semantic query extractor.

Its purpose is not to be a general chatbot.

Its purpose is:

```text
Natural Turkish Request
        ↓
Small Fine-Tuned Model
        ↓
Strict JSON Query Representation
```

The project must optimize for:

```text
small model
few tokens
low latency
low memory
strict schema
deterministic output
high exact-match accuracy
easy future customization
```

The final production database and employer-specific JSON schema will be integrated only after the V1 extraction system has been proven to work.
