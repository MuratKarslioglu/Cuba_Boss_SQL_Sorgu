# Required V2 Project Changes

1. Add `QueryStatus`: `valid`, `insufficient_information`, `ambiguous`, `conflicting`.
2. Add `clarification: str | None`.
3. Make `operation` and `target` nullable.
4. Validation rule:
   - `valid` => operation + target required; clarification normally null.
   - non-valid => operation + target must be null; clarification required.
5. Update dataset loader to preserve `metadata`.
6. Never train on `challenge_test.jsonl`.
7. Report metrics by `status`, `family`, and `noise`.
8. Add rejection precision/recall and status confusion matrix.
9. Keep deterministic inference (`do_sample=False`).
10. Do not silently convert ambiguous/conflicting inputs into valid queries.
