# Turkish Real-World SQL-JSON Extraction Dataset V2

Professional synthetic/adversarial dataset for fine-tuning a small Turkish structured-extraction LLM.

## Splits
- train: 12,000
- validation: 2,000
- test: 3,000
- challenge_test: 1,000 (**never train on this**)

## V2 additions
`status`: `valid`, `insufficient_information`, `ambiguous`, `conflicting`.
`clarification`: null for valid queries; a concise clarification/rejection reason for non-valid queries.
For non-valid examples, `operation` and `target` are null.

## Real-world difficulty
Conversational Turkish, typos/ASCII Turkish, filler words, verbose business context, telegraphic prompts, negation, threshold equivalence, multi-value filters, grouping, top-N/bottom-N, user corrections, compound filters, contradictions, underspecified requests, irrelevant requests, and prompt-injection-like distractions.

## Evaluation
Measure valid JSON rate, schema validity, status accuracy, rejection precision/recall, strict exact match, semantic structural match, filter field/operator/value accuracy, group/order/limit accuracy, and challenge-set performance.

This dataset is for architecture validation. Replace/map the synthetic ontology when the employer's actual JSON contract and DB schema arrive.
