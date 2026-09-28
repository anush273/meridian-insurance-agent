# Task

Prepare a structured case review and draft for a human reviewer.

## Evidence

`support_queue:C2`, `order_record:MR-1088`, `issue_record:C2` and
`policy_register:policy.md` only.

## Definitions

Facts remain paired with source IDs; missing information remains explicit;
review states use the supplied meanings. See [`definitions.md`](definitions.md).

## Boundaries

No return approval, refund, customer message, record change or invented fact.

## Output contract

The fields in Appendix E, with the four Class 2 fields preserved.

## Missing-information behaviour

Name `delivery_date`; do not decide inside or outside the two-day condition;
set `NEEDS_INFORMATION`.

## Human-review requirement

A person confirms the delivery date from the agreed source before the dependent
review continues.
