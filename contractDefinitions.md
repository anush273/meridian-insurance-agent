# Output Contract Definitions

The fields a support-draft result must contain, their types, and the behaviour
required of each. Terms such as `source_id`, *known fact*, *evidence link* and
*conflicting information* are defined in [`definitions.md`](definitions.md).

| Field | Type | Required behaviour |
|-------|------|--------------------|
| `case_id` | string | Matches the selected case. |
| `case_summary` | string | Summarises only supplied evidence. |
| `known_facts` | list of objects | Every item has `field`, `value` and a `source_id` from `evidence_map.json`. |
| `evidence_refs` | list of strings | Contains supplied policy IDs such as `P2` and `P4`. |
| `evidence_links` | list of objects | Every item has `claim` and a supplied `source_id`. |
| `missing_information` | list of strings | Contains required fields that are unavailable. |
| `conflicting_information` | list of objects | Preserves the `description` and `source_ids` for differences; empty when none are supplied. |
| `draft_reply` | string | Proposed wording only; does not claim an action occurred. |
| `review_status` | allowed string | One of `READY_FOR_HUMAN_REVIEW`, `NEEDS_INFORMATION` or `BLOCKED`. |
| `human_action_required` | string | Names the next human-owned step without inventing a person. |

## Notes

- Every `source_id` must resolve to an entry in `evidence_map.json`. A claim
  without a supplied source is not an evidence link.
- `review_status` is a closed set. Any other value is a contract violation,
  including a value that reports the tool's own failure.
- `draft_reply` is proposed wording for a person to review. It is never a sent
  message, an approval, or a record of something that happened.
