"""Meridian Retail support-draft review tool.

Usage:
    python app.py --case C1
    python app.py --case C2 --simulate-timeout
    python app.py --case C3 --offline-demo

Drafts a customer support reply for one case and prints it for a human
reviewer. This tool only prints text. It never sends a message, approves a
refund, or changes a record -- a human must act on the draft.

The model is instructed by the "context pack" in this folder rather than a
hardcoded prompt: task.md (what to do), definitions.md (term meanings),
contractDefinitions.md (the required output shape), policy.md (the business
rules), and evidence_map.json (which source_ids exist for each case).
task.md's Evidence section is written for one case; it's regenerated per
selected --case from evidence_map.json rather than edited by hand.

Model backend is chosen via APP_MODEL_BACKEND ("ollama", the default, or
"openrouter"). See .env.example for the variables each backend needs.
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from ollama import Client       # local model backend
from openai import OpenAI       # OpenRouter backend (OpenAI-compatible API)

# Project files are always found relative to this file, not the caller's
# working directory, so the app can be run from anywhere.
BASE_DIR = Path(__file__).resolve().parent
POLICY_PATH = BASE_DIR / "policy.md"
CASES_PATH = BASE_DIR / "cases.json"
DEMO_OUTPUTS_PATH = BASE_DIR / "demo_outputs.json"
TASK_PATH = BASE_DIR / "task.md"
DEFINITIONS_PATH = BASE_DIR / "definitions.md"
CONTRACT_DEFINITIONS_PATH = BASE_DIR / "contractDefinitions.md"
EVIDENCE_MAP_PATH = BASE_DIR / "evidence_map.json"

# --- backend configuration defaults -----------------------------------------
DEFAULT_BACKEND = "ollama"
VALID_BACKENDS = ("ollama", "openrouter")
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

VALID_CASES = ("C1", "C2", "C3")
# The only three statuses a real model response is allowed to use (per
# contractDefinitions.md's closed set). UNAVAILABLE and REJECTED_OUT_OF_POLICY
# are separate, app-generated statuses -- the model itself never returns them.
VALID_REVIEW_STATUSES = {"READY_FOR_HUMAN_REVIEW", "NEEDS_INFORMATION", "BLOCKED"}
UNAVAILABLE_STATUS = "UNAVAILABLE"
REJECTED_STATUS = "REJECTED_OUT_OF_POLICY"
# Refund-eligibility window: a deterministic, code-enforced check, separate
# from policy.md's "two days" wording (kept as-is; not reconciled with this).
REFUND_WINDOW_HOURS = 78

# The output contract's ten fields, per contractDefinitions.md. The model's
# JSON response must contain exactly these keys, no more, no less.
REQUIRED_RESULT_KEYS = {
    "case_id",
    "case_summary",
    "known_facts",
    "evidence_refs",
    "evidence_links",
    "missing_information",
    "conflicting_information",
    "draft_reply",
    "review_status",
    "human_action_required",
}

# Shown whenever the app can't produce a real draft (config error, provider
# error, timeout, bad output) so the human reviewer knows what to do next.
MANUAL_FALLBACK_MESSAGE = (
    "Model output unavailable. A human agent must draft this reply manually "
    "using the SUPPLIED FACTS, SUPPLIED TASK and SUPPLIED POLICY above "
    "before responding to the customer. No message has been sent, no "
    "refund has been approved, and no record has been changed."
)

# Load .env once at import time so every env var lookup below just works.
load_dotenv(BASE_DIR / ".env")


class ModelOutputError(Exception):
    """Raised when the model's response is not valid per the required schema."""


def load_policy(path=POLICY_PATH):
    # Read the policy markdown as plain text; it's injected into the prompt as-is.
    with open(path, "r") as f:
        return f.read()


def load_cases(path=CASES_PATH):
    # cases.json is a single object keyed by case id (C1, C2, C3).
    with open(path, "r") as f:
        return json.load(f)


def load_task(path=TASK_PATH):
    with open(path, "r") as f:
        return f.read()


def load_definitions(path=DEFINITIONS_PATH):
    with open(path, "r") as f:
        return f.read()


def load_contract_definitions(path=CONTRACT_DEFINITIONS_PATH):
    with open(path, "r") as f:
        return f.read()


def load_evidence_map(path=EVIDENCE_MAP_PATH):
    with open(path, "r") as f:
        return json.load(f)


def load_context_pack():
    """Loads every file in the context pack: the task instructions, term and
    output-contract definitions, the policy, and the evidence map. This is
    the single place that reads "everything in the folder" the model and the
    deterministic short-circuits are allowed to draw on."""
    return {
        "task": load_task(),
        "definitions": load_definitions(),
        "contract_definitions": load_contract_definitions(),
        "policy": load_policy(),
        "evidence_map": load_evidence_map(),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Draft a Meridian Retail support reply for human review.")
    parser.add_argument("--case", required=True, choices=VALID_CASES)  # which case to draft for
    mode_group = parser.add_mutually_exclusive_group()  # at most one of these two special modes
    mode_group.add_argument("--simulate-timeout", action="store_true")  # force the safe-unavailable path, no API call
    mode_group.add_argument("--offline-demo", action="store_true")      # replay a prerecorded result, no API call
    return parser.parse_args(argv)


def case_source_ids(case_id, evidence_map):
    # The source_ids task.md's Evidence section is allowed to name for this
    # case, taken straight from evidence_map.json (its insertion order).
    return list(evidence_map[case_id].keys())


def field_source_map(case_id, evidence_map):
    # Reverse-indexes evidence_map.json's per-case entries -- e.g.
    # "order_record:MR-1042": "cases.json#C1 (order_id, product, delivery_date)"
    # -- into field -> source_id, so every case fact can be paired with the
    # source_id that covers it (parsed from the parenthesised field list).
    mapping = {}
    for source_id, description in evidence_map[case_id].items():
        if "(" in description and description.endswith(")"):
            fields_part = description[description.index("(") + 1 : -1]
            for field in fields_part.split(","):
                mapping[field.strip()] = source_id
    return mapping


def build_known_facts(case, case_id, evidence_map, exclude=()):
    # Every present, non-null case field paired with the source_id that
    # covers it. Skips a field with no mapped source rather than inventing
    # one, and skips anything named in `exclude` (e.g. a field known to be
    # missing, which by definition isn't a "known fact").
    sources = field_source_map(case_id, evidence_map)
    facts = []
    for field, value in case.items():
        if field in exclude or value is None:
            continue
        source_id = sources.get(field)
        if source_id is None:
            continue
        facts.append({"field": field, "value": value, "source_id": source_id})
    return facts


# Matches task.md's "## Evidence\n\n...\n\n## Definitions" block so its
# content can be regenerated per case without touching the rest of the file.
_EVIDENCE_SECTION_PATTERN = re.compile(r"(## Evidence\n\n).*?(\n\n## Definitions)", re.DOTALL)


def build_task_for_case(task_text, case_id, evidence_map):
    # task.md is written with one case's Evidence section hardcoded. Rather
    # than trust the model to generalize it, or hand-maintain one task.md per
    # case, regenerate just that section from evidence_map.json for whichever
    # --case was selected, and leave every other section verbatim.
    source_ids = case_source_ids(case_id, evidence_map)
    quoted = [f"`{sid}`" for sid in source_ids]
    evidence_line = ", ".join(quoted[:-1]) + f" and {quoted[-1]} only." if len(quoted) > 1 else f"{quoted[0]} only."
    replacement = f"## Evidence\n\n{evidence_line}\n\n## Definitions"
    new_text, count = _EVIDENCE_SECTION_PATTERN.subn(replacement, task_text)
    if count != 1:
        raise ValueError("task.md's Evidence section could not be found/replaced")
    return new_text


def build_system_prompt(case_id, context):
    # Assembles the full context pack into the model's instructions: the
    # (per-case) task, term definitions, the required output shape, and the
    # policy -- in that order, so the task's own instructions lead.
    task_text = build_task_for_case(context["task"], case_id, context["evidence_map"])
    return (
        f"{task_text}\n\n"
        "DEFINITIONS:\n"
        f"{context['definitions']}\n\n"
        "OUTPUT CONTRACT:\n"
        f"{context['contract_definitions']}\n\n"
        "SUPPLIED POLICY:\n"
        f"{context['policy']}\n\n"
        "Respond with only a single JSON object matching the output "
        "contract exactly, and no other text."
    )


def build_messages(case, case_id, evidence_map):
    # Flattens the case dict into plain text, with each field labelled with
    # the source_id that covers it -- so the model doesn't have to infer
    # which source_id backs a fact, it's handed the pairing directly.
    sources = field_source_map(case_id, evidence_map)
    lines = []
    for field, value in case.items():
        source_id = sources.get(field)
        suffix = f"  [source_id: {source_id}]" if source_id else ""
        lines.append(f"{field}: {value}{suffix}")
    user_content = (
        "SUPPLIED FACTS:\n"
        + "\n".join(lines)
        + "\n\nDraft the reply now, following the task and policy exactly."
    )
    return [{"role": "user", "content": user_content}]


def get_backend():
    # APP_MODEL_BACKEND picks which client/call path the rest of the app uses.
    backend = os.environ.get("APP_MODEL_BACKEND", DEFAULT_BACKEND).strip().lower()
    if backend not in VALID_BACKENDS:
        raise RuntimeError(f"APP_MODEL_BACKEND must be one of {VALID_BACKENDS}, got '{backend}'")
    return backend


def get_ollama_host():
    return os.environ.get("APP_OLLAMA_HOST", DEFAULT_OLLAMA_HOST)


def get_client():
    # Returns the right SDK client object for whichever backend is configured.
    if get_backend() == "openrouter":
        api_key = os.environ.get("APP_OPENROUTER_API_KEY", "").strip()
        if not api_key:
            # Caught by resolve_result() and turned into a safe unavailable
            # result — never raised all the way up to the user as a traceback.
            raise RuntimeError("APP_OPENROUTER_API_KEY is not set.")
        # OpenRouter speaks the OpenAI API shape, so the OpenAI SDK just
        # needs pointing at OpenRouter's base URL instead of api.openai.com.
        return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, timeout=20.0, max_retries=0)
    # Ollama needs no credential — just where the local server is listening.
    return Client(host=get_ollama_host(), timeout=20.0)


def get_model_id():
    # Which model name to send to the client, per backend.
    if get_backend() == "openrouter":
        model_id = os.environ.get("APP_OPENROUTER_MODEL", "").strip()
        if not model_id:
            # No sensible default here: OpenRouter's free models rotate over
            # time, so a hardcoded default would eventually go stale/break.
            raise RuntimeError("APP_OPENROUTER_MODEL is not set.")
        return model_id
    return os.environ.get("APP_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def _call_ollama(client, model_id, case, context):
    case_id = case["case_id"]
    messages = [{"role": "system", "content": build_system_prompt(case_id, context)}]
    messages.extend(build_messages(case, case_id, context["evidence_map"]))
    # format="json" asks Ollama to constrain output to syntactically valid JSON
    # (it does NOT guarantee the *content* is correct — see parse_and_validate).
    response = client.chat(model=model_id, messages=messages, format="json")
    return response["message"]["content"]


def _call_openrouter(client, model_id, case, context):
    case_id = case["case_id"]
    messages = [{"role": "system", "content": build_system_prompt(case_id, context)}]
    messages.extend(build_messages(case, case_id, context["evidence_map"]))
    response = client.chat.completions.create(
        model=model_id,
        # Generous budget: the output contract is now ten fields (several
        # nested lists), and some free OpenRouter models are "reasoning"
        # models that spend most of their tokens on a hidden chain of
        # thought before writing the final JSON answer. A tight limit
        # truncates them before they ever emit content (see finish_reason
        # "length" -> content=None).
        max_tokens=3000,
        response_format={"type": "json_object"},  # OpenAI-style JSON mode
        messages=messages,
    )
    return response.choices[0].message.content


def call_model(client, model_id, case, context):
    # Single entry point get_online_result() calls; picks the right
    # backend-specific implementation based on current configuration.
    if get_backend() == "openrouter":
        return _call_openrouter(client, model_id, case, context)
    return _call_ollama(client, model_id, case, context)


def parse_and_validate_model_output(raw_text, case_id, evidence_map):
    # Structural/type validation only — this checks the JSON is *shaped*
    # correctly, not that its content is actually correct (a model can pass
    # every check here and still, say, claim a supplied field is missing).
    try:
        data = json.loads(raw_text)
    except (TypeError, ValueError) as exc:
        raise ModelOutputError(f"response was not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ModelOutputError("response JSON was not an object")

    # Exactly these ten keys — no extras, none missing.
    if set(data.keys()) != REQUIRED_RESULT_KEYS:
        raise ModelOutputError(f"response had unexpected keys: {sorted(data.keys())}")

    valid_source_ids = set(evidence_map[case_id].keys())

    if data["case_id"] != case_id:
        raise ModelOutputError(f"case_id must be '{case_id}'")

    if not isinstance(data["case_summary"], str):
        raise ModelOutputError("case_summary must be a string")

    if not isinstance(data["known_facts"], list):
        raise ModelOutputError("known_facts must be a list")
    for item in data["known_facts"]:
        if not isinstance(item, dict) or set(item.keys()) != {"field", "value", "source_id"}:
            raise ModelOutputError("each known_facts item must have exactly field, value, source_id")
        if item["source_id"] not in valid_source_ids:
            raise ModelOutputError(
                f"known_facts source_id '{item['source_id']}' not in evidence_map for {case_id}"
            )

    if not isinstance(data["evidence_refs"], list) or not all(
        isinstance(item, str) for item in data["evidence_refs"]
    ):
        raise ModelOutputError("evidence_refs must be a list of strings")

    if not isinstance(data["evidence_links"], list):
        raise ModelOutputError("evidence_links must be a list")
    for item in data["evidence_links"]:
        if not isinstance(item, dict) or set(item.keys()) != {"claim", "source_id"}:
            raise ModelOutputError("each evidence_links item must have exactly claim, source_id")
        if item["source_id"] not in valid_source_ids:
            raise ModelOutputError(
                f"evidence_links source_id '{item['source_id']}' not in evidence_map for {case_id}"
            )

    if not isinstance(data["missing_information"], list) or not all(
        isinstance(item, str) for item in data["missing_information"]
    ):
        raise ModelOutputError("missing_information must be a list of strings")

    if not isinstance(data["conflicting_information"], list):
        raise ModelOutputError("conflicting_information must be a list")
    for item in data["conflicting_information"]:
        if not isinstance(item, dict) or set(item.keys()) != {"description", "source_ids"}:
            raise ModelOutputError("each conflicting_information item must have exactly description, source_ids")
        if not isinstance(item["source_ids"], list) or not all(
            sid in valid_source_ids for sid in item["source_ids"]
        ):
            raise ModelOutputError("conflicting_information source_ids must all be in evidence_map")

    if not isinstance(data["draft_reply"], str):
        raise ModelOutputError("draft_reply must be a string")

    if data["review_status"] not in VALID_REVIEW_STATUSES:
        raise ModelOutputError(
            f"review_status must be one of {sorted(VALID_REVIEW_STATUSES)}"
        )

    if not isinstance(data["human_action_required"], str):
        raise ModelOutputError("human_action_required must be a string")

    return data


def unavailable_result(case, evidence_map):
    # The one "safe" shape returned whenever a real draft can't be produced,
    # regardless of *why* (bad config, provider error, timeout, bad JSON).
    return {
        "case_id": case["case_id"],
        "case_summary": "",
        "known_facts": [],
        "evidence_refs": [],
        "evidence_links": [],
        "missing_information": [],
        "conflicting_information": [],
        "draft_reply": "",
        "review_status": UNAVAILABLE_STATUS,
        "human_action_required": (
            "A human agent must draft this reply manually; the automated "
            "draft is unavailable."
        ),
    }


def get_online_result(client, model_id, case, context, model_fn=call_model):
    """Calls the model and validates its output. Any provider error, timeout,
    or invalid response degrades to the same safe unavailable result rather
    than raising — this is the single boundary all model failures pass
    through, by design, so a traceback or a stale draft never reaches
    a human reviewer."""
    case_id = case["case_id"]
    try:
        raw_text = model_fn(client, model_id, case, context)  # the only network call
        data = parse_and_validate_model_output(raw_text, case_id, context["evidence_map"])
        return data, None  # (result, error) — no error on success
    except Exception as exc:  # noqa: BLE001 - intentional safety boundary, see docstring
        return unavailable_result(case, context["evidence_map"]), str(exc)


def is_delivery_date_missing(case):
    # Treat both a JSON null and an empty/whitespace-only string as "missing".
    value = case.get("delivery_date")
    return value is None or (isinstance(value, str) and value.strip() == "")


def missing_delivery_date_result(case, evidence_map):
    # Hardcoded per policy P2: if delivery_date is missing, ask for it —
    # this exact text is returned instead of asking the model to draft it.
    case_id = case["case_id"]
    return {
        "case_id": case_id,
        "case_summary": (
            "Delivery date is not on file; the two-day damage-reporting "
            "rule cannot be evaluated yet."
        ),
        "known_facts": build_known_facts(case, case_id, evidence_map, exclude=("delivery_date",)),
        "evidence_refs": ["P2", "P4"],
        "evidence_links": [],
        "missing_information": ["delivery_date"],
        "conflicting_information": [],
        "draft_reply": (
            "Before Meridian Retail can assess whether this case falls "
            "within the two-day damage-reporting rule, please provide the "
            "delivery date. A human support agent will review the request "
            "once that information is available."
        ),
        "review_status": "NEEDS_INFORMATION",
        "human_action_required": (
            "A human agent must obtain the delivery date from the customer "
            "before this case can proceed."
        ),
    }


def is_within_refund_window(case, now=None):
    # Whether delivery_date is within the last REFUND_WINDOW_HOURS of `now`.
    # `now` is injectable (tests pass a fixed value); defaults to the real
    # current time. Assumes delivery_date is present -- call this only after
    # is_delivery_date_missing() has already ruled out the missing case.
    if now is None:
        now = datetime.now()
    delivery_date = datetime.strptime(case["delivery_date"], "%Y-%m-%d")
    elapsed_hours = (now - delivery_date).total_seconds() / 3600
    return elapsed_hours <= REFUND_WINDOW_HOURS


def refund_window_expired_result(case, evidence_map):
    # Hardcoded, deterministic rejection: whether delivery_date is more than
    # REFUND_WINDOW_HOURS in the past is a fact a date subtraction can check,
    # so this skips the model entirely, the same way missing_delivery_date_result
    # does. Wording stays informational (a human still confirms), consistent
    # with the "no invented fact / no action claimed" boundary in task.md and
    # contractDefinitions.md -- the tool never claims to have decided anything.
    case_id = case["case_id"]
    return {
        "case_id": case_id,
        "case_summary": (
            f"Delivery date on file is more than {REFUND_WINDOW_HOURS} hours "
            f"old; the case falls outside the refund review window."
        ),
        "known_facts": build_known_facts(case, case_id, evidence_map),
        "evidence_refs": [],
        "evidence_links": [],
        "missing_information": [],
        "conflicting_information": [],
        "draft_reply": (
            f"Thank you for contacting Meridian Retail. Based on the "
            f"delivery date on file, this request falls outside our "
            f"{REFUND_WINDOW_HOURS}-hour refund review window, so it does "
            f"not qualify for return review under this rule. A human agent "
            f"will confirm this determination before any final decision."
        ),
        "review_status": REJECTED_STATUS,
        "human_action_required": (
            "A human agent must confirm this rejection before it is "
            "communicated to the customer."
        ),
    }


def resolve_result(case, context, model_fn=call_model):
    """Determines the draft result for the online (non-simulated,
    non-offline) path. Returns (result, notes). When delivery_date is
    missing, returns the deterministic P2 result without constructing a
    client or calling the model at all. When delivery_date is present but
    more than REFUND_WINDOW_HOURS old, returns a deterministic rejection,
    likewise without calling the model."""
    evidence_map = context["evidence_map"]

    # Short-circuit before touching the network at all: both of these are
    # facts a plain comparison can check, and the answer is always the same,
    # so don't guess.
    if is_delivery_date_missing(case):
        return missing_delivery_date_result(case, evidence_map), []

    if not is_within_refund_window(case):
        return refund_window_expired_result(case, evidence_map), []

    try:
        client = get_client()      # may raise if the backend isn't configured
        model_id = get_model_id()  # e.g. missing API key or model name
    except RuntimeError as exc:
        return unavailable_result(case, evidence_map), [f"CONFIGURATION ERROR: {exc}", MANUAL_FALLBACK_MESSAGE]

    result, error = get_online_result(client, model_id, case, context, model_fn=model_fn)
    if error:
        return result, [f"MODEL UNAVAILABLE: {error}", MANUAL_FALLBACK_MESSAGE]
    return result, []


def load_offline_result(case_id, path=DEMO_OUTPUTS_PATH):
    # Used by --offline-demo: replays a hand-authored example instead of
    # calling any model, so the demo works with no backend configured at all.
    with open(path, "r") as f:
        data = json.load(f)
    entry = data.get(case_id)
    if entry is None:
        raise KeyError(f"No offline demo result found for {case_id}")
    return {key: entry[key] for key in REQUIRED_RESULT_KEYS}


def print_section(title, body):
    print(f"\n=== {title} ===")
    print(body)


def print_supplied_facts(case):
    # Shows the reviewer exactly what the model was given — nothing hidden.
    lines = [f"{key}: {value}" for key, value in case.items()]
    print_section("SUPPLIED FACTS", "\n".join(lines))


def print_supplied_task(case_id, context):
    # Shows the reviewer the exact (per-case) task instructions the model
    # was given, since task.md now drives the draft rather than a fixed
    # hardcoded prompt.
    task_text = build_task_for_case(context["task"], case_id, context["evidence_map"])
    print_section("SUPPLIED TASK", task_text.rstrip("\n"))


def print_supplied_policy(policy_text):
    print_section("SUPPLIED POLICY", policy_text.rstrip("\n"))


def print_draft_result(result, notes):
    lines = [
        f"case_id: {result['case_id']}",
        f"review_status: {result['review_status']}",
        f"case_summary: {result['case_summary']}",
        f"draft_reply: {result['draft_reply']}",
        f"known_facts: {result['known_facts']}",
        f"evidence_refs: {result['evidence_refs']}",
        f"evidence_links: {result['evidence_links']}",
        f"missing_information: {result['missing_information']}",
        f"conflicting_information: {result['conflicting_information']}",
        f"human_action_required: {result['human_action_required']}",
    ]
    body = "\n".join(lines)
    if notes:
        # Extra context for the reviewer: why the model wasn't called, or
        # what went wrong, plus the manual-fallback instructions if needed.
        body += "\n\n" + "\n".join(notes)
    print_section("DRAFT RESULT FOR HUMAN REVIEW", body)


def main(argv=None):
    args = parse_args(argv)
    context = load_context_pack()
    cases = load_cases(CASES_PATH)
    case = cases[args.case]

    # These three sections always print, regardless of mode or outcome below.
    print_supplied_facts(case)
    print_supplied_task(args.case, context)
    print_supplied_policy(context["policy"])

    notes = []

    if args.simulate_timeout:
        # Forces the same safe-unavailable state a real timeout would produce,
        # without making any network call — useful for demos/tests.
        result = unavailable_result(case, context["evidence_map"])
        notes.append("SIMULATED TIMEOUT: no model API call was made.")
        notes.append(MANUAL_FALLBACK_MESSAGE)

    elif args.offline_demo:
        # Replays a prerecorded result instead of calling a real model.
        result = load_offline_result(args.case)
        notes.append(
            "OFFLINE DEMO: no model API was called. This is a prerecorded "
            "synthetic result loaded from demo_outputs.json."
        )

    else:
        # The real path: deterministic short-circuit, or an actual model call,
        # depending on the case (see resolve_result).
        result, extra_notes = resolve_result(case, context)
        notes.extend(extra_notes)

    print_draft_result(result, notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
