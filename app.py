"""Meridian Retail support-draft review tool.

Usage:
    python app.py --case C1
    python app.py --case C2 --simulate-timeout
    python app.py --case C3 --offline-demo

Drafts a customer support reply for one case, using only the facts in
cases.json and the rules in policy.md, and prints it for a human reviewer.
This tool only prints text. It never sends a message, approves a refund, or
changes a record — a human must act on the draft.

Model backend is chosen via APP_MODEL_BACKEND ("ollama", the default, or
"openrouter"). See .env.example for the variables each backend needs.
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from ollama import Client
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
POLICY_PATH = BASE_DIR / "policy.md"
CASES_PATH = BASE_DIR / "cases.json"
DEMO_OUTPUTS_PATH = BASE_DIR / "demo_outputs.json"

DEFAULT_BACKEND = "ollama"
VALID_BACKENDS = ("ollama", "openrouter")
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
VALID_CASES = ("C1", "C2", "C3")
VALID_REVIEW_STATUSES = {"READY_FOR_HUMAN_REVIEW", "NEEDS_INFORMATION", "BLOCKED"}
UNAVAILABLE_STATUS = "UNAVAILABLE"
REQUIRED_RESULT_KEYS = {"draft_reply", "evidence_refs", "missing_information", "review_status"}

MANUAL_FALLBACK_MESSAGE = (
    "Model output unavailable. A human agent must draft this reply manually "
    "using the SUPPLIED FACTS and SUPPLIED POLICY above before responding to "
    "the customer. No message has been sent, no refund has been approved, "
    "and no record has been changed."
)

load_dotenv(BASE_DIR / ".env")


class ModelOutputError(Exception):
    """Raised when the model's response is not valid per the required schema."""


def load_policy(path=POLICY_PATH):
    with open(path, "r") as f:
        return f.read()


def load_cases(path=CASES_PATH):
    with open(path, "r") as f:
        return json.load(f)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Draft a Meridian Retail support reply for human review.")
    parser.add_argument("--case", required=True, choices=VALID_CASES)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--simulate-timeout", action="store_true")
    mode_group.add_argument("--offline-demo", action="store_true")
    return parser.parse_args(argv)


def build_system_prompt(policy_text):
    return (
        "You are drafting a customer support reply for Meridian Retail. "
        "Use only the supplied facts and the supplied policy below. Do not "
        "use any outside knowledge and do not invent facts.\n\n"
        "SUPPLIED POLICY:\n"
        f"{policy_text}\n\n"
        "Respond with only a single JSON object and no other text, matching "
        "exactly this shape:\n"
        '{"draft_reply": "<string>", '
        '"evidence_refs": ["<policy id such as P1>", "..."], '
        '"missing_information": ["<field name>", "..."], '
        '"review_status": "READY_FOR_HUMAN_REVIEW" | "NEEDS_INFORMATION" | "BLOCKED"}\n\n'
        "review_status must be exactly one of those three values. "
        "evidence_refs must list only policy IDs (P1, P2, P3, P4) that support the draft. "
        "missing_information must list case field names that are missing and needed."
    )


def build_messages(case):
    case_summary = (
        f"case_id: {case['case_id']}\n"
        f"order_id: {case['order_id']}\n"
        f"product: {case['product']}\n"
        f"customer_question: {case['customer_question']}\n"
        f"delivery_date: {case['delivery_date']}\n"
        f"issue_status: {case['issue_status']}\n"
        f"issue_details: {case['issue_details']}\n"
    )
    user_content = (
        "SUPPLIED FACTS:\n"
        f"{case_summary}\n"
        "Draft the reply now, following the policy exactly."
    )
    return [{"role": "user", "content": user_content}]


def get_backend():
    backend = os.environ.get("APP_MODEL_BACKEND", DEFAULT_BACKEND).strip().lower()
    if backend not in VALID_BACKENDS:
        raise RuntimeError(f"APP_MODEL_BACKEND must be one of {VALID_BACKENDS}, got '{backend}'")
    return backend


def get_ollama_host():
    return os.environ.get("APP_OLLAMA_HOST", DEFAULT_OLLAMA_HOST)


def get_client():
    if get_backend() == "openrouter":
        api_key = os.environ.get("APP_OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("APP_OPENROUTER_API_KEY is not set.")
        return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, timeout=20.0, max_retries=0)
    return Client(host=get_ollama_host(), timeout=20.0)


def get_model_id():
    if get_backend() == "openrouter":
        model_id = os.environ.get("APP_OPENROUTER_MODEL", "").strip()
        if not model_id:
            raise RuntimeError("APP_OPENROUTER_MODEL is not set.")
        return model_id
    return os.environ.get("APP_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def _call_ollama(client, model_id, policy_text, case):
    messages = [{"role": "system", "content": build_system_prompt(policy_text)}]
    messages.extend(build_messages(case))
    response = client.chat(model=model_id, messages=messages, format="json")
    return response["message"]["content"]


def _call_openrouter(client, model_id, policy_text, case):
    messages = [{"role": "system", "content": build_system_prompt(policy_text)}]
    messages.extend(build_messages(case))
    response = client.chat.completions.create(
        model=model_id,
        # Generous budget: some free OpenRouter models are "reasoning"
        # models that spend most of their tokens on a hidden chain of
        # thought before writing the final JSON answer. A tight limit
        # truncates them before they ever emit content (see finish_reason
        # "length" -> content=None).
        max_tokens=2000,
        response_format={"type": "json_object"},
        messages=messages,
    )
    return response.choices[0].message.content


def call_model(client, model_id, policy_text, case):
    if get_backend() == "openrouter":
        return _call_openrouter(client, model_id, policy_text, case)
    return _call_ollama(client, model_id, policy_text, case)


def parse_and_validate_model_output(raw_text):
    try:
        data = json.loads(raw_text)
    except (TypeError, ValueError) as exc:
        raise ModelOutputError(f"response was not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ModelOutputError("response JSON was not an object")

    if set(data.keys()) != REQUIRED_RESULT_KEYS:
        raise ModelOutputError(f"response had unexpected keys: {sorted(data.keys())}")

    if not isinstance(data["draft_reply"], str):
        raise ModelOutputError("draft_reply must be a string")

    if not isinstance(data["evidence_refs"], list) or not all(
        isinstance(item, str) for item in data["evidence_refs"]
    ):
        raise ModelOutputError("evidence_refs must be a list of strings")

    if not isinstance(data["missing_information"], list) or not all(
        isinstance(item, str) for item in data["missing_information"]
    ):
        raise ModelOutputError("missing_information must be a list of strings")

    if data["review_status"] not in VALID_REVIEW_STATUSES:
        raise ModelOutputError(
            f"review_status must be one of {sorted(VALID_REVIEW_STATUSES)}"
        )

    return data


def unavailable_result():
    return {
        "draft_reply": "",
        "evidence_refs": [],
        "missing_information": [],
        "review_status": UNAVAILABLE_STATUS,
    }


def get_online_result(client, model_id, policy_text, case, model_fn=call_model):
    """Calls the model and validates its output. Any provider error, timeout,
    or invalid response degrades to the same safe unavailable result rather
    than raising — this is the single boundary all model failures pass
    through, by design, so a traceback or a stale draft never reaches
    a human reviewer."""
    try:
        raw_text = model_fn(client, model_id, policy_text, case)
        data = parse_and_validate_model_output(raw_text)
        return data, None
    except Exception as exc:  # noqa: BLE001 - intentional safety boundary, see docstring
        return unavailable_result(), str(exc)


def is_delivery_date_missing(case):
    value = case.get("delivery_date")
    return value is None or (isinstance(value, str) and value.strip() == "")


def missing_delivery_date_result():
    return {
        "draft_reply": (
            "Before Meridian Retail can assess whether this case falls "
            "within the two-day damage-reporting rule, please provide the "
            "delivery date. A human support agent will review the request "
            "once that information is available."
        ),
        "evidence_refs": ["P2", "P4"],
        "missing_information": ["delivery_date"],
        "review_status": "NEEDS_INFORMATION",
    }


def resolve_result(case, policy_text, model_fn=call_model):
    """Determines the draft result for the online (non-simulated,
    non-offline) path. Returns (result, notes). When delivery_date is
    missing, returns the deterministic P2 result without constructing a
    client or calling the model at all."""
    if is_delivery_date_missing(case):
        return missing_delivery_date_result(), []

    try:
        client = get_client()
        model_id = get_model_id()
    except RuntimeError as exc:
        return unavailable_result(), [f"CONFIGURATION ERROR: {exc}", MANUAL_FALLBACK_MESSAGE]

    result, error = get_online_result(client, model_id, policy_text, case, model_fn=model_fn)
    if error:
        return result, [f"MODEL UNAVAILABLE: {error}", MANUAL_FALLBACK_MESSAGE]
    return result, []


def load_offline_result(case_id, path=DEMO_OUTPUTS_PATH):
    with open(path, "r") as f:
        data = json.load(f)
    entry = data.get(case_id)
    if entry is None:
        raise KeyError(f"No offline demo result found for {case_id}")
    return {
        "draft_reply": entry["draft_reply"],
        "evidence_refs": entry["evidence_refs"],
        "missing_information": entry["missing_information"],
        "review_status": entry["review_status"],
    }


def print_section(title, body):
    print(f"\n=== {title} ===")
    print(body)


def print_supplied_facts(case):
    lines = [f"{key}: {value}" for key, value in case.items()]
    print_section("SUPPLIED FACTS", "\n".join(lines))


def print_supplied_policy(policy_text):
    print_section("SUPPLIED POLICY", policy_text.rstrip("\n"))


def print_draft_result(result, notes):
    lines = [
        f"review_status: {result['review_status']}",
        f"draft_reply: {result['draft_reply']}",
        f"evidence_refs: {result['evidence_refs']}",
        f"missing_information: {result['missing_information']}",
    ]
    body = "\n".join(lines)
    if notes:
        body += "\n\n" + "\n".join(notes)
    print_section("DRAFT RESULT FOR HUMAN REVIEW", body)


def main(argv=None):
    args = parse_args(argv)
    policy_text = load_policy(POLICY_PATH)
    cases = load_cases(CASES_PATH)
    case = cases[args.case]

    print_supplied_facts(case)
    print_supplied_policy(policy_text)

    notes = []

    if args.simulate_timeout:
        result = unavailable_result()
        notes.append("SIMULATED TIMEOUT: no model API call was made.")
        notes.append(MANUAL_FALLBACK_MESSAGE)

    elif args.offline_demo:
        result = load_offline_result(args.case)
        notes.append(
            "OFFLINE DEMO: no model API was called. This is a prerecorded "
            "synthetic result loaded from demo_outputs.json."
        )

    else:
        result, extra_notes = resolve_result(case, policy_text)
        notes.extend(extra_notes)

    print_draft_result(result, notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
