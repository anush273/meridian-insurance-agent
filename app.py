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
from ollama import Client       # local model backend
from openai import OpenAI       # OpenRouter backend (OpenAI-compatible API)

# Project files are always found relative to this file, not the caller's
# working directory, so the app can be run from anywhere.
BASE_DIR = Path(__file__).resolve().parent
POLICY_PATH = BASE_DIR / "policy.md"
CASES_PATH = BASE_DIR / "cases.json"
DEMO_OUTPUTS_PATH = BASE_DIR / "demo_outputs.json"

# --- backend configuration defaults -----------------------------------------
DEFAULT_BACKEND = "ollama"
VALID_BACKENDS = ("ollama", "openrouter")
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

VALID_CASES = ("C1", "C2", "C3")
# The only three statuses a real model response is allowed to use.
# UNAVAILABLE (below) is a separate, app-generated status for failures.
VALID_REVIEW_STATUSES = {"READY_FOR_HUMAN_REVIEW", "NEEDS_INFORMATION", "BLOCKED"}
UNAVAILABLE_STATUS = "UNAVAILABLE"
# The model's JSON response must contain exactly these keys, no more, no less.
REQUIRED_RESULT_KEYS = {"draft_reply", "evidence_refs", "missing_information", "review_status"}

# Shown whenever the app can't produce a real draft (config error, provider
# error, timeout, bad output) so the human reviewer knows what to do next.
MANUAL_FALLBACK_MESSAGE = (
    "Model output unavailable. A human agent must draft this reply manually "
    "using the SUPPLIED FACTS and SUPPLIED POLICY above before responding to "
    "the customer. No message has been sent, no refund has been approved, "
    "and no record has been changed."
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Draft a Meridian Retail support reply for human review.")
    parser.add_argument("--case", required=True, choices=VALID_CASES)  # which case to draft for
    mode_group = parser.add_mutually_exclusive_group()  # at most one of these two special modes
    mode_group.add_argument("--simulate-timeout", action="store_true")  # force the safe-unavailable path, no API call
    mode_group.add_argument("--offline-demo", action="store_true")      # replay a prerecorded result, no API call
    return parser.parse_args(argv)


def build_system_prompt(policy_text):
    # The system prompt carries the policy text and pins down the exact JSON
    # shape the model must reply with. Told explicitly to use only supplied
    # facts/policy so it can't invent details not present in the case data.
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
    # Flatten the case dict into a plain-text block the model can read;
    # this becomes the "SUPPLIED FACTS" the model is told to rely on.
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


def _call_ollama(client, model_id, policy_text, case):
    messages = [{"role": "system", "content": build_system_prompt(policy_text)}]
    messages.extend(build_messages(case))
    # format="json" asks Ollama to constrain output to syntactically valid JSON
    # (it does NOT guarantee the *content* is correct — see parse_and_validate).
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
        response_format={"type": "json_object"},  # OpenAI-style JSON mode
        messages=messages,
    )
    return response.choices[0].message.content


def call_model(client, model_id, policy_text, case):
    # Single entry point get_online_result() calls; picks the right
    # backend-specific implementation based on current configuration.
    if get_backend() == "openrouter":
        return _call_openrouter(client, model_id, policy_text, case)
    return _call_ollama(client, model_id, policy_text, case)


def parse_and_validate_model_output(raw_text):
    # Structural/type validation only — this checks the JSON is *shaped*
    # correctly, not that its content is actually correct (a model can pass
    # every check here and still, say, claim a supplied field is missing).
    try:
        data = json.loads(raw_text)
    except (TypeError, ValueError) as exc:
        raise ModelOutputError(f"response was not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ModelOutputError("response JSON was not an object")

    # Exactly these four keys — no extras, none missing.
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
    # The one "safe" shape returned whenever a real draft can't be produced,
    # regardless of *why* (bad config, provider error, timeout, bad JSON).
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
        raw_text = model_fn(client, model_id, policy_text, case)  # the only network call
        data = parse_and_validate_model_output(raw_text)
        return data, None  # (result, error) — no error on success
    except Exception as exc:  # noqa: BLE001 - intentional safety boundary, see docstring
        return unavailable_result(), str(exc)


def is_delivery_date_missing(case):
    # Treat both a JSON null and an empty/whitespace-only string as "missing".
    value = case.get("delivery_date")
    return value is None or (isinstance(value, str) and value.strip() == "")


def missing_delivery_date_result():
    # Hardcoded per policy P2: if delivery_date is missing, ask for it —
    # this exact text is returned instead of asking the model to draft it.
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
    # Short-circuit before touching the network at all: this case doesn't
    # need a model call, and the answer is always the same, so don't guess.
    if is_delivery_date_missing(case):
        return missing_delivery_date_result(), []

    try:
        client = get_client()      # may raise if the backend isn't configured
        model_id = get_model_id()  # e.g. missing API key or model name
    except RuntimeError as exc:
        return unavailable_result(), [f"CONFIGURATION ERROR: {exc}", MANUAL_FALLBACK_MESSAGE]

    result, error = get_online_result(client, model_id, policy_text, case, model_fn=model_fn)
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
    # Shows the reviewer exactly what the model was given — nothing hidden.
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
        # Extra context for the reviewer: why the model wasn't called, or
        # what went wrong, plus the manual-fallback instructions if needed.
        body += "\n\n" + "\n".join(notes)
    print_section("DRAFT RESULT FOR HUMAN REVIEW", body)


def main(argv=None):
    args = parse_args(argv)
    policy_text = load_policy(POLICY_PATH)
    cases = load_cases(CASES_PATH)
    case = cases[args.case]

    # These two sections always print, regardless of mode or outcome below.
    print_supplied_facts(case)
    print_supplied_policy(policy_text)

    notes = []

    if args.simulate_timeout:
        # Forces the same safe-unavailable state a real timeout would produce,
        # without making any network call — useful for demos/tests.
        result = unavailable_result()
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
        # depending on whether delivery_date is present (see resolve_result).
        result, extra_notes = resolve_result(case, policy_text)
        notes.extend(extra_notes)

    print_draft_result(result, notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
