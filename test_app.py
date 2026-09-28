import json
from datetime import datetime

import pytest
from ollama import Client
from openai import OpenAI

import app


# Real context pack (task.md, definitions.md, contractDefinitions.md,
# policy.md, evidence_map.json), loaded once. These are the actual project
# files, same pattern as using app.POLICY_PATH/app.CASES_PATH directly.
CONTEXT = app.load_context_pack()
EVIDENCE_MAP = CONTEXT["evidence_map"]

# Computed at import time so this stays inside app.REFUND_WINDOW_HOURS no
# matter when the suite runs -- tests that aren't about the refund window
# shouldn't have to think about it.
RECENT_DELIVERY_DATE = datetime.now().strftime("%Y-%m-%d")

VALID_CASE = {
    "case_id": "C1",
    "order_id": "MR-1042",
    "product": "Jacket",
    "customer_question": "q",
    "delivery_date": RECENT_DELIVERY_DATE,
    "issue_status": "open",
    "issue_details": "d",
}

VALID_MODEL_RESULT = {
    "case_id": "C1",
    "case_summary": "Jacket zip broken, reported within two days of delivery.",
    "known_facts": [
        {"field": "order_id", "value": "MR-1042", "source_id": "order_record:MR-1042"},
    ],
    "evidence_refs": ["P1", "P4"],
    "evidence_links": [
        {"claim": "Damage reported within two days can be referred for review.", "source_id": "policy_register:policy.md"},
    ],
    "missing_information": [],
    "conflicting_information": [],
    "draft_reply": "Thanks for reaching out, this is being referred for review.",
    "review_status": "READY_FOR_HUMAN_REVIEW",
    "human_action_required": "A human agent must confirm eligibility.",
}
VALID_MODEL_JSON = json.dumps(VALID_MODEL_RESULT)


# --- input files -------------------------------------------------------

def test_load_policy_reads_file():
    text = app.load_policy(app.POLICY_PATH)
    assert "P1." in text
    assert "P4." in text


def test_load_cases_reads_all_three_cases():
    cases = app.load_cases(app.CASES_PATH)
    assert set(cases.keys()) == {"C1", "C2", "C3"}
    assert cases["C1"]["delivery_date"] == "2026-09-15"
    assert cases["C2"]["delivery_date"] is None
    assert cases["C3"]["issue_status"] == "open"


def test_load_context_pack_has_all_expected_keys():
    context = app.load_context_pack()
    assert set(context.keys()) == {"task", "definitions", "contract_definitions", "policy", "evidence_map"}
    assert "## Evidence" in context["task"]
    assert {"C1", "C2", "C3"} <= set(context["evidence_map"].keys())


def test_demo_outputs_file_has_all_three_cases_with_required_shape():
    with open(app.DEMO_OUTPUTS_PATH) as f:
        data = json.load(f)
    for case_id in ("C1", "C2", "C3"):
        entry = data[case_id]
        assert set(entry.keys()) == app.REQUIRED_RESULT_KEYS
        assert entry["case_id"] == case_id
        assert entry["review_status"] in app.VALID_REVIEW_STATUSES


# --- evidence map / known facts -------------------------------------------

def test_case_source_ids_returns_that_cases_four_ids():
    ids = app.case_source_ids("C1", EVIDENCE_MAP)
    assert set(ids) == {
        "support_queue:C1",
        "order_record:MR-1042",
        "issue_record:C1",
        "policy_register:policy.md",
    }


def test_field_source_map_maps_fields_to_source_ids():
    mapping = app.field_source_map("C1", EVIDENCE_MAP)
    assert mapping["order_id"] == "order_record:MR-1042"
    assert mapping["product"] == "order_record:MR-1042"
    assert mapping["customer_question"] == "support_queue:C1"
    assert mapping["issue_status"] == "issue_record:C1"


def test_build_known_facts_excludes_null_and_excluded_fields():
    cases = app.load_cases(app.CASES_PATH)
    facts = app.build_known_facts(cases["C2"], "C2", EVIDENCE_MAP, exclude=("delivery_date",))
    fields = {f["field"] for f in facts}
    assert "delivery_date" not in fields  # excluded explicitly
    assert "order_id" in fields
    for fact in facts:
        assert fact["source_id"] in EVIDENCE_MAP["C2"]


# --- task.md per-case generalization ---------------------------------------

def test_build_task_for_case_substitutes_evidence_section():
    task_text = app.build_task_for_case(CONTEXT["task"], "C1", EVIDENCE_MAP)
    assert "support_queue:C1" in task_text
    assert "order_record:MR-1042" in task_text
    assert "issue_record:C1" in task_text
    assert "support_queue:C2" not in task_text  # original C2 wording is gone


def test_build_task_for_case_preserves_other_sections():
    task_text = app.build_task_for_case(CONTEXT["task"], "C3", EVIDENCE_MAP)
    assert "## Boundaries" in task_text
    assert "No return approval, refund, customer message" in task_text
    assert "## Output contract" in task_text


# --- prompt building -----------------------------------------------------

def test_build_messages_includes_case_fields_and_source_ids():
    messages = app.build_messages(VALID_CASE, "C1", EVIDENCE_MAP)
    joined = json.dumps(messages)
    assert "MR-1042" in joined
    assert "Jacket" in joined
    assert "order_record:MR-1042" in joined


def test_build_system_prompt_includes_task_definitions_contract_and_policy():
    prompt = app.build_system_prompt("C1", CONTEXT)
    assert "Prepare a structured case review" in prompt  # from task.md
    assert "source_id" in prompt  # from definitions.md
    assert "Output Contract Definitions" in prompt  # from contractDefinitions.md
    assert "P1." in prompt  # from policy.md


# --- schema validation ---------------------------------------------------

def test_parse_and_validate_model_output_accepts_valid_json():
    data = app.parse_and_validate_model_output(VALID_MODEL_JSON, "C1", EVIDENCE_MAP)
    assert data["review_status"] == "READY_FOR_HUMAN_REVIEW"
    assert data["case_id"] == "C1"


def test_parse_and_validate_model_output_rejects_non_json():
    with pytest.raises(app.ModelOutputError):
        app.parse_and_validate_model_output("not json at all", "C1", EVIDENCE_MAP)


def test_parse_and_validate_model_output_rejects_missing_key():
    bad = dict(VALID_MODEL_RESULT)
    del bad["human_action_required"]
    with pytest.raises(app.ModelOutputError):
        app.parse_and_validate_model_output(json.dumps(bad), "C1", EVIDENCE_MAP)


def test_parse_and_validate_model_output_rejects_extra_key():
    bad = dict(VALID_MODEL_RESULT)
    bad["extra_field"] = "nope"
    with pytest.raises(app.ModelOutputError):
        app.parse_and_validate_model_output(json.dumps(bad), "C1", EVIDENCE_MAP)


def test_parse_and_validate_model_output_rejects_wrong_case_id():
    bad = dict(VALID_MODEL_RESULT)
    bad["case_id"] = "C2"  # doesn't match the case_id passed in
    with pytest.raises(app.ModelOutputError):
        app.parse_and_validate_model_output(json.dumps(bad), "C1", EVIDENCE_MAP)


def test_parse_and_validate_model_output_rejects_wrong_type():
    bad = dict(VALID_MODEL_RESULT)
    bad["evidence_refs"] = "P1"  # should be a list
    with pytest.raises(app.ModelOutputError):
        app.parse_and_validate_model_output(json.dumps(bad), "C1", EVIDENCE_MAP)


def test_parse_and_validate_model_output_rejects_bad_review_status():
    bad = dict(VALID_MODEL_RESULT)
    bad["review_status"] = "MAYBE"
    with pytest.raises(app.ModelOutputError):
        app.parse_and_validate_model_output(json.dumps(bad), "C1", EVIDENCE_MAP)


def test_parse_and_validate_model_output_rejects_unresolvable_known_facts_source_id():
    bad = dict(VALID_MODEL_RESULT)
    bad["known_facts"] = [{"field": "order_id", "value": "MR-1042", "source_id": "made_up_source:XYZ"}]
    with pytest.raises(app.ModelOutputError):
        app.parse_and_validate_model_output(json.dumps(bad), "C1", EVIDENCE_MAP)


def test_parse_and_validate_model_output_rejects_malformed_known_facts_item():
    bad = dict(VALID_MODEL_RESULT)
    bad["known_facts"] = [{"field": "order_id", "value": "MR-1042"}]  # missing source_id
    with pytest.raises(app.ModelOutputError):
        app.parse_and_validate_model_output(json.dumps(bad), "C1", EVIDENCE_MAP)


def test_parse_and_validate_model_output_rejects_unresolvable_evidence_links_source_id():
    bad = dict(VALID_MODEL_RESULT)
    bad["evidence_links"] = [{"claim": "x", "source_id": "made_up_source:XYZ"}]
    with pytest.raises(app.ModelOutputError):
        app.parse_and_validate_model_output(json.dumps(bad), "C1", EVIDENCE_MAP)


def test_parse_and_validate_model_output_accepts_empty_conflicting_information():
    good = dict(VALID_MODEL_RESULT)
    good["conflicting_information"] = []
    data = app.parse_and_validate_model_output(json.dumps(good), "C1", EVIDENCE_MAP)
    assert data["conflicting_information"] == []


def test_parse_and_validate_model_output_rejects_unresolvable_conflicting_information_source_id():
    bad = dict(VALID_MODEL_RESULT)
    bad["conflicting_information"] = [{"description": "x", "source_ids": ["made_up_source:XYZ"]}]
    with pytest.raises(app.ModelOutputError):
        app.parse_and_validate_model_output(json.dumps(bad), "C1", EVIDENCE_MAP)


# --- get_online_result (fake model function, no real API calls) ---------

def test_get_online_result_success_with_fake_model_for_c1():
    def fake_model(client, model_id, case, context):
        assert case["case_id"] == "C1"
        return VALID_MODEL_JSON

    result, error = app.get_online_result(None, "fake-model", VALID_CASE, CONTEXT, model_fn=fake_model)

    assert error is None
    assert result["review_status"] == "READY_FOR_HUMAN_REVIEW"
    assert result["evidence_refs"] == ["P1", "P4"]


def test_get_online_result_invalid_model_output_falls_back_to_unavailable():
    def fake_model(client, model_id, case, context):
        return "this is not JSON"

    result, error = app.get_online_result(None, "fake-model", VALID_CASE, CONTEXT, model_fn=fake_model)

    assert result == app.unavailable_result(VALID_CASE, EVIDENCE_MAP)
    assert error is not None


def test_get_online_result_timeout_falls_back_to_unavailable():
    def fake_model(client, model_id, case, context):
        raise TimeoutError("simulated provider timeout")

    result, error = app.get_online_result(None, "fake-model", VALID_CASE, CONTEXT, model_fn=fake_model)

    assert result == app.unavailable_result(VALID_CASE, EVIDENCE_MAP)
    assert "timeout" in error.lower()


def test_get_online_result_provider_error_falls_back_to_unavailable():
    def fake_model(client, model_id, case, context):
        raise RuntimeError("connection refused")

    result, error = app.get_online_result(None, "fake-model", VALID_CASE, CONTEXT, model_fn=fake_model)

    assert result == app.unavailable_result(VALID_CASE, EVIDENCE_MAP)
    assert error is not None


# --- missing delivery_date short-circuit (P2) ------------------------------

def test_missing_delivery_date_returns_deterministic_result_without_calling_model():
    def fail_if_called(*args, **kwargs):
        raise AssertionError("model should not be called when delivery_date is missing")

    cases = app.load_cases(app.CASES_PATH)
    case_c2 = cases["C2"]
    assert app.is_delivery_date_missing(case_c2)

    result, notes = app.resolve_result(case_c2, CONTEXT, model_fn=fail_if_called)

    assert notes == []
    assert result["review_status"] == "NEEDS_INFORMATION"
    assert result["missing_information"] == ["delivery_date"]
    assert result["evidence_refs"] == ["P2", "P4"]
    assert result["case_id"] == "C2"
    assert "delivery_date" not in {f["field"] for f in result["known_facts"]}


def test_is_delivery_date_missing_true_for_none_and_blank():
    assert app.is_delivery_date_missing({"delivery_date": None}) is True
    assert app.is_delivery_date_missing({"delivery_date": ""}) is True
    assert app.is_delivery_date_missing({"delivery_date": "   "}) is True


def test_is_delivery_date_missing_false_when_present():
    assert app.is_delivery_date_missing({"delivery_date": "2026-09-15"}) is False


def test_resolve_result_calls_model_when_delivery_date_present():
    def fake_model(client, model_id, case, context):
        return VALID_MODEL_JSON

    result, notes = app.resolve_result(VALID_CASE, CONTEXT, model_fn=fake_model)

    assert notes == []
    assert result["review_status"] == "READY_FOR_HUMAN_REVIEW"


# --- refund window short-circuit (78 hours) --------------------------------

def test_is_within_refund_window_true_just_inside():
    now = datetime(2026, 9, 20, 12, 0, 0)
    case = {"delivery_date": "2026-09-18"}  # 60 hours before `now`
    assert app.is_within_refund_window(case, now=now) is True


def test_is_within_refund_window_true_at_exact_boundary():
    now = datetime(2026, 9, 20, 6, 0, 0)
    case = {"delivery_date": "2026-09-17"}  # exactly 78 hours before `now`
    assert app.is_within_refund_window(case, now=now) is True


def test_is_within_refund_window_false_just_outside():
    now = datetime(2026, 9, 20, 6, 0, 1)
    case = {"delivery_date": "2026-09-17"}  # 78 hours and 1 second before `now`
    assert app.is_within_refund_window(case, now=now) is False


def test_is_within_refund_window_false_well_outside():
    now = datetime(2026, 9, 28, 0, 0, 0)
    case = {"delivery_date": "2026-09-15"}  # about 312 hours before `now`
    assert app.is_within_refund_window(case, now=now) is False


def test_refund_window_expired_result_shape():
    cases = app.load_cases(app.CASES_PATH)
    result = app.refund_window_expired_result(cases["C1"], EVIDENCE_MAP)
    assert result["review_status"] == app.REJECTED_STATUS
    assert result["missing_information"] == []
    assert result["case_id"] == "C1"
    assert "78-hour" in result["draft_reply"]
    assert set(result.keys()) == app.REQUIRED_RESULT_KEYS


def test_resolve_result_rejects_without_calling_model_when_window_expired():
    def fail_if_called(*args, **kwargs):
        raise AssertionError("model should not be called when the refund window has expired")

    expired_case = {**VALID_CASE, "delivery_date": "2020-01-01"}  # always > 78h in the past

    result, notes = app.resolve_result(expired_case, CONTEXT, model_fn=fail_if_called)

    assert notes == []
    assert result == app.refund_window_expired_result(expired_case, EVIDENCE_MAP)


def test_main_case_c2_returns_deterministic_result_without_calling_client_or_model(
    monkeypatch, capsys
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("get_client/model should not be called for C2 (missing delivery_date)")

    monkeypatch.setattr(app, "get_client", fail_if_called)
    monkeypatch.setattr(app, "call_model", fail_if_called)

    exit_code = app.main(["--case", "C2"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "review_status: NEEDS_INFORMATION" in out
    assert "please provide the delivery date" in out
    assert "['P2', 'P4']" in out
    assert "['delivery_date']" in out


# --- offline demo ---------------------------------------------------------

def test_load_offline_result_for_each_case():
    for case_id in ("C1", "C2", "C3"):
        result = app.load_offline_result(case_id)
        assert set(result.keys()) == app.REQUIRED_RESULT_KEYS
        assert result["case_id"] == case_id
        assert result["review_status"] in app.VALID_REVIEW_STATUSES


def test_load_offline_result_unknown_case_raises():
    with pytest.raises(KeyError):
        app.load_offline_result("C99")


# --- unavailable state -----------------------------------------------------

def test_unavailable_result_shape():
    result = app.unavailable_result(VALID_CASE, EVIDENCE_MAP)
    assert result["case_id"] == "C1"
    assert result["draft_reply"] == ""
    assert result["evidence_refs"] == []
    assert result["missing_information"] == []
    assert result["review_status"] == "UNAVAILABLE"
    assert set(result.keys()) == app.REQUIRED_RESULT_KEYS


# --- backend selection -------------------------------------------------------

def test_get_backend_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("APP_MODEL_BACKEND", raising=False)
    assert app.get_backend() == "ollama"


def test_get_backend_reads_env_override(monkeypatch):
    monkeypatch.setenv("APP_MODEL_BACKEND", "OpenRouter")
    assert app.get_backend() == "openrouter"


def test_get_backend_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("APP_MODEL_BACKEND", "not-a-real-backend")
    with pytest.raises(RuntimeError):
        app.get_backend()


# --- ollama backend -----------------------------------------------------------

def test_get_ollama_host_defaults(monkeypatch):
    monkeypatch.delenv("APP_OLLAMA_HOST", raising=False)
    assert app.get_ollama_host() == app.DEFAULT_OLLAMA_HOST


def test_get_ollama_host_reads_env_override(monkeypatch):
    monkeypatch.setenv("APP_OLLAMA_HOST", "http://example.local:11434")
    assert app.get_ollama_host() == "http://example.local:11434"


def test_get_model_id_defaults_for_ollama(monkeypatch):
    monkeypatch.delenv("APP_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("APP_OLLAMA_MODEL", raising=False)
    assert app.get_model_id() == app.DEFAULT_OLLAMA_MODEL


def test_get_model_id_reads_env_override_for_ollama(monkeypatch):
    monkeypatch.delenv("APP_MODEL_BACKEND", raising=False)
    monkeypatch.setenv("APP_OLLAMA_MODEL", "some-other-model")
    assert app.get_model_id() == "some-other-model"


def test_get_client_returns_ollama_client_without_requiring_any_config(monkeypatch):
    monkeypatch.delenv("APP_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("APP_OLLAMA_HOST", raising=False)
    monkeypatch.delenv("APP_OLLAMA_MODEL", raising=False)
    client = app.get_client()
    assert isinstance(client, Client)


def test_call_model_dispatches_to_ollama_by_default(monkeypatch):
    monkeypatch.delenv("APP_MODEL_BACKEND", raising=False)

    class FakeClient:
        def chat(self, **kwargs):
            assert kwargs["format"] == "json"
            return {"message": {"role": "assistant", "content": VALID_MODEL_JSON}}

    result = app.call_model(FakeClient(), "fake-model", VALID_CASE, CONTEXT)
    assert result == VALID_MODEL_JSON


# --- openrouter backend ---------------------------------------------------

def test_get_client_raises_without_openrouter_api_key(monkeypatch):
    monkeypatch.setenv("APP_MODEL_BACKEND", "openrouter")
    monkeypatch.delenv("APP_OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        app.get_client()


def test_get_client_returns_openrouter_client_with_key(monkeypatch):
    monkeypatch.setenv("APP_MODEL_BACKEND", "openrouter")
    monkeypatch.setenv("APP_OPENROUTER_API_KEY", "sk-or-test-key")
    client = app.get_client()
    assert isinstance(client, OpenAI)


def test_get_model_id_raises_without_openrouter_model(monkeypatch):
    monkeypatch.setenv("APP_MODEL_BACKEND", "openrouter")
    monkeypatch.delenv("APP_OPENROUTER_MODEL", raising=False)
    with pytest.raises(RuntimeError):
        app.get_model_id()


def test_get_model_id_reads_openrouter_model(monkeypatch):
    monkeypatch.setenv("APP_MODEL_BACKEND", "openrouter")
    monkeypatch.setenv("APP_OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    assert app.get_model_id() == "meta-llama/llama-3.1-8b-instruct:free"


def test_call_model_dispatches_to_openrouter(monkeypatch):
    monkeypatch.setenv("APP_MODEL_BACKEND", "openrouter")

    class FakeMessage:
        content = VALID_MODEL_JSON

    class FakeChoice:
        message = FakeMessage()

    class FakeCompletionResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["response_format"] == {"type": "json_object"}
            return FakeCompletionResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    result = app.call_model(FakeClient(), "fake-model", VALID_CASE, CONTEXT)
    assert result == VALID_MODEL_JSON


def test_resolve_result_configuration_error_when_openrouter_key_missing(monkeypatch):
    monkeypatch.setenv("APP_MODEL_BACKEND", "openrouter")
    monkeypatch.delenv("APP_OPENROUTER_API_KEY", raising=False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("model should not be called when config is invalid")

    result, notes = app.resolve_result(VALID_CASE, CONTEXT, model_fn=fail_if_called)

    assert result == app.unavailable_result(VALID_CASE, EVIDENCE_MAP)
    assert any("CONFIGURATION ERROR" in note for note in notes)


# --- CLI-level behaviour ----------------------------------------------------

def test_parse_args_requires_case():
    with pytest.raises(SystemExit):
        app.parse_args([])


def test_parse_args_rejects_invalid_case():
    with pytest.raises(SystemExit):
        app.parse_args(["--case", "C99"])


def test_main_simulate_timeout_never_calls_client(monkeypatch, capsys):
    def fail_if_called():
        raise AssertionError("get_client should not be called in --simulate-timeout mode")

    monkeypatch.setattr(app, "get_client", fail_if_called)

    exit_code = app.main(["--case", "C1", "--simulate-timeout"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "UNAVAILABLE" in out
    assert "SIMULATED TIMEOUT" in out
    assert app.MANUAL_FALLBACK_MESSAGE in out


def test_main_offline_demo_never_calls_client_and_labels_output(monkeypatch, capsys):
    def fail_if_called():
        raise AssertionError("get_client should not be called in --offline-demo mode")

    monkeypatch.setattr(app, "get_client", fail_if_called)

    exit_code = app.main(["--case", "C2", "--offline-demo"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "no model API was called" in out
    assert "SUPPLIED FACTS" in out
    assert "SUPPLIED TASK" in out
    assert "SUPPLIED POLICY" in out
    assert "DRAFT RESULT FOR HUMAN REVIEW" in out


def test_main_prints_case_specific_task_evidence_section(monkeypatch, capsys):
    monkeypatch.setattr(app, "get_client", lambda: (_ for _ in ()).throw(AssertionError("unused")))

    app.main(["--case", "C3", "--offline-demo"])
    out = capsys.readouterr().out

    assert "support_queue:C3" in out
    assert "order_record:MR-1126" in out
