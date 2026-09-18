import json

import pytest
from ollama import Client

import app


VALID_CASE = {
    "case_id": "C1",
    "order_id": "MR-1042",
    "product": "Jacket",
    "customer_question": "q",
    "delivery_date": "2026-09-15",
    "issue_status": "open",
    "issue_details": "d",
}

VALID_MODEL_JSON = json.dumps(
    {
        "draft_reply": "Thanks for reaching out, this is being referred for review.",
        "evidence_refs": ["P1", "P4"],
        "missing_information": [],
        "review_status": "READY_FOR_HUMAN_REVIEW",
    }
)


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


def test_demo_outputs_file_has_all_three_cases_with_required_shape():
    with open(app.DEMO_OUTPUTS_PATH) as f:
        data = json.load(f)
    for case_id in ("C1", "C2", "C3"):
        entry = data[case_id]
        assert isinstance(entry["draft_reply"], str)
        assert isinstance(entry["evidence_refs"], list)
        assert isinstance(entry["missing_information"], list)
        assert entry["review_status"] in app.VALID_REVIEW_STATUSES


# --- prompt building -----------------------------------------------------

def test_build_messages_includes_case_fields():
    messages = app.build_messages(VALID_CASE)
    joined = json.dumps(messages)
    assert "MR-1042" in joined
    assert "Jacket" in joined


def test_build_system_prompt_includes_policy_text():
    prompt = app.build_system_prompt("POLICY TEXT P1.")
    assert "POLICY TEXT P1." in prompt


# --- schema validation ---------------------------------------------------

def test_parse_and_validate_model_output_accepts_valid_json():
    data = app.parse_and_validate_model_output(VALID_MODEL_JSON)
    assert data["review_status"] == "READY_FOR_HUMAN_REVIEW"
    assert data["evidence_refs"] == ["P1", "P4"]


def test_parse_and_validate_model_output_rejects_non_json():
    with pytest.raises(app.ModelOutputError):
        app.parse_and_validate_model_output("not json at all")


def test_parse_and_validate_model_output_rejects_missing_key():
    bad = json.dumps({"draft_reply": "x", "evidence_refs": [], "review_status": "BLOCKED"})
    with pytest.raises(app.ModelOutputError):
        app.parse_and_validate_model_output(bad)


def test_parse_and_validate_model_output_rejects_extra_key():
    bad = json.dumps(
        {
            "draft_reply": "x",
            "evidence_refs": [],
            "missing_information": [],
            "review_status": "BLOCKED",
            "extra_field": "nope",
        }
    )
    with pytest.raises(app.ModelOutputError):
        app.parse_and_validate_model_output(bad)


def test_parse_and_validate_model_output_rejects_wrong_type():
    bad = json.dumps(
        {
            "draft_reply": "x",
            "evidence_refs": "P1",  # should be a list
            "missing_information": [],
            "review_status": "BLOCKED",
        }
    )
    with pytest.raises(app.ModelOutputError):
        app.parse_and_validate_model_output(bad)


def test_parse_and_validate_model_output_rejects_bad_review_status():
    bad = json.dumps(
        {
            "draft_reply": "x",
            "evidence_refs": [],
            "missing_information": [],
            "review_status": "MAYBE",
        }
    )
    with pytest.raises(app.ModelOutputError):
        app.parse_and_validate_model_output(bad)


# --- get_online_result (fake model function, no real API calls) ---------

def test_get_online_result_success_with_fake_model_for_c1():
    def fake_model(client, model_id, policy_text, case):
        assert case["case_id"] == "C1"
        return VALID_MODEL_JSON

    result, error = app.get_online_result(None, "fake-model", "policy", VALID_CASE, model_fn=fake_model)

    assert error is None
    assert result["review_status"] == "READY_FOR_HUMAN_REVIEW"
    assert result["evidence_refs"] == ["P1", "P4"]


def test_get_online_result_invalid_model_output_falls_back_to_unavailable():
    def fake_model(client, model_id, policy_text, case):
        return "this is not JSON"

    result, error = app.get_online_result(None, "fake-model", "policy", VALID_CASE, model_fn=fake_model)

    assert result == app.unavailable_result()
    assert error is not None


def test_get_online_result_timeout_falls_back_to_unavailable():
    def fake_model(client, model_id, policy_text, case):
        raise TimeoutError("simulated provider timeout")

    result, error = app.get_online_result(None, "fake-model", "policy", VALID_CASE, model_fn=fake_model)

    assert result == app.unavailable_result()
    assert "timeout" in error.lower()


def test_get_online_result_provider_error_falls_back_to_unavailable():
    def fake_model(client, model_id, policy_text, case):
        raise RuntimeError("connection refused")

    result, error = app.get_online_result(None, "fake-model", "policy", VALID_CASE, model_fn=fake_model)

    assert result == app.unavailable_result()
    assert error is not None


# --- missing delivery_date short-circuit (P2) ------------------------------

def test_missing_delivery_date_returns_deterministic_result_without_calling_model():
    def fail_if_called(*args, **kwargs):
        raise AssertionError("model should not be called when delivery_date is missing")

    cases = app.load_cases(app.CASES_PATH)
    case_c2 = cases["C2"]
    assert app.is_delivery_date_missing(case_c2)

    result, notes = app.resolve_result(case_c2, "policy text", model_fn=fail_if_called)

    assert notes == []
    assert result == {
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


def test_is_delivery_date_missing_true_for_none_and_blank():
    assert app.is_delivery_date_missing({"delivery_date": None}) is True
    assert app.is_delivery_date_missing({"delivery_date": ""}) is True
    assert app.is_delivery_date_missing({"delivery_date": "   "}) is True


def test_is_delivery_date_missing_false_when_present():
    assert app.is_delivery_date_missing({"delivery_date": "2026-09-15"}) is False


def test_resolve_result_calls_model_when_delivery_date_present():
    def fake_model(client, model_id, policy_text, case):
        return VALID_MODEL_JSON

    result, notes = app.resolve_result(VALID_CASE, "policy text", model_fn=fake_model)

    assert notes == []
    assert result["review_status"] == "READY_FOR_HUMAN_REVIEW"


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
        assert result["review_status"] in app.VALID_REVIEW_STATUSES


def test_load_offline_result_unknown_case_raises():
    with pytest.raises(KeyError):
        app.load_offline_result("C99")


# --- unavailable state -----------------------------------------------------

def test_unavailable_result_shape():
    result = app.unavailable_result()
    assert result["draft_reply"] == ""
    assert result["evidence_refs"] == []
    assert result["missing_information"] == []
    assert result["review_status"] == "UNAVAILABLE"


# --- client construction / env vars -----------------------------------------

def test_get_ollama_host_defaults(monkeypatch):
    monkeypatch.delenv("APP_OLLAMA_HOST", raising=False)
    assert app.get_ollama_host() == app.DEFAULT_OLLAMA_HOST


def test_get_ollama_host_reads_env_override(monkeypatch):
    monkeypatch.setenv("APP_OLLAMA_HOST", "http://example.local:11434")
    assert app.get_ollama_host() == "http://example.local:11434"


def test_get_model_id_defaults(monkeypatch):
    monkeypatch.delenv("APP_OLLAMA_MODEL", raising=False)
    assert app.get_model_id() == app.DEFAULT_MODEL


def test_get_model_id_reads_env_override(monkeypatch):
    monkeypatch.setenv("APP_OLLAMA_MODEL", "some-other-model")
    assert app.get_model_id() == "some-other-model"


def test_get_client_returns_client_without_requiring_any_config(monkeypatch):
    monkeypatch.delenv("APP_OLLAMA_HOST", raising=False)
    monkeypatch.delenv("APP_OLLAMA_MODEL", raising=False)
    client = app.get_client()
    assert isinstance(client, Client)


def test_call_model_extracts_content_from_chat_response():
    class FakeClient:
        def chat(self, **kwargs):
            assert kwargs["format"] == "json"
            return {"message": {"role": "assistant", "content": VALID_MODEL_JSON}}

    result = app.call_model(FakeClient(), "fake-model", "policy text", VALID_CASE)
    assert result == VALID_MODEL_JSON


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
    assert "SUPPLIED POLICY" in out
    assert "DRAFT RESULT FOR HUMAN REVIEW" in out
