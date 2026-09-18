import setup_env


def test_write_env_file_contains_key_and_model(tmp_path):
    env_path = tmp_path / ".env"
    setup_env.write_env_file(env_path, "sk-real-key", "claude-haiku-4-5-20251001")

    content = env_path.read_text()
    assert "APP_ANTHROPIC_API_KEY=sk-real-key" in content
    assert "APP_ANTHROPIC_MODEL=claude-haiku-4-5-20251001" in content


def test_prompt_api_key_strips_whitespace_and_uses_hidden_reader():
    calls = []

    def fake_hidden_reader(prompt):
        calls.append(prompt)
        return "  sk-hidden-key  \n"

    key = setup_env.prompt_api_key(reader=fake_hidden_reader)

    assert key == "sk-hidden-key"
    assert len(calls) == 1


def test_prompt_model_returns_default_on_blank_input():
    model = setup_env.prompt_model(reader=lambda prompt: "", default="claude-haiku-4-5-20251001")
    assert model == "claude-haiku-4-5-20251001"


def test_prompt_model_returns_typed_value():
    model = setup_env.prompt_model(reader=lambda prompt: "custom-model-id", default="claude-haiku-4-5-20251001")
    assert model == "custom-model-id"


def test_main_writes_key_to_env_file_without_printing_it(tmp_path, capsys):
    env_path = tmp_path / ".env"
    secret_key = "sk-super-secret-value"

    exit_code = setup_env.main(
        env_path=env_path,
        api_key_reader=lambda prompt: secret_key,
        model_reader=lambda prompt: "",
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert env_path.read_text() == (
        f"APP_ANTHROPIC_API_KEY={secret_key}\n"
        f"APP_ANTHROPIC_MODEL={setup_env.DEFAULT_MODEL}\n"
    )
    assert secret_key not in captured.out
    assert secret_key not in captured.err


def test_main_aborts_without_writing_file_when_no_key_entered(tmp_path, capsys):
    env_path = tmp_path / ".env"

    exit_code = setup_env.main(
        env_path=env_path,
        api_key_reader=lambda prompt: "",
        model_reader=lambda prompt: "",
    )

    assert exit_code == 1
    assert not env_path.exists()
