import setup_env


def test_prompt_backend_returns_default_on_blank_input():
    backend = setup_env.prompt_backend(reader=lambda prompt: "", default="ollama")
    assert backend == "ollama"


def test_prompt_backend_returns_typed_value():
    backend = setup_env.prompt_backend(reader=lambda prompt: "OpenRouter", default="ollama")
    assert backend == "openrouter"


def test_prompt_host_returns_default_on_blank_input():
    host = setup_env.prompt_host(reader=lambda prompt: "", default="http://localhost:11434")
    assert host == "http://localhost:11434"


def test_prompt_ollama_model_returns_default_on_blank_input():
    model = setup_env.prompt_ollama_model(reader=lambda prompt: "", default="llama3.2")
    assert model == "llama3.2"


def test_prompt_openrouter_api_key_strips_whitespace_and_uses_hidden_reader():
    calls = []

    def fake_hidden_reader(prompt):
        calls.append(prompt)
        return "  sk-or-hidden-key  \n"

    key = setup_env.prompt_openrouter_api_key(reader=fake_hidden_reader)

    assert key == "sk-or-hidden-key"
    assert len(calls) == 1


def test_prompt_openrouter_model_returns_typed_value():
    model = setup_env.prompt_openrouter_model(reader=lambda prompt: "meta-llama/llama-3.1-8b-instruct:free")
    assert model == "meta-llama/llama-3.1-8b-instruct:free"


def test_main_defaults_to_ollama_and_writes_host_and_model(tmp_path):
    env_path = tmp_path / ".env"

    exit_code = setup_env.main(
        env_path=env_path,
        backend_reader=lambda prompt: "",
        host_reader=lambda prompt: "",
        ollama_model_reader=lambda prompt: "",
    )

    content = env_path.read_text()
    assert exit_code == 0
    assert "APP_MODEL_BACKEND=ollama" in content
    assert f"APP_OLLAMA_HOST={setup_env.DEFAULT_OLLAMA_HOST}" in content
    assert f"APP_OLLAMA_MODEL={setup_env.DEFAULT_OLLAMA_MODEL}" in content
    assert "OPENROUTER" not in content


def test_main_openrouter_writes_key_to_env_file_without_printing_it(tmp_path, capsys):
    env_path = tmp_path / ".env"
    secret_key = "sk-or-super-secret-value"

    exit_code = setup_env.main(
        env_path=env_path,
        backend_reader=lambda prompt: "openrouter",
        api_key_reader=lambda prompt: secret_key,
        openrouter_model_reader=lambda prompt: "meta-llama/llama-3.1-8b-instruct:free",
    )

    captured = capsys.readouterr()
    content = env_path.read_text()

    assert exit_code == 0
    assert "APP_MODEL_BACKEND=openrouter" in content
    assert f"APP_OPENROUTER_API_KEY={secret_key}" in content
    assert "APP_OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct:free" in content
    assert "APP_OLLAMA" not in content
    assert secret_key not in captured.out
    assert secret_key not in captured.err


def test_main_openrouter_aborts_without_writing_file_when_no_key_entered(tmp_path):
    env_path = tmp_path / ".env"

    exit_code = setup_env.main(
        env_path=env_path,
        backend_reader=lambda prompt: "openrouter",
        api_key_reader=lambda prompt: "",
        openrouter_model_reader=lambda prompt: "meta-llama/llama-3.1-8b-instruct:free",
    )

    assert exit_code == 1
    assert not env_path.exists()


def test_main_openrouter_aborts_without_writing_file_when_no_model_entered(tmp_path):
    env_path = tmp_path / ".env"

    exit_code = setup_env.main(
        env_path=env_path,
        backend_reader=lambda prompt: "openrouter",
        api_key_reader=lambda prompt: "sk-or-some-key",
        openrouter_model_reader=lambda prompt: "",
    )

    assert exit_code == 1
    assert not env_path.exists()


def test_main_unrecognized_backend_falls_back_to_ollama(tmp_path):
    env_path = tmp_path / ".env"

    exit_code = setup_env.main(
        env_path=env_path,
        backend_reader=lambda prompt: "not-a-real-backend",
        host_reader=lambda prompt: "",
        ollama_model_reader=lambda prompt: "",
    )

    content = env_path.read_text()
    assert exit_code == 0
    assert "APP_MODEL_BACKEND=ollama" in content
