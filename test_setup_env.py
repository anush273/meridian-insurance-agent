import setup_env


def test_write_env_file_contains_host_and_model(tmp_path):
    env_path = tmp_path / ".env"
    setup_env.write_env_file(env_path, "http://localhost:11434", "llama3.2")

    content = env_path.read_text()
    assert "APP_OLLAMA_HOST=http://localhost:11434" in content
    assert "APP_OLLAMA_MODEL=llama3.2" in content


def test_prompt_host_returns_default_on_blank_input():
    host = setup_env.prompt_host(reader=lambda prompt: "", default="http://localhost:11434")
    assert host == "http://localhost:11434"


def test_prompt_host_returns_typed_value():
    host = setup_env.prompt_host(
        reader=lambda prompt: "http://remote-host:11434", default="http://localhost:11434"
    )
    assert host == "http://remote-host:11434"


def test_prompt_model_returns_default_on_blank_input():
    model = setup_env.prompt_model(reader=lambda prompt: "", default="llama3.2")
    assert model == "llama3.2"


def test_prompt_model_returns_typed_value():
    model = setup_env.prompt_model(reader=lambda prompt: "custom-model-id", default="llama3.2")
    assert model == "custom-model-id"


def test_main_writes_host_and_model_to_env_file(tmp_path, capsys):
    env_path = tmp_path / ".env"

    exit_code = setup_env.main(
        env_path=env_path,
        host_reader=lambda prompt: "",
        model_reader=lambda prompt: "",
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert env_path.read_text() == (
        f"APP_OLLAMA_HOST={setup_env.DEFAULT_HOST}\n"
        f"APP_OLLAMA_MODEL={setup_env.DEFAULT_MODEL}\n"
    )
    assert setup_env.DEFAULT_MODEL in captured.out


def test_main_respects_typed_overrides(tmp_path):
    env_path = tmp_path / ".env"

    setup_env.main(
        env_path=env_path,
        host_reader=lambda prompt: "http://gpu-box:11434",
        model_reader=lambda prompt: "qwen2.5:7b",
    )

    content = env_path.read_text()
    assert "APP_OLLAMA_HOST=http://gpu-box:11434" in content
    assert "APP_OLLAMA_MODEL=qwen2.5:7b" in content
