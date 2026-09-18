"""One-time local setup: prompts for the local Ollama host and model, and
writes a project-only .env file. Local Ollama has no API key -- there is
nothing secret to prompt for or hide here."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"


def prompt_host(reader=input, default=DEFAULT_HOST):
    value = reader(f"Ollama host [{default}]: ").strip()
    return value or default


def prompt_model(reader=input, default=DEFAULT_MODEL):
    value = reader(f"Model to use [{default}]: ").strip()
    return value or default


def write_env_file(env_path, host, model):
    env_path = Path(env_path)
    content = f"APP_OLLAMA_HOST={host}\nAPP_OLLAMA_MODEL={model}\n"
    env_path.write_text(content)


def main(env_path=ENV_PATH, host_reader=input, model_reader=input):
    host = prompt_host(host_reader, DEFAULT_HOST)
    model = prompt_model(model_reader, DEFAULT_MODEL)
    write_env_file(env_path, host, model)

    print(f"Saved configuration to {env_path}")
    print(f"Host set to: {host}")
    print(f"Model set to: {model}")
    print(f"Make sure the model is pulled: ollama pull {model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
