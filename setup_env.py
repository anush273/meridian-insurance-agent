"""One-time local setup: asks which model backend to use, prompts for the
settings that backend needs, and writes a project-only .env file.

Ollama needs no secret (just a host and model). OpenRouter needs a real API
key, so that prompt uses hidden input and the key is never printed."""

import getpass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

DEFAULT_BACKEND = "ollama"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
VALID_BACKENDS = ("ollama", "openrouter")


def prompt_backend(reader=input, default=DEFAULT_BACKEND):
    value = reader(f"Model backend, 'ollama' or 'openrouter' [{default}]: ").strip().lower()
    return value or default


def prompt_host(reader=input, default=DEFAULT_OLLAMA_HOST):
    value = reader(f"Ollama host [{default}]: ").strip()
    return value or default


def prompt_ollama_model(reader=input, default=DEFAULT_OLLAMA_MODEL):
    value = reader(f"Ollama model to use [{default}]: ").strip()
    return value or default


def prompt_openrouter_api_key(reader=getpass.getpass):
    return reader("Enter your OpenRouter API key (input hidden): ").strip()


def prompt_openrouter_model(reader=input):
    return reader(
        "OpenRouter model id -- pick a free one from "
        "https://openrouter.ai/models?max_price=0, e.g. "
        "'meta-llama/llama-3.1-8b-instruct:free': "
    ).strip()


def write_env_file(env_path, lines):
    Path(env_path).write_text("\n".join(lines) + "\n")


def main(
    env_path=ENV_PATH,
    backend_reader=input,
    host_reader=input,
    ollama_model_reader=input,
    api_key_reader=getpass.getpass,
    openrouter_model_reader=input,
):
    backend = prompt_backend(backend_reader)
    if backend not in VALID_BACKENDS:
        print(f"Unrecognized backend '{backend}', defaulting to '{DEFAULT_BACKEND}'.")
        backend = DEFAULT_BACKEND

    lines = [f"APP_MODEL_BACKEND={backend}"]

    if backend == "openrouter":
        api_key = prompt_openrouter_api_key(api_key_reader)
        if not api_key:
            print("No API key entered. Aborting setup; .env was not written.")
            return 1
        model = prompt_openrouter_model(openrouter_model_reader)
        if not model:
            print("No model entered. Aborting setup; .env was not written.")
            return 1
        lines.append(f"APP_OPENROUTER_API_KEY={api_key}")
        lines.append(f"APP_OPENROUTER_MODEL={model}")
    else:
        host = prompt_host(host_reader)
        model = prompt_ollama_model(ollama_model_reader)
        lines.append(f"APP_OLLAMA_HOST={host}")
        lines.append(f"APP_OLLAMA_MODEL={model}")

    write_env_file(env_path, lines)

    print(f"Saved configuration to {env_path}")
    print(f"Backend set to: {backend}")
    if backend == "openrouter":
        print("The API key was written to .env but was not printed to the terminal.")
    print(f"Model set to: {model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
