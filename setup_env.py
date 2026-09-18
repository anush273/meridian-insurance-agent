"""One-time local setup: prompts for the app's Anthropic API key (hidden
input) and model, and writes a project-only .env file. The key is never
printed to the terminal."""

import getpass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def prompt_api_key(reader=getpass.getpass):
    return reader("Enter your Anthropic API key (input hidden): ").strip()


def prompt_model(reader=input, default=DEFAULT_MODEL):
    value = reader(f"Model to use [{default}]: ").strip()
    return value or default


def write_env_file(env_path, api_key, model):
    env_path = Path(env_path)
    content = f"APP_ANTHROPIC_API_KEY={api_key}\nAPP_ANTHROPIC_MODEL={model}\n"
    env_path.write_text(content)


def main(env_path=ENV_PATH, api_key_reader=getpass.getpass, model_reader=input):
    api_key = prompt_api_key(api_key_reader)
    if not api_key:
        print("No API key entered. Aborting setup; .env was not written.")
        return 1

    model = prompt_model(model_reader, DEFAULT_MODEL)
    write_env_file(env_path, api_key, model)

    print(f"Saved configuration to {env_path}")
    print("The API key was written to .env but was not printed to the terminal.")
    print(f"Model set to: {model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
