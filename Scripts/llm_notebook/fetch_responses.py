#!/usr/bin/env python
"""Query each model with each scenario's prompt and cache the code response.

For every scenario in config.toml and every model in the model list, send the
scenario's prompt (text only, no image) to the model and save the code it
returns to responses/<scenario>_<model_id>.py.

API keys are read from a .env file next to this script (see .env.example), or
from the environment variables named by api_key_env in config.toml.

Run:
  pixi run --environment intro-ia python Scripts/llm_notebook/fetch_responses.py
"""

import argparse
import json
import os
import re
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESPONSES_DIR = HERE / "responses"

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_COMPATIBLE = {
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/chat/completions",
}


def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def load_env(path):
    """Load KEY=VALUE pairs from a .env file into os.environ (no overwriting)."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def extract_code(text):
    """Pull the first fenced code block out of a model response."""
    match = re.search(
        r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return text.strip()


def post_json(url, headers, body):
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def call_anthropic(api_key, model, prompt):
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    result = post_json(ANTHROPIC_URL, headers, body)
    for block in result["content"]:
        if block.get("type") == "text":
            return block["text"]
    raise RuntimeError(f"no text block in Anthropic response: {result}")


def call_openai_compatible(base_url, api_key, model, prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    result = post_json(base_url, headers, body)
    content = result["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return content


def call_model(model, prompt):
    provider = model["provider"]
    api_key = os.environ.get(model["api_key_env"])
    if not api_key:
        raise RuntimeError(f"missing environment variable {model['api_key_env']}")
    if provider == "anthropic":
        return call_anthropic(api_key, model["model"], prompt)
    if provider in OPENAI_COMPATIBLE:
        return call_openai_compatible(
            OPENAI_COMPATIBLE[provider], api_key, model["model"], prompt
        )
    raise RuntimeError(f"unknown provider {provider!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-fetch every model even if a cached response already exists",
    )
    args = parser.parse_args()

    load_env(HERE / ".env")
    config = tomllib.loads(read_text(HERE / "config.toml"))
    models = config["models"]
    for scenario in config["scenarios"]:
        prompt = read_text(HERE / scenario["prompt_file"]).strip()
        for model in models:
            key = f"{scenario['name']}/{model['id']}"
            out_path = RESPONSES_DIR / f"{scenario['name']}_{model['id']}.py"
            if out_path.exists() and not args.force:
                print(f"{key}: skipped (already exists)")
                continue
            try:
                text = call_model(model, prompt)
                code = extract_code(text)
                out_path.write_text(code + "\n", encoding="utf-8")
                print(f"{key}: wrote {out_path.name}")
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"{key}: FAILED ({exc})")


if __name__ == "__main__":
    main()
