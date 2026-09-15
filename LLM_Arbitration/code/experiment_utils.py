"""
Shared helpers used across all experiment scripts.
Keeping this in one place means every script logs and connects to
the local model the same way -- one thing to fix if something changes.
"""
import os
from datetime import datetime

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")


def make_ollama_client(model: str = "llama3.1:8b") -> OpenAIChatCompletionClient:
    """One shared way to connect any agent to the local Ollama model.
    Swap base_url/api_key/model here later to point at the real paid API instead."""
    return OpenAIChatCompletionClient(
        model=model,
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
    )


def make_groq_client(model: str = "openai/gpt-oss-20b") -> OpenAIChatCompletionClient:
    """Free tier, runs on Groq's own hardware -- much faster than local Ollama.
    Needs GROQ_API_KEY set as an environment variable
    (see PowerShell: $env:GROQ_API_KEY="...").
    Rate-limited (~30 requests/minute on free tier), but fine for this project's scale.
    Note: llama-3.1-8b-instant was deprecated by Groq in 2026 -- gpt-oss-20b is
    their current recommended lightweight replacement."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable not set. "
            "Run: $env:GROQ_API_KEY=\"your-key-here\" in PowerShell before running this script."
        )
    return OpenAIChatCompletionClient(
        model=model,
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
    )


def describe_priorities(value2issue: dict, value2reason: dict) -> str:
    """Turn a CaSiNo participant's priority fields into a readable block of text."""
    lines = []
    for level in ["High", "Medium", "Low"]:
        item = value2issue.get(level, "?")
        reason = value2reason.get(level, "")
        lines.append(f"- {level} priority: {item} (reason: \"{reason}\")")
    return "\n".join(lines)


def transcript_to_text(messages) -> str:
    """Turn an AutoGen message list into plain 'Speaker: text' lines."""
    lines = []
    for m in messages:
        if getattr(m, "source", None) in (None, "user"):
            continue
        lines.append(f"{m.source}: {m.content}")
    return "\n".join(lines)


def save_log(content: str, prefix: str) -> str:
    """Save a run's output to logs/<prefix>_<timestamp>.txt and return the path.
    Every experiment script should call this at the end so nothing gets lost."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.txt"
    path = os.path.join(LOGS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n[Saved log to {path}]")
    return path
