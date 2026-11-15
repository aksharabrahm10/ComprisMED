"""Groq Chat Completions client; credentials are read from the local environment."""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
JARGON_EXAMPLES = (
    Path(__file__).resolve().parents[1] / "data" / "jargon_training.csv"
)


@lru_cache(maxsize=1)
def _simplification_example() -> tuple[str, str]:
    example_source = "Please administer the medication orally."
    example_plain = "Please give the medicine by mouth."
    with JARGON_EXAMPLES.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    available = {(row["label"], row["text"]) for row in rows}
    if ("jargon", example_source) not in available or (
        "plain",
        example_plain,
    ) not in available:
        raise RuntimeError(
            "The synthetic jargon examples needed for the AI prompt are missing."
        )
    return example_source, example_plain


def _complete(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Groq is not configured. Add your key to GROQ_API_KEY in the local .env file."
        )
    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": os.getenv("GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=90,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not connect to Groq. Check your internet connection and try again."
        ) from exc
    if not response.ok:
        if response.status_code == 401:
            message = "Groq rejected the API key. Check GROQ_API_KEY in .env."
        elif response.status_code == 429:
            message = "Groq rate limit or quota reached. Check your Groq account and retry."
        elif response.status_code == 404:
            message = "Groq model was not found. Check GROQ_MODEL in .env."
        else:
            message = f"Groq request failed (HTTP {response.status_code})."
        raise RuntimeError(message)
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Groq returned an unexpected response.") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Groq returned an empty response. Please retry.")
    return content.strip()


def simplify_with_groq(
    source: str, language: str = "English", required_facts: list[str] | None = None
) -> str:
    example_source, example_plain = _simplification_example()
    facts_note = (
        "\nDetails that must remain unchanged: " + "; ".join(required_facts)
        if required_facts
        else ""
    )
    prompt = (
        f"Rewrite the instruction in clear, plain {language} for an older adult. "
        "This is a general-purpose task: simplify the meaning of the supplied text "
        "even when its wording or medical terms are not in the example below. The "
        "synthetic example is only a style guide, not a vocabulary limit.\n\n"
        "Style example from the project's synthetic jargon examples:\n"
        f"Original: {example_source}\n"
        f"Plain language: - {example_plain}\n\n"
        "For the actual instruction, use short, easy-to-read bullet notes, usually one "
        "action or warning per bullet. Keep every instruction; do not summarize away "
        "details. Preserve medicine names, amounts, dates, frequencies, durations, "
        "warnings, conditions, and do-not instructions. Do not add advice or infer "
        "missing facts. Briefly explain specialized words in everyday language. "
        "Return only the bullet notes: no title, preamble, conclusion, or commentary."
        f"{facts_note}\n\nInstruction:\n{source}"
    )
    return _complete(prompt)


def translate_with_groq(text: str, language: str) -> str:
    prompt = (
        f"Translate the following verified plain-language text into {language}. "
        "Preserve all numbers, dates, medication names, frequencies, durations, "
        "negations, warnings, and bullet structure exactly. Do not add advice. "
        "Return only the translated bullet notes.\n\n"
        f"{text}"
    )
    return _complete(prompt)
