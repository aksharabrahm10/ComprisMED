"""Groq Chat Completions client; credentials are read from the local environment."""

from __future__ import annotations

import csv
import os
import re
import time
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
MAX_RATE_LIMIT_RETRIES = 3
MAX_RATE_LIMIT_WAIT_SECONDS = 60


def _delay_from_header(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass

    units = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", value.casefold())
    if not parts:
        return None
    return sum(float(amount) * units[unit] for amount, unit in parts)


class _GroqRateLimitError(RuntimeError):
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers
        self.retry_after = _delay_from_header(
            headers.get("retry-after")
            or headers.get("x-ratelimit-reset-tokens")
            or headers.get("x-ratelimit-reset-requests")
        )
        super().__init__(self.message)

    @property
    def message(self) -> str:
        if self.headers.get("x-ratelimit-remaining-tokens") == "0":
            limit = "Groq's token-per-minute limit"
        elif self.headers.get("x-ratelimit-remaining-requests") == "0":
            limit = "Groq's request-per-day limit"
        else:
            limit = "Groq's rate limit or quota"
        wait = (
            self.headers.get("retry-after")
            or self.headers.get("x-ratelimit-reset-tokens")
            or self.headers.get("x-ratelimit-reset-requests")
        )
        detail = f" Wait at least {wait} before retrying." if wait else ""
        return (
            f"{limit} was reached (HTTP 429).{detail} Check your Groq account "
            "limits and rerun the experiment."
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


def _complete_once(prompt: str) -> str:
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
            raise _GroqRateLimitError(dict(getattr(response, "headers", {})))
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


def _complete(prompt: str) -> str:
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return _complete_once(prompt)
        except _GroqRateLimitError as exc:
            if attempt == MAX_RATE_LIMIT_RETRIES:
                raise RuntimeError(exc.message) from exc
            delay = exc.retry_after
            if delay is None:
                delay = min(2.0**attempt, MAX_RATE_LIMIT_WAIT_SECONDS)
            if delay > MAX_RATE_LIMIT_WAIT_SECONDS:
                raise RuntimeError(exc.message) from exc
            time.sleep(delay)
    raise RuntimeError("Groq request ended without a response.")


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


def repair_with_groq(
    source: str,
    draft: str,
    required_facts: dict[str, list[str]],
) -> str:
    details = [
        f"{category}: {fact}"
        for category, facts in required_facts.items()
        for fact in facts
    ]
    if not details:
        raise ValueError("At least one required source detail is needed for repair.")
    prompt = (
        "Correct the plain-language version of this fictional medical instruction. "
        "Use the source as the authority. Keep every instruction, warning, condition, "
        "restriction, medicine name, amount, date, frequency, duration, and negation. "
        "Do not add advice or infer missing facts. Write short, easy-to-read bullet "
        "notes and return only the corrected notes.\n\n"
        f"Source instruction:\n{source}\n\n"
        f"Previous plain-language draft:\n{draft}\n\n"
        "The source details that the checker could not confirm are:\n"
        + "\n".join(f"- {detail}" for detail in details)
    )
    return _complete(prompt)


def simplify_summary_with_groq(source: str) -> str:
    example_source, example_plain = _simplification_example()
    prompt = (
        "Rewrite this medical research-summary paragraph in plain language for a "
        "general reader. Preserve the findings, numbers, comparisons, uncertainty, "
        "and limitations in the source. Do not turn research findings into personal "
        "medical advice or instructions, and do not invent conclusions. Use short "
        "bullet notes and return only the notes.\n\n"
        "Style example from the project's synthetic jargon examples:\n"
        f"Original: {example_source}\n"
        f"Plain language: - {example_plain}\n\n"
        f"Research summary:\n{source}"
    )
    return _complete(prompt)


def repair_summary_with_groq(
    source: str,
    draft: str,
    required_facts: dict[str, list[str]],
) -> str:
    details = [
        f"{category}: {fact}"
        for category, facts in required_facts.items()
        for fact in facts
    ]
    if not details:
        raise ValueError("At least one required source detail is needed for repair.")
    prompt = (
        "Correct this plain-language medical research-summary rewrite using only the "
        "source as evidence. Preserve the source's findings, numbers, comparisons, "
        "uncertainty, and limitations. Do not turn research into personal medical "
        "advice, add conclusions, or omit the details listed by the checker. Return "
        "only short, clear bullet notes.\n\n"
        f"Source research summary:\n{source}\n\n"
        f"Previous plain-language draft:\n{draft}\n\n"
        "The source details that the checker could not confirm are:\n"
        + "\n".join(f"- {detail}" for detail in details)
    )
    return _complete(prompt)


def translate_with_groq(text: str, language: str) -> str:
    prompt = (
        f"Write the entire answer in {language}. Translate every ordinary word "
        "and phrase, including instruction verbs, warnings, negations, and words "
        "for frequencies and durations. Do not copy English phrases into the "
        f"{language} translation. Leave medication names, numbers, and standard "
        "dosage units unchanged; preserve all facts and bullet order. Do not add "
        "advice. Before answering, check that every sentence is in "
        f"{language}, except for medication names, numbers, and dosage units. "
        "Return only the translated bullet notes.\n\n"
        f"{text}"
    )
    return _complete(prompt)
