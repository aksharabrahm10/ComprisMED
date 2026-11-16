from types import SimpleNamespace

import pytest

from comprismed import groq


def test_groq_client_posts_to_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "local-test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> SimpleNamespace:
        captured["url"] = url
        captured.update(kwargs)
        return SimpleNamespace(
            ok=True,
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "Plain-language draft"}}]},
        )

    monkeypatch.setattr(groq.requests, "post", fake_post)
    result = groq.simplify_with_groq("Synthetic source", required_facts=["5 mg"])

    assert result == "Plain-language draft"
    assert captured["url"] == groq.API_URL
    assert captured["json"]["model"] == "test-model"
    assert captured["headers"]["Authorization"] == "Bearer local-test-key"
    prompt = captured["json"]["messages"][0]["content"]
    assert "5 mg" in prompt
    assert "Please administer the medication orally." in prompt
    assert "Please give the medicine by mouth." in prompt
    assert "general-purpose task" in prompt
    assert "bullet notes" in prompt


def test_groq_client_reports_missing_key_without_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        groq.simplify_with_groq("Synthetic source")


def test_groq_client_reports_invalid_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "local-test-key")
    monkeypatch.setattr(
        groq.requests,
        "post",
        lambda *args, **kwargs: SimpleNamespace(ok=False, status_code=401),
    )
    with pytest.raises(RuntimeError, match="rejected the API key"):
        groq.simplify_with_groq("Synthetic source")


def test_groq_client_retries_transient_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "local-test-key")
    responses = iter(
        [
            SimpleNamespace(
                ok=False,
                status_code=429,
                headers={"retry-after": "2"},
            ),
            SimpleNamespace(
                ok=True,
                status_code=200,
                headers={},
                json=lambda: {"choices": [{"message": {"content": "Recovered"}}]},
            ),
        ]
    )
    waits: list[float] = []
    monkeypatch.setattr(groq.requests, "post", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(groq.time, "sleep", waits.append)

    assert groq._complete("Synthetic request") == "Recovered"
    assert waits == [2.0]


def test_groq_client_reports_specific_limit_without_long_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "local-test-key")
    waits: list[float] = []
    monkeypatch.setattr(
        groq.requests,
        "post",
        lambda *args, **kwargs: SimpleNamespace(
            ok=False,
            status_code=429,
            headers={
                "retry-after": "2m",
                "x-ratelimit-remaining-tokens": "0",
                "x-ratelimit-reset-tokens": "2m",
            },
        ),
    )
    monkeypatch.setattr(groq.time, "sleep", waits.append)

    with pytest.raises(RuntimeError, match="token-per-minute limit.*2m"):
        groq._complete("Synthetic request")
    assert waits == []


def test_repair_prompt_uses_draft_and_only_failed_source_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "local-test-key")
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            ok=True,
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "Corrected notes"}}]},
        )

    monkeypatch.setattr(groq.requests, "post", fake_post)
    result = groq.repair_with_groq(
        "Take 5 mg twice daily.",
        "Take 10 mg twice daily.",
        {"number_or_dose": ["5 mg"]},
    )
    prompt = captured["json"]["messages"][0]["content"]
    assert result == "Corrected notes"
    assert "Previous plain-language draft" in prompt
    assert "Take 10 mg twice daily." in prompt
    assert "number_or_dose: 5 mg" in prompt
    assert "frequency:" not in prompt


def test_repair_requires_at_least_one_detail() -> None:
    with pytest.raises(ValueError, match="At least one"):
        groq.repair_with_groq("Source", "Draft", {})


def test_translation_prompt_requires_all_text_to_be_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "local-test-key")
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            ok=True,
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "Notas traducidas"}}]},
        )

    monkeypatch.setattr(groq.requests, "post", fake_post)
    result = groq.translate_with_groq("- Take 5 mg twice daily.", "Spanish")
    prompt = captured["json"]["messages"][0]["content"]

    assert result == "Notas traducidas"
    assert "Write the entire answer in Spanish" in prompt
    assert "Do not copy English phrases" in prompt
    assert "words for frequencies and durations" in prompt
    assert "- Take 5 mg twice daily." in prompt
