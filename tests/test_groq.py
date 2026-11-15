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
