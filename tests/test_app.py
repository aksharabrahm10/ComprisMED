from __future__ import annotations

from comprismed.facts import Check


def test_run_demo_shows_only_actionable_detail_warnings(monkeypatch):
    import app

    source = "Take 5 mg twice daily for 7 days."
    monkeypatch.setattr(
        app,
        "simplify",
        lambda text, **kwargs: {
            "output": source,
            "checks": [
                Check("number_or_dose", "5 mg", "preserved", "5 mg"),
                Check("frequency", "twice daily", "preserved", "twice daily"),
                Check("duration", "7 days", "preserved", "7 days"),
            ],
        },
    )

    simplified, review, translate_update = app.run_demo(source)

    assert simplified == "- Take 5 mg twice daily for 7 days."
    assert review == (
        "No changes were detected in the important details we checked."
    )
    assert "Preserved" not in review
    assert translate_update["interactive"] is True


def test_run_demo_flags_missing_and_changed_details(monkeypatch):
    import app

    source = "Take 5 mg twice daily."
    monkeypatch.setattr(
        app,
        "simplify",
        lambda *_args, **_kwargs: {
            "output": "Take 10 mg.",
            "checks": [
                Check("number_or_dose", "5 mg", "altered", "10 mg"),
                Check("frequency", "twice daily", "omitted"),
            ],
        },
    )

    _, review, _ = app.run_demo(source)

    assert "**Changed** · Dose or amount: 5 mg" in review
    assert "output says “10 mg”" in review
    assert "**Missing** · How often: twice daily" in review
    assert "Preserved" not in review


def test_run_demo_handles_instructions_without_checkable_details(monkeypatch):
    import app

    source = "Your appointment is next week."
    monkeypatch.setattr(
        app,
        "simplify",
        lambda *_args, **_kwargs: {
            "output": source,
            "checks": [],
        },
    )

    _, review, _ = app.run_demo(source)

    assert review == "No specific details were available to check."


def test_run_demo_uses_verified_simplification(monkeypatch):
    import app

    calls = []

    def fake_simplify(text, **kwargs):
        calls.append((text, kwargs))
        return {
            "output": "Take 5 mg twice daily.",
            "checks": [Check("number_or_dose", "5 mg", "preserved", "5 mg")],
        }

    monkeypatch.setattr(app, "simplify", fake_simplify)

    app.run_demo("Take 5 mg twice daily.")

    assert calls == [
        ("Take 5 mg twice daily.", {"mode": "verified", "use_api": True})
    ]


def test_translation_requires_simplified_text(monkeypatch):
    import gradio as gr

    import app

    monkeypatch.setattr(app, "translate_with_groq", lambda *_args: "translated")
    try:
        app.translate_result("", "Spanish")
    except gr.Error as exc:
        assert "Simplify the instructions" in str(exc)
    else:
        raise AssertionError("Translation should require simplified text.")

    assert app.translate_result("Draft", "Spanish") == "- translated"


def test_run_demo_rejects_empty_instructions():
    import gradio as gr

    import app

    try:
        app.run_demo("  ")
    except gr.Error as exc:
        assert "Enter the instructions first" in str(exc)
    else:
        raise AssertionError("Empty instructions should be rejected.")


def test_format_notes_displays_only_bullet_steps():
    import app

    assert app._format_notes(
        "Here are your steps:\n1. Take one pill.\n- Call if you feel dizzy."
    ) == "- Take one pill.\n- Call if you feel dizzy."
    assert app._format_notes(
        "- Take one pill. - Call if you feel dizzy."
    ) == "- Take one pill.\n- Call if you feel dizzy."
