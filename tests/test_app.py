from __future__ import annotations

from comprismed.facts import Check, extract_facts


def test_run_demo_shows_review_state_and_checks(monkeypatch):
    import app

    monkeypatch.setattr(
        app,
        "simplify",
        lambda *_args, **_kwargs: {
            "initial_output": "Take 5 mg twice daily for 7 days.",
            "output": "Take 5 mg twice daily for 7 days.",
            "source_facts": extract_facts("Take 5 mg twice daily for 7 days."),
            "checks": [Check("number", "5 mg", "preserved", "5 mg")],
            "review_required": False,
        },
    )
    monkeypatch.setattr(
        app,
        "detect_jargon",
        lambda _text: {"term_matches": [("oral", "by mouth")]},
    )

    simplified, review, baseline, baseline_review, jargon, translate_update = (
        app.run_demo("Take 5 mg twice daily for 7 days.")
    )

    assert simplified == "- Take 5 mg twice daily for 7 days."
    assert baseline == simplified
    assert "Preserved" in baseline_review
    assert "No changes found" in review
    assert "does not confirm" in review
    assert "oral" in jargon
    assert translate_update["interactive"] is True


def test_run_demo_flags_unresolved_details(monkeypatch):
    import app

    monkeypatch.setattr(
        app,
        "simplify",
        lambda *_args, **_kwargs: {
            "initial_output": "Take your medicine.",
            "output": "Take your medicine.",
            "source_facts": extract_facts("Take medicine twice daily."),
            "checks": [Check("frequency", "twice daily", "omitted")],
            "review_required": True,
        },
    )
    monkeypatch.setattr(
        app, "detect_jargon", lambda _text: {"term_matches": []}
    )

    _, review, _, _, _, _ = app.run_demo("Take medicine twice daily.")

    assert "Please compare these details" in review
    assert "Not found" in review


def test_run_demo_compares_first_draft_with_repaired_output(monkeypatch):
    import app

    source = "Take 5 mg twice daily for 7 days."
    results = iter([
        {"output": "Take your medicine regularly.",
         "source_facts": extract_facts(source), "checks": [],
         "review_required": False},
        {
            "output": "Take 5 mg twice daily for 7 days.",
            "source_facts": extract_facts(source),
            "checks": [
                Check("number_or_dose", "5 mg", "preserved", "5 mg"),
                Check("frequency", "twice daily", "preserved", "twice daily"),
                Check("duration", "7 days", "preserved", "7 days"),
            ],
            "review_required": False,
        },
    ])
    monkeypatch.setattr(app, "simplify", lambda *_args, **_kwargs: next(results))
    monkeypatch.setattr(
        app, "detect_jargon", lambda _text: {"term_matches": []}
    )

    verified, final_checks, baseline, baseline_checks, _, _ = app.run_demo(source)

    assert baseline == "- Take your medicine regularly."
    assert verified == "- Take 5 mg twice daily for 7 days."
    assert "Not found" in baseline_checks
    assert "No changes found" in final_checks


def test_run_demo_does_not_claim_success_when_no_details_are_checked(monkeypatch):
    import app

    source = "Your appointment is next week."
    monkeypatch.setattr(
        app,
        "simplify",
        lambda *_args, **_kwargs: {
            "initial_output": source,
            "output": source,
            "source_facts": extract_facts(source),
            "checks": [],
            "review_required": False,
        },
    )
    monkeypatch.setattr(
        app, "detect_jargon", lambda _text: {"term_matches": []}
    )

    _, review, _, _, _, _ = app.run_demo(source)

    assert "Nothing was available to check" in review
    assert "No changes found" not in review


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


def test_format_notes_displays_only_bullet_steps():
    import app

    assert app._format_notes(
        "Here are your steps:\n1. Take one pill.\n- Call if you feel dizzy."
    ) == "- Take one pill.\n- Call if you feel dizzy."
    assert app._format_notes(
        "- Take one pill. - Call if you feel dizzy."
    ) == "- Take one pill.\n- Call if you feel dizzy."
