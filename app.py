"""Accessible Gradio interface for the comprisMED research prototype."""

from __future__ import annotations

import re

import gradio as gr

from comprismed.facts import compare_facts
from comprismed.groq import translate_with_groq
from comprismed.jargon import detect_jargon
from comprismed.pipeline import simplify


EXAMPLE = (
    "Your doctor asked you to take 5 mg of the medicine by mouth twice daily "
    "for 7 days. Do not stop early. Call the clinic if you develop a rash."
)

CSS = """
:root { color-scheme: light; }
html, body, #root, .gradio-container {
    width: 100% !important;
    max-width: none !important;
    min-height: 100% !important;
    margin: 0 !important;
    background: #f5f7fa !important;
    color: #172033 !important;
    font-family: Inter, "Segoe UI", Arial, sans-serif !important;
}
.gradio-container {
    min-height: 100vh !important;
    box-sizing: border-box !important;
    padding: 18px clamp(16px, 3vw, 48px) 32px !important;
}
.gradio-container, .gradio-container * {
    font-family: Inter, "Segoe UI", Arial, sans-serif !important;
}
.cm-shell { width: 100% !important; max-width: none !important; gap: 16px !important; }
.cm-header { padding: 0 0 4px !important; gap: 4px !important; }
.cm-header h1 { color: #111827 !important; font-size: clamp(1.7rem, 3vw, 2.25rem) !important; }
.cm-header p { color: #475569 !important; font-size: 1rem !important; }
.cm-safety, .cm-safety * { color: #26364f !important; }
.cm-safety {
    background: #eaf1ff !important;
    border-left: 4px solid #2563eb !important;
    border-radius: 6px !important;
    padding: 10px 14px !important;
    line-height: 1.5 !important;
}
.block.cm-safety {
    background: transparent !important;
    border-left: 0 !important;
    padding: 0 !important;
}
.cm-workspace {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    align-items: stretch !important;
    gap: 16px !important;
    width: 100% !important;
}
.cm-panel {
    min-width: 0 !important;
    background: #fff !important;
    border: 1px solid #d5dce7 !important;
    border-radius: 14px !important;
    padding: clamp(14px, 2vw, 26px) !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, .04) !important;
}
.cm-panel .block, .cm-panel .form, .cm-panel .container,
.cm-compare .block, .cm-compare .form {
    background: transparent !important;
}
.cm-panel h2, .cm-panel h3, .cm-panel p, .cm-panel label,
.cm-panel .prose, .cm-panel .prose *, .cm-panel .markdown,
.cm-compare h2, .cm-compare h3, .cm-compare p, .cm-compare label,
.cm-compare .prose, .cm-compare .prose * {
    color: #172033 !important;
}
.cm-panel h2, .cm-panel h3 { margin: 0 0 6px !important; }
.cm-helper { color: #526178 !important; font-size: .94rem !important; line-height: 1.5 !important; }
.cm-panel textarea {
    min-height: 42vh !important;
    resize: vertical !important;
    background: #fff !important;
    color: #111827 !important;
    border: 1px solid #b8c2d1 !important;
    border-radius: 10px !important;
    font-size: 1.05rem !important;
    line-height: 1.65 !important;
}
.cm-panel textarea::placeholder { color: #64748b !important; opacity: 1 !important; }
.cm-panel input {
    background: #fff !important;
    color: #172033 !important;
}
.cm-panel div.wrap:has(input) {
    background: #fff !important;
    border: 1px solid #b8c2d1 !important;
}
.cm-output textarea {
    min-height: 29vh !important;
    background: #f8faff !important;
    color: #111827 !important;
    border-color: #b7c8e8 !important;
    font-size: 1.08rem !important;
    line-height: 1.8 !important;
}
.cm-translation textarea {
    min-height: 12vh !important;
    background: #fff !important;
    color: #111827 !important;
    border-color: #b8c2d1 !important;
}
.cm-checks {
    background: #f8faff !important;
    border: 1px solid #d7e2f5 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
}
.cm-checks, .cm-checks * { color: #172033 !important; }
.cm-actions { align-items: center !important; gap: 10px !important; }
button { min-height: 46px !important; }
button.primary {
    background: #2457c5 !important;
    color: #fff !important;
    border: 1px solid #2457c5 !important;
}
button.primary:hover { background: #1947a8 !important; }
button.secondary, button:not(.primary):not(.label-wrap) {
    background: #fff !important;
    color: #172033 !important;
    border: 1px solid #b8c2d1 !important;
}
button.secondary:hover, button:not(.primary):not(.label-wrap):hover {
    background: #eef4ff !important;
}
button.label-wrap, button.label-wrap * {
    background: #fff !important;
    color: #172033 !important;
}
.cm-compare {
    width: 100% !important;
    background: #fff !important;
    border: 1px solid #d5dce7 !important;
    border-radius: 14px !important;
    padding: 12px 18px !important;
}
.cm-compare textarea { color: #111827 !important; background: #f8faff !important; }
.cm-safety blockquote { border-left: 0 !important; padding-left: 0 !important; }
footer, .built-with-gradio { display: none !important; }
@media (max-width: 760px) {
    .gradio-container { padding: 12px !important; }
    .cm-workspace { grid-template-columns: minmax(0, 1fr) !important; }
    .cm-panel textarea { min-height: 26vh !important; }
    .cm-output textarea { min-height: 24vh !important; }
}
"""


def _format_notes(text: str) -> str:
    """Keep the model's content, presenting each returned line as a jot note."""
    cleaned = text.strip().replace("```text", "").replace("```", "").strip()
    preambles = {
        "here are your steps:",
        "here are the steps:",
        "here are the simplified instructions:",
        "here is the simplified version:",
        "simplified version:",
        "plain-language version:",
        "plain language version:",
    }
    lines = []
    for raw_line in cleaned.splitlines():
        raw_line = re.sub(
            r"\s+(?=(?:[-*•]\s+|\d+[.)]\s+))", "\n", raw_line
        )
        for line in raw_line.splitlines():
            line = line.strip()
            line = re.sub(r"^#{1,6}\s*", "", line)
            line = re.sub(r"^(?:[-*•]\s*|\d+[.)]\s*)", "", line).strip()
            line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
            line = re.sub(r"\*(.*?)\*", r"\1", line)
            line = line.replace("`", "")
            if line and line.casefold() not in preambles:
                lines.append(line)
    if not lines and cleaned:
        lines = [cleaned]
    return "\n".join(f"- {line}" for line in lines)


def _format_checks(checks: list[object]) -> str:
    if not checks:
        return "No supported numbers, dates, frequencies, durations, or do-not details were found to compare."
    labels = {"preserved": "Preserved", "omitted": "Not found", "altered": "Changed"}
    categories = {
        "number_or_dose": "Dose or amount",
        "number": "Dose or amount",
        "frequency": "How often",
        "duration": "How long",
        "date": "Date",
        "negation": "Do-not wording",
    }
    rows = []
    for check in checks:
        status = getattr(check, "status", "")
        category = categories.get(
            str(getattr(check, "category", "")),
            str(getattr(check, "category", "")).replace("_", " ").title(),
        )
        fact = str(getattr(check, "source_fact", ""))
        marker = "✓" if status == "preserved" else "!"
        rows.append(f"- {marker} **{labels.get(status, 'Review')}** · {category}: {fact}")
    return "\n".join(rows)


def run_demo(text: str) -> tuple[str, str, str, str, str, dict[str, object]]:
    if not text or not text.strip():
        raise gr.Error("Enter the instructions first.")
    try:
        baseline_result = simplify(text, mode="baseline", use_api=True)
        verified_result = simplify(text, mode="verified", use_api=True)
        jargon = detect_jargon(text)
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    baseline = _format_notes(str(baseline_result["output"]))
    verified = _format_notes(str(verified_result["output"]))
    source_facts = verified_result["source_facts"]
    baseline_checks = compare_facts(text, str(baseline_result["output"]), source_facts)
    verified_checks = verified_result["checks"]
    needs_review = bool(verified_result["review_required"])

    if needs_review:
        check_summary = (
            "### Please compare these details with the original\n"
            "Some supported details may be missing or changed. Contact your care team "
            "if anything is unclear.\n\n"
            f"{_format_checks(verified_checks)}"
        )
    elif verified_checks:
        check_summary = (
            "### Checked details\n"
            "No changes found in the details listed below. This does not confirm the "
            "explanation is medically correct.\n\n"
            f"{_format_checks(verified_checks)}"
        )
    else:
        check_summary = (
            "### Nothing was available to check\n"
            "The checker found no supported numbers, dates, schedules, durations, or "
            "do-not details. This does not confirm the explanation is correct."
        )
    jargon_matches = jargon["term_matches"]
    jargon_summary = (
        "\n".join(f"- **{term}**: {definition}" for term, definition in jargon_matches)
        if jargon_matches
        else "No glossary matches were found."
    )
    return (
        verified,
        check_summary,
        baseline,
        _format_checks(baseline_checks),
        jargon_summary,
        gr.update(interactive=True),
    )


def translate_result(text: str, language: str) -> str:
    if not text or not text.strip():
        raise gr.Error("Simplify the instructions before translating.")
    if not language:
        raise gr.Error("Choose a language.")
    try:
        return _format_notes(translate_with_groq(text, language))
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc


with gr.Blocks(
    title="comprisMED",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "Arial", "sans-serif"],
    ),
    css=CSS,
    analytics_enabled=False,
) as demo:
    with gr.Column(elem_classes=["cm-shell"]):
        with gr.Column(elem_classes=["cm-header"]):
            gr.Markdown("# comprisMED")
            gr.Markdown(
                "Make medical instructions easier to read, then check selected details."
            )
            gr.Markdown(
                "**Safety note:** This project is a prototype and is not medical advice. "
                "Use fictional text only; do not enter personal health information.",
                elem_classes=["cm-safety"],
            )

        with gr.Row(elem_classes=["cm-workspace"]):
            with gr.Column(elem_classes=["cm-panel"]):
                gr.Markdown("### Original instructions")
                gr.Markdown(
                    "Paste or type the text you want explained.",
                    elem_classes=["cm-helper"],
                )
                instructions = gr.Textbox(
                    label="",
                    placeholder="Enter fictional instructions here...",
                    lines=12,
                    max_lines=24,
                    show_label=False,
                )
                with gr.Row(elem_classes=["cm-actions"]):
                    example_button = gr.Button("Use example", size="sm")
                    simplify_button = gr.Button(
                        "Simplify instructions", variant="primary", size="lg"
                    )

            with gr.Column(elem_classes=["cm-panel"]):
                gr.Markdown("### Simple version")
                gr.Markdown(
                    "The AI turns the instructions into short, easy-to-read steps.",
                    elem_classes=["cm-helper"],
                )
                simplified_output = gr.Textbox(
                    label="",
                    placeholder="Your simplified steps will appear here.",
                    lines=8,
                    max_lines=18,
                    interactive=False,
                    show_label=False,
                    elem_classes=["cm-output"],
                )
                checks_output = gr.Markdown(
                    "Important detail checks will appear here.",
                    elem_classes=["cm-checks"],
                )

                gr.Markdown("### Translate")
                with gr.Row(elem_classes=["cm-actions"]):
                    language = gr.Dropdown(
                        choices=[
                            "Arabic", "Chinese (Simplified)", "French", "German",
                            "Hindi", "Italian", "Korean", "Portuguese", "Russian",
                            "Spanish", "Tagalog", "Vietnamese",
                        ],
                        value="Spanish",
                        label="Language",
                        scale=2,
                    )
                    translate_button = gr.Button(
                        "Translate", scale=1, interactive=False
                    )
                translation_output = gr.Textbox(
                    label="Translation",
                    placeholder="Your translation will appear here.",
                    lines=4,
                    max_lines=10,
                    interactive=False,
                    elem_classes=["cm-translation"],
                )
                gr.Markdown(
                    "AI translations are not independently checked.",
                    elem_classes=["cm-helper"],
                )

        with gr.Accordion(
            "Compare AI-only and AI + detail check", open=False,
            elem_classes=["cm-compare"],
        ):
            gr.Markdown(
                "Both versions are AI-generated. The checked version may be revised "
                "if a selected detail does not match."
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### AI-only version")
                    baseline_output = gr.Textbox(
                        label="",
                        lines=5,
                        interactive=False,
                        show_label=False,
                    )
                    baseline_checks_output = gr.Markdown()
        with gr.Accordion(
            "Possible medical terms", open=False, elem_classes=["cm-compare"]
        ):
            jargon_output = gr.Markdown("Term explanations will appear here.")

    example_button.click(lambda: EXAMPLE, outputs=instructions)
    simplify_button.click(
        run_demo,
        inputs=instructions,
        outputs=[
            simplified_output,
            checks_output,
            baseline_output,
            baseline_checks_output,
            jargon_output,
            translate_button,
        ],
        show_progress="full",
    )
    translate_button.click(
        translate_result,
        inputs=[simplified_output, language],
        outputs=translation_output,
        show_progress="full",
    )


if __name__ == "__main__":
    demo.launch(show_api=False)
