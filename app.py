"""Accessible Gradio interface for the comprisMED research prototype."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import gradio as gr

from comprismed.groq import translate_with_groq
from comprismed.ocr import (
    ImageInputError,
    extract_instruction_text,
    prepare_image_upload,
)
from comprismed.pipeline import simplify


EXAMPLE = (
    "Your doctor asked you to take 5 mg of the medicine by mouth twice daily "
    "for 7 days. Do not stop early. Call the clinic if you develop a rash."
)

CSS = """
:root { color-scheme: light; }
html, body, #root, .gradio-container, .gradio-container * {
    font-family: Inter, "Segoe UI", Arial, sans-serif !important;
}
html, body, #root, .gradio-container {
    width: 100% !important;
    max-width: none !important;
    min-height: 100% !important;
    margin: 0 !important;
    background: #f7f5ef !important;
    color: #1d3440 !important;
}
.gradio-container {
    min-height: 100vh !important;
    box-sizing: border-box !important;
    padding: clamp(18px, 4vw, 52px) clamp(16px, 6vw, 88px) 48px !important;
}
.cm-shell {
    width: 100% !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
    gap: clamp(18px, 3vw, 30px) !important;
}
.cm-header {
    display: flex !important;
    align-items: center !important;
    padding: 0 0 2px !important;
}
.cm-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}
.cm-brand-mark {
    display: flex;
    width: 52px;
    height: 52px;
    flex: 0 0 52px;
    align-items: center;
    justify-content: center;
    border-radius: 16px;
    background: #dcebe6;
}
.cm-brand h1 {
    margin: 0;
    color: #1d3440;
    font-size: clamp(2rem, 4vw, 2.65rem);
    font-weight: 700;
    letter-spacing: -.055em;
    line-height: 1.1;
}
.cm-safety {
    color: #344c53 !important;
    background: #e9eee8 !important;
    border: 1px solid #d5e0d9 !important;
    border-radius: 12px !important;
    padding: 13px 17px !important;
    font-size: 1rem !important;
    line-height: 1.55 !important;
}
.cm-safety, .cm-safety * { color: #344c53 !important; }
.cm-workspace {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    align-items: stretch !important;
    gap: clamp(16px, 2vw, 24px) !important;
    width: 100% !important;
}
.cm-panel {
    min-width: 0 !important;
    background: #fffefa !important;
    border: 1px solid #e6e2d8 !important;
    border-radius: 20px !important;
    padding: clamp(20px, 3vw, 34px) !important;
    box-shadow: 0 8px 26px rgba(29, 52, 64, .045) !important;
}
.cm-panel .block, .cm-panel .form, .cm-panel .container {
    background: transparent !important;
}
.cm-panel h2, .cm-panel h3, .cm-panel p, .cm-panel label,
.cm-panel .prose, .cm-panel .prose *, .cm-panel .markdown {
    color: #1d3440 !important;
}
.cm-panel h2, .cm-panel h3 {
    margin: 0 0 16px !important;
    font-size: 1.35rem !important;
    font-weight: 650 !important;
    letter-spacing: -.02em !important;
}
.cm-panel textarea {
    min-height: 34vh !important;
    resize: vertical !important;
    background: #fffefa !important;
    color: #1d3440 !important;
    border: 1px solid #d8d8cf !important;
    border-radius: 12px !important;
    font-size: 1.08rem !important;
    line-height: 1.75 !important;
    padding: 16px !important;
}
.cm-panel textarea::placeholder { color: #687a79 !important; opacity: 1 !important; }
.cm-output textarea,
.cm-output textarea:disabled {
    min-height: 25vh !important;
    background: #1d3440 !important;
    color: #fffefa !important;
    -webkit-text-fill-color: #fffefa !important;
    opacity: 1 !important;
    border: 1.5px solid #477a73 !important;
    line-height: 1.9 !important;
    font-size: 1.1rem !important;
    font-weight: 500 !important;
    box-shadow: inset 0 0 0 1px rgba(255, 254, 250, .06) !important;
}
.cm-translation textarea,
.cm-translation textarea:disabled {
    min-height: 12vh !important;
    background: #1d3440 !important;
    color: #fffefa !important;
    -webkit-text-fill-color: #fffefa !important;
    opacity: 1 !important;
    border: 1.5px solid #477a73 !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
}
.cm-output textarea::placeholder,
.cm-translation textarea::placeholder {
    color: #cfddd8 !important;
    -webkit-text-fill-color: #cfddd8 !important;
    opacity: 1 !important;
}
.cm-panel label, .cm-panel .wrap label {
    color: #344c53 !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}
.cm-panel input:not([type="checkbox"]), .cm-panel [role="combobox"] {
    min-height: 52px !important;
    background: #fffefa !important;
    color: #1d3440 !important;
    border: 1px solid #d8d8cf !important;
    border-radius: 12px !important;
    font-size: 1.05rem !important;
}
.cm-checks {
    margin-top: 16px !important;
    background: #f7f5ef !important;
    border: 1px solid #e6e2d8 !important;
    border-radius: 12px !important;
    padding: 14px 17px !important;
}
.cm-checks, .cm-checks * {
    color: #344c53 !important;
    font-size: 1rem !important;
    line-height: 1.65 !important;
}
.cm-checks:has(.md:empty) { display: none !important; }
.cm-checks p { margin: 0 !important; }
.cm-checks:has(p:only-child) {
    background: transparent !important;
    border-color: transparent !important;
    padding: 4px 0 !important;
}
.cm-description {
    margin-top: 6px !important;
    color: #536a68 !important;
    font-size: .94rem !important;
    line-height: 1.6 !important;
    text-align: center !important;
}
.cm-description, .cm-description * { color: #536a68 !important; }
.cm-input-actions { gap: 12px !important; }
.cm-upload-title {
    margin: 20px 0 10px !important;
    color: #344c53 !important;
    font-size: 1rem !important;
    font-weight: 650 !important;
}
.cm-upload-row { align-items: stretch !important; gap: 14px !important; }
.cm-photo-upload, .cm-photo-preview.block {
    min-width: 0 !important;
    border: 1px solid #d8d8cf !important;
    border-radius: 12px !important;
    background: #f7f5ef !important;
    color: #1d3440 !important;
}
.cm-photo-preview.prose {
    min-height: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}
.cm-photo-upload label, .cm-photo-upload .file-preview,
.cm-photo-upload .file-preview * {
    color: #1d3440 !important;
}
.cm-photo-upload button,
.cm-photo-upload button * {
    color: #1d3440 !important;
}
.cm-photo-upload .or {
    color: #536a68 !important;
}
.cm-photo-preview {
    display: flex !important;
    min-height: 120px !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
}
.cm-photo-preview.block:not(:has(img)) {
    display: none !important;
    min-height: 0 !important;
    padding: 0 !important;
    border: 0 !important;
}
.cm-photo-preview.block:has(img) {
    display: flex !important;
    min-height: 120px !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
}
.cm-photo-preview.prose img {
    display: block !important;
    max-width: 100% !important;
    max-height: 230px !important;
    object-fit: contain !important;
    border-radius: 8px !important;
}
.cm-photo-status {
    margin: 8px 0 !important;
    color: #344c53 !important;
    font-size: .96rem !important;
    line-height: 1.55 !important;
}
.cm-photo-confirm {
    margin: 6px 0 12px !important;
    padding: 10px 12px !important;
    border-radius: 10px !important;
    background: #e9eee8 !important;
}
.cm-photo-confirm, .cm-photo-confirm * { color: #1d3440 !important; }
.cm-photo-confirm input[type="checkbox"] {
    width: 24px !important;
    min-width: 24px !important;
    height: 24px !important;
    min-height: 24px !important;
    padding: 0 !important;
    accent-color: #477a73 !important;
    opacity: 1 !important;
    visibility: visible !important;
}
.cm-photo-confirm input[type="checkbox"]:checked {
    background-color: #477a73 !important;
}
.cm-language-select {
    margin-top: 20px !important;
}
.cm-language-select [role="combobox"] {
    min-height: 54px !important;
    background: #fffefa !important;
    color: #1d3440 !important;
    border: 1px solid #aebeb6 !important;
    border-radius: 12px !important;
    font-size: 1.05rem !important;
}
.cm-language-select .container,
.cm-language-select .wrap,
.cm-language-select .wrap-inner,
.cm-language-select .secondary-wrap {
    background: transparent !important;
}
.cm-language-select input[role="listbox"] {
    min-height: 54px !important;
    background: #fffefa !important;
    color: #1d3440 !important;
    -webkit-text-fill-color: #1d3440 !important;
    border: 1px solid #aebeb6 !important;
    border-radius: 12px !important;
    font-size: 1.05rem !important;
    padding: 12px 44px 12px 16px !important;
}
.cm-language-select .icon-wrap {
    display: flex !important;
    width: 40px !important;
    align-items: center !important;
    justify-content: center !important;
    background: transparent !important;
}
.cm-language-select .icon-wrap svg { display: none !important; }
.cm-language-select .icon-wrap::after {
    width: 8px;
    height: 8px;
    margin-top: -4px;
    border-right: 2px solid #1d3440;
    border-bottom: 2px solid #1d3440;
    content: "";
    transform: rotate(45deg);
}
.cm-language-select [data-testid="block-info"] {
    display: block !important;
    padding: 0 0 7px !important;
    background: transparent !important;
    color: #344c53 !important;
    border: 0 !important;
    border-radius: 0 !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}
.cm-language-select [role="option"] {
    color: #1d3440 !important;
    font-size: 1rem !important;
}
#dropdown-options,
#dropdown-options *,
ul.options[role="listbox"],
ul.options[role="listbox"] * {
    color: #1d3440 !important;
}
ul.options[role="listbox"] {
    background: #fffefa !important;
    border: 1px solid #d8d8cf !important;
}
ul.options[role="listbox"] [role="option"] {
    background: #fffefa !important;
}
ul.options[role="listbox"] [role="option"][aria-selected="true"],
ul.options[role="listbox"] [role="option"]:hover {
    background: #e9eee8 !important;
}
button {
    min-height: 54px !important;
    border-radius: 12px !important;
    font-size: 1.05rem !important;
    font-weight: 650 !important;
}
button.primary {
    background: #477a73 !important;
    color: #fffefa !important;
    border: 1px solid #477a73 !important;
}
button.primary:hover { background: #37665f !important; }
button.secondary, button:not(.primary):not(.label-wrap) {
    background: #fffefa !important;
    color: #28444a !important;
    border: 1px solid #ccd7d0 !important;
}
button.secondary:hover, button:not(.primary):not(.label-wrap):hover { background: #edf3ee !important; }
button:focus-visible, textarea:focus-visible, input:focus-visible {
    outline: 3px solid #79a69d !important;
    outline-offset: 2px !important;
}
footer, .built-with-gradio { display: none !important; }
@media (max-width: 760px) {
    .gradio-container { padding: 20px 14px 32px !important; }
    .cm-workspace { grid-template-columns: minmax(0, 1fr) !important; }
    .cm-panel { padding: 20px !important; }
    .cm-panel textarea { min-height: 25vh !important; }
    .cm-output textarea { min-height: 22vh !important; }
    .cm-brand-mark { width: 46px; height: 46px; flex-basis: 46px; }
    .cm-language-select { margin-top: 14px !important; }
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
    issues = [check for check in checks if getattr(check, "status", "") != "preserved"]
    if not checks:
        return "No specific details were available to check."
    if not issues:
        return "No changes were detected in the important details we checked."
    labels = {"omitted": "Missing", "altered": "Changed"}
    categories = {
        "number_or_dose": "Dose or amount",
        "number": "Dose or amount",
        "frequency": "How often",
        "duration": "How long",
        "date": "Date",
        "negation": "Do-not wording",
    }
    rows = []
    for check in issues:
        status = getattr(check, "status", "")
        category = categories.get(
            str(getattr(check, "category", "")),
            str(getattr(check, "category", "")).replace("_", " ").title(),
        )
        fact = str(getattr(check, "source_fact", ""))
        matched_fact = getattr(check, "matched_fact", None)
        detail = f" — output says “{matched_fact}”" if matched_fact else ""
        rows.append(
            f"- **{labels.get(status, 'Review')}** · {category}: {fact}{detail}"
        )
    return "\n".join(rows)


def run_demo(
    text: str,
    extracted_text_needs_review: bool = False,
    extracted_text_confirmed: bool = False,
    confirmed_ocr_text: str | None = None,
) -> tuple[str, str, dict[str, object]]:
    if not text or not text.strip():
        raise gr.Error("Enter the instructions first.")
    if extracted_text_needs_review and (
        not extracted_text_confirmed or confirmed_ocr_text != text
    ):
        raise gr.Error(
            "Please check the extracted text against the photo and confirm it "
            "before simplifying."
        )

    try:
        verified_result = simplify(text, mode="verified", use_api=True)
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    verified = _format_notes(str(verified_result["output"]))
    check_summary = _format_checks(verified_result["checks"])
    return verified, check_summary, gr.update(interactive=True)


def translate_result(text: str, language: str) -> str:
    if not text or not text.strip():
        raise gr.Error("Simplify the instructions before translating.")
    if not language:
        raise gr.Error("Choose a language.")
    try:
        return _format_notes(translate_with_groq(text, language))
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc


def _delete_temporary_upload(upload_path: str) -> None:
    """Delete only files Gradio placed in its temporary upload directory."""
    root = Path(
        os.environ.get("GRADIO_TEMP_DIR")
        or Path(tempfile.gettempdir()) / "gradio"
    ).resolve()
    path = Path(upload_path).resolve()
    try:
        if os.path.commonpath((str(root), str(path))) != str(root):
            raise RuntimeError(
                "The uploaded file was outside the temporary upload folder. "
                "For privacy, it was not processed."
            )
    except ValueError as exc:
        raise RuntimeError(
            "The uploaded file was outside the temporary upload folder. "
            "For privacy, it was not processed."
        ) from exc
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(
            "The temporary image could not be cleared. Please restart the app "
            "before uploading another document."
        ) from exc


def _preview_markup(data_uri: str) -> str:
    if not data_uri:
        return ""
    return (
        '<img src="'
        + data_uri
        + '" alt="Preview of the fictional instruction photo">'
    )


def prepare_photo_upload(upload_path):
    if not upload_path:
        return (
            "",
            "Choose a JPG, JPEG, or PNG photo first.",
            None,
            gr.update(value=None),
            "",
            False,
            gr.update(value=False, visible=False),
            None,
            "",
            "",
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    try:
        prepared = prepare_image_upload(upload_path)
        return (
            _preview_markup(prepared.preview_data_uri),
            "Photo preview ready. It stays on this device. Select “Extract text” "
            "to read it locally.",
            prepared.image_bytes,
            gr.update(value=None),
            "",
            False,
            gr.update(value=False, visible=False),
            None,
            "",
            "",
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    except ImageInputError as exc:
        return (
            "",
            str(exc),
            None,
            gr.update(value=None),
            "",
            False,
            gr.update(value=False, visible=False),
            None,
            "",
            "",
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    finally:
        _delete_temporary_upload(upload_path)


def extract_photo_text(image_bytes):
    if not image_bytes:
        return (
            "",
            "Upload a photo before extracting text.",
            False,
            gr.update(value=False, visible=False),
            None,
            gr.update(interactive=True),
        )
    try:
        text = extract_instruction_text(image_bytes)
    except (ImageInputError, RuntimeError) as exc:
        return (
            "",
            str(exc),
            False,
            gr.update(value=False, visible=False),
            None,
            gr.update(interactive=True),
        )
    return (
        text,
        "Text extracted on this device. Compare every line with the photo and "
        "correct anything that looks wrong.",
        True,
        gr.update(value=False, visible=True),
        None,
        gr.update(interactive=False),
    )


def begin_photo_extraction() -> tuple[
    str, str, str, str, dict[str, object], dict[str, object]
]:
    return (
        "Reading text on this device…",
        "",
        "",
        "",
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def reset_ocr_confirmation(extracted_text_needs_review):
    confirmation = (
        gr.update(value=False) if extracted_text_needs_review else gr.update()
    )
    return (
        confirmation,
        "",
        "",
        "",
        gr.update(interactive=False),
        None,
        gr.update(interactive=not bool(extracted_text_needs_review)),
    )


def record_ocr_confirmation(
    text: str, confirmed: bool
) -> tuple[str | None, dict[str, object]]:
    return text if confirmed else None, gr.update(interactive=confirmed)


def load_example():
    return (
        EXAMPLE,
        None,
        False,
        gr.update(value=False, visible=False),
        "",
        "",
        None,
        "",
        "",
        "",
        gr.update(interactive=False),
        gr.update(interactive=True),
    )


def _bypass_proxy_for_localhost() -> None:
    for key in ("NO_PROXY", "no_proxy"):
        entries = [
            item.strip()
            for item in os.environ.get(key, "").split(",")
            if item.strip()
        ]
        normalized = {item.casefold() for item in entries}
        for host in ("localhost", "127.0.0.1"):
            if host not in normalized:
                entries.append(host)
        os.environ[key] = ",".join(entries)


with gr.Blocks(
    title="ComprisMED",
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="teal",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "Arial", "sans-serif"],
    ),
    css=CSS,
    analytics_enabled=False,
    delete_cache=(60, 60),
) as demo:
    with gr.Column(elem_classes=["cm-shell"]):
        with gr.Column(elem_classes=["cm-header"]):
            gr.HTML(
                """
                <div class="cm-brand" aria-label="ComprisMED">
                  <span class="cm-brand-mark" aria-hidden="true">
                    <svg width="30" height="30" viewBox="0 0 32 32" fill="none"
                         xmlns="http://www.w3.org/2000/svg">
                      <path d="M7 5.5h13l5 5v12A3.5 3.5 0 0 1 21.5 26h-11A3.5 3.5 0 0 1 7 22.5v-17Z"
                            stroke="#315B5A" stroke-width="1.8" stroke-linejoin="round"/>
                      <path d="M19.5 5.8v5h5M11 16h10M11 20h7"
                            stroke="#315B5A" stroke-width="1.8" stroke-linecap="round"
                            stroke-linejoin="round"/>
                      <path d="M24.2 23.8 28 27" stroke="#477A73" stroke-width="1.8"
                            stroke-linecap="round"/>
                    </svg>
                  </span>
                  <h1>ComprisMED</h1>
                </div>
                """
            )
            gr.Markdown(
                "**Safety note:** This is a prototype only, not medical advice.",
                elem_classes=["cm-safety"],
            )

        with gr.Row(elem_classes=["cm-workspace"]):
            with gr.Column(elem_classes=["cm-panel"]):
                gr.Markdown("### Your instructions")
                instructions = gr.Textbox(
                    label="",
                    placeholder="Paste medical instructions here",
                    lines=9,
                    max_lines=24,
                    show_label=False,
                )
                gr.Markdown("Or upload a photo instead", elem_classes=["cm-upload-title"])
                with gr.Row(elem_classes=["cm-upload-row"]):
                    with gr.Column(scale=1, min_width=220):
                        photo_upload = gr.File(
                            label="Upload a photo (JPG, JPEG, or PNG)",
                            file_types=[".jpg", ".jpeg", ".png"],
                            file_count="single",
                            type="filepath",
                            elem_classes=["cm-photo-upload"],
                        )
                        extract_button = gr.Button(
                            "Extract text",
                            variant="secondary",
                            size="lg",
                        )
                        photo_preview = gr.HTML(
                            "",
                            elem_classes=["cm-photo-preview"],
                        )
                ocr_status = gr.Markdown("", elem_classes=["cm-photo-status"])
                ocr_image = gr.State(None)
                ocr_requires_review = gr.State(False)
                confirmed_ocr_text = gr.State(None)
                ocr_confirmation = gr.Checkbox(
                    label="I checked the extracted text against the photo",
                    value=False,
                    visible=False,
                    elem_classes=["cm-photo-confirm"],
                )
                example_button = gr.Button(
                    "Use sample instructions", variant="secondary", size="sm"
                )
                with gr.Row(elem_classes=["cm-input-actions"]):
                    simplify_button = gr.Button(
                        "Simplify", variant="primary", size="lg"
                    )

            with gr.Column(elem_classes=["cm-panel"]):
                gr.Markdown("### Simplified instructions")
                simplified_output = gr.Textbox(
                    label="",
                    placeholder="Your simplified steps will appear here",
                    lines=7,
                    max_lines=18,
                    interactive=False,
                    show_label=False,
                    elem_classes=["cm-output"],
                )
                checks_output = gr.Markdown(
                    "",
                    elem_classes=["cm-checks"],
                )
                language = gr.Dropdown(
                    choices=[
                        "Arabic", "Chinese (Simplified)", "French", "German",
                        "Hindi", "Italian", "Korean", "Portuguese", "Russian",
                        "Spanish", "Tagalog", "Vietnamese",
                    ],
                    value="Spanish",
                    label="Translate result to",
                    elem_classes=["cm-language-select"],
                    allow_custom_value=False,
                )
                translate_button = gr.Button(
                    "Translate", variant="secondary", interactive=False
                )
                translation_output = gr.Textbox(
                    label="",
                    placeholder="Translation will appear here",
                    lines=3,
                    max_lines=10,
                    interactive=False,
                    show_label=False,
                    elem_classes=["cm-translation"],
                )

        gr.Markdown(
            "ComprisMED uses AI to simplify medical instructions and machine-learning "
            "to check selected details against the original.",
            elem_classes=["cm-description"],
        )

    photo_upload.upload(
        prepare_photo_upload,
        inputs=photo_upload,
        outputs=[
            photo_preview,
            ocr_status,
            ocr_image,
            photo_upload,
            instructions,
            ocr_requires_review,
            ocr_confirmation,
            confirmed_ocr_text,
            simplified_output,
            checks_output,
            translation_output,
            translate_button,
            simplify_button,
        ],
        show_progress="hidden",
        show_api=False,
    )
    extract_button.click(
        begin_photo_extraction,
        outputs=[
            ocr_status,
            simplified_output,
            checks_output,
            translation_output,
            translate_button,
            simplify_button,
        ],
        show_progress="hidden",
    ).then(
        extract_photo_text,
        inputs=ocr_image,
        outputs=[
            instructions,
            ocr_status,
            ocr_requires_review,
            ocr_confirmation,
            confirmed_ocr_text,
            simplify_button,
        ],
        show_progress="full",
    )
    instructions.input(
        reset_ocr_confirmation,
        inputs=ocr_requires_review,
        outputs=[
            ocr_confirmation,
            simplified_output,
            checks_output,
            translation_output,
            translate_button,
            confirmed_ocr_text,
            simplify_button,
        ],
        show_progress="hidden",
    )
    ocr_confirmation.change(
        record_ocr_confirmation,
        inputs=[instructions, ocr_confirmation],
        outputs=[confirmed_ocr_text, simplify_button],
        show_progress="hidden",
    )
    example_button.click(
        load_example,
        outputs=[
            instructions,
            ocr_image,
            ocr_requires_review,
            ocr_confirmation,
            ocr_status,
            photo_preview,
            confirmed_ocr_text,
            simplified_output,
            checks_output,
            translation_output,
            translate_button,
            simplify_button,
        ],
    )
    simplify_event = simplify_button.click(
        run_demo,
        inputs=[
            instructions,
            ocr_requires_review,
            ocr_confirmation,
            confirmed_ocr_text,
        ],
        outputs=[simplified_output, checks_output, translate_button],
        show_progress="full",
    )
    simplify_event.then(
        fn=None,
        inputs=[],
        outputs=[],
        js="() => { setTimeout(() => document.querySelectorAll('.cm-output textarea').forEach((field) => { field.scrollTop = 0; }), 0); return []; }",
        show_progress="hidden",
    )
    translate_event = translate_button.click(
        translate_result,
        inputs=[simplified_output, language],
        outputs=translation_output,
        show_progress="full",
    )
    translate_event.then(
        fn=None,
        inputs=[],
        outputs=[],
        js="() => { setTimeout(() => document.querySelectorAll('.cm-translation textarea').forEach((field) => { field.scrollTop = 0; }), 0); return []; }",
        show_progress="hidden",
    )


if __name__ == "__main__":
    _bypass_proxy_for_localhost()
    demo.launch(show_api=False, max_file_size="12mb")
