from __future__ import annotations

import io
import os
from pathlib import Path
import pytest
from PIL import Image, ImageDraw, ImageFont

from comprismed import ocr


def make_png_bytes(text: str = "Take 5 mg twice daily for 7 days.") -> bytes:
    image = Image.new("RGB", (1400, 360), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 100), text, fill="black", font=ImageFont.load_default())
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_prepare_image_upload_returns_only_normalized_bytes_and_preview(
    tmp_path: Path,
) -> None:
    upload = tmp_path / "fictional.jpg"
    image = Image.new("RGB", (120, 80), "white")
    image.save(upload, format="JPEG")

    prepared = ocr.prepare_image_upload(upload)

    assert prepared.image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert prepared.preview_data_uri.startswith("data:image/png;base64,")


def test_prepare_image_upload_rejects_unsupported_and_invalid_files(
    tmp_path: Path,
) -> None:
    unsupported = tmp_path / "instructions.pdf"
    unsupported.write_bytes(b"not an image")
    with pytest.raises(ocr.ImageInputError, match="Unsupported file"):
        ocr.prepare_image_upload(unsupported)

    invalid = tmp_path / "instructions.png"
    invalid.write_bytes(b"not a png")
    with pytest.raises(ocr.ImageInputError, match="could not be opened"):
        ocr.prepare_image_upload(invalid)


def test_extract_instruction_text_returns_recognized_lines(monkeypatch) -> None:
    monkeypatch.setattr(
        ocr,
        "_get_ocr_engine",
        lambda: lambda _image: (
            [
                ([], "Take 5 mg twice daily", 0.96),
                ([], "for 7 days.", 0.93),
            ],
            0.1,
        ),
    )
    extracted = ocr.extract_instruction_text(make_png_bytes())
    assert extracted == "Take 5 mg twice daily\nfor 7 days."


def test_readable_ocr_text_joins_words_on_same_row_and_preserves_line_breaks():
    results = [
        ([[0, 0], [45, 0], [45, 12], [0, 12]], "Take", 0.96),
        ([[52, 1], [70, 1], [70, 13], [52, 13]], "5 mg", 0.96),
        ([[0, 24], [65, 24], [65, 36], [0, 36]], "twice daily", 0.96),
    ]

    assert ocr._readable_ocr_text(results) == "Take 5 mg\ntwice daily"


def test_extract_instruction_text_rejects_empty_or_low_confidence_text(
    monkeypatch,
) -> None:
    monkeypatch.setattr(ocr, "_get_ocr_engine", lambda: lambda _image: (None, 0.1))
    with pytest.raises(ocr.ImageInputError, match="could not read clear text"):
        ocr.extract_instruction_text(make_png_bytes())

    monkeypatch.setattr(
        ocr,
        "_get_ocr_engine",
        lambda: lambda _image: ([([], "Unclear fictional words", 0.3)], 0.1),
    )
    with pytest.raises(ocr.ImageInputError, match="too unclear"):
        ocr.extract_instruction_text(make_png_bytes())


def test_photo_upload_cleans_temporary_file_and_clears_old_text(
    monkeypatch, tmp_path: Path
) -> None:
    import app

    upload_root = tmp_path / "gradio"
    upload_root.mkdir()
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(upload_root))
    upload = upload_root / "temporary.png"
    Image.new("RGB", (120, 80), "white").save(upload, format="PNG")

    (
        preview,
        status,
        image_bytes,
        clear_upload,
        instructions,
        review,
        checkbox,
        reviewed_text,
        simplified,
        checks,
        translation,
        translate_button,
        simplify_button,
    ) = app.prepare_photo_upload(str(upload))

    assert not upload.exists()
    assert preview.startswith('<img src="data:image/png;base64,')
    assert "Photo preview ready" in status
    assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert clear_upload["value"] is None
    assert instructions == ""
    assert review is False
    assert checkbox["visible"] is False
    assert reviewed_text is None
    assert simplified == ""
    assert checks == ""
    assert translation == ""
    assert translate_button["interactive"] is False
    assert simplify_button["interactive"] is False


def test_load_example_clears_previous_results() -> None:
    import app

    (
        text,
        image,
        review,
        confirmation,
        status,
        preview,
        reviewed_text,
        simplified,
        checks,
        translation,
        translate_button,
        simplify_button,
    ) = app.load_example()

    assert text == app.EXAMPLE
    assert image is None
    assert review is False
    assert confirmation["visible"] is False
    assert status == ""
    assert preview == ""
    assert reviewed_text is None
    assert simplified == ""
    assert checks == ""
    assert translation == ""
    assert translate_button["interactive"] is False
    assert simplify_button["interactive"] is True


def test_run_demo_requires_review_for_ocr_text() -> None:
    import gradio as gr

    import app

    with pytest.raises(gr.Error, match="check the extracted text"):
        app.run_demo("Take 5 mg twice daily.", True, False, None)


def test_run_demo_rejects_confirmation_for_different_ocr_text() -> None:
    import gradio as gr

    import app

    with pytest.raises(gr.Error, match="check the extracted text"):
        app.run_demo(
            "Take 10 mg twice daily.",
            True,
            True,
            "Take 5 mg twice daily.",
        )


def test_run_demo_accepts_confirmed_ocr_text(monkeypatch) -> None:
    import app

    monkeypatch.setattr(
        app,
        "simplify",
        lambda _text, **_kwargs: {
            "output": "Take 5 mg twice daily.",
            "checks": [],
        },
    )
    simplified, _, _ = app.run_demo(
        "Take 5 mg twice daily.",
        True,
        True,
        "Take 5 mg twice daily.",
    )
    assert simplified == "- Take 5 mg twice daily."


def test_editing_ocr_text_clears_confirmation_and_previous_results() -> None:
    import app

    (
        confirmation,
        simplified,
        checks,
        translation,
        translate_button,
        reviewed_text,
        simplify_button,
    ) = app.reset_ocr_confirmation(True)
    assert confirmation["value"] is False
    assert simplified == ""
    assert checks == ""
    assert translation == ""
    assert translate_button["interactive"] is False
    assert reviewed_text is None
    assert simplify_button["interactive"] is False


def test_begin_photo_extraction_clears_previous_results() -> None:
    import app

    status, simplified, checks, translation, translate_button, simplify_button = (
        app.begin_photo_extraction()
    )

    assert "Reading text on this device" in status
    assert simplified == ""
    assert checks == ""
    assert translation == ""
    assert translate_button["interactive"] is False
    assert simplify_button["interactive"] is False


def test_ocr_confirmation_controls_simplify_button() -> None:
    import app

    confirmed_text, button = app.record_ocr_confirmation(
        "Take 5 mg twice daily.", True
    )
    assert confirmed_text == "Take 5 mg twice daily."
    assert button["interactive"] is True

    confirmed_text, button = app.record_ocr_confirmation(
        "Take 5 mg twice daily.", False
    )
    assert confirmed_text is None
    assert button["interactive"] is False


def test_local_launch_proxy_bypass_preserves_existing_entries(monkeypatch) -> None:
    import app

    monkeypatch.setenv("NO_PROXY", "example.test")
    monkeypatch.setenv("no_proxy", "example.test")

    app._bypass_proxy_for_localhost()

    assert "example.test" in os.environ["NO_PROXY"]
    assert "localhost" in os.environ["NO_PROXY"]
    assert "127.0.0.1" in os.environ["no_proxy"]
