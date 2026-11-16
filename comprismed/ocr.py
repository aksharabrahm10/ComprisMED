"""Local OCR for fictional instruction images; image bytes never leave the app."""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 30_000_000
MIN_TEXT_CHARACTERS = 8
MIN_OCR_CONFIDENCE = 0.5
MIN_LINE_CONFIDENCE = 0.2


class ImageInputError(ValueError):
    """The selected image is unsupported, unreadable, or too unclear."""


@dataclass(frozen=True)
class PreparedImage:
    image_bytes: bytes
    preview_data_uri: str


def _decode_image(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ImageInputError("Please upload a JPG, JPEG, or PNG image.")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ImageInputError(
                    "This image is too large to read. Try a smaller photo."
                )
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except ImageInputError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageInputError(
            "This image could not be opened. Try another JPG or PNG photo."
        ) from exc


def prepare_image_upload(path: str | Path) -> PreparedImage:
    """Validate and decode an upload, returning only in-memory image data."""
    image_path = Path(path)
    if image_path.suffix.casefold() not in ALLOWED_EXTENSIONS:
        raise ImageInputError("Unsupported file. Choose a JPG, JPEG, or PNG image.")
    try:
        size = image_path.stat().st_size
        if size == 0:
            raise ImageInputError("This image is empty. Choose another photo.")
        if size > MAX_UPLOAD_BYTES:
            raise ImageInputError(
                "This image is larger than 12 MB. Choose a smaller photo."
            )
        image_bytes = image_path.read_bytes()
    except OSError as exc:
        raise ImageInputError(
            "The uploaded image could not be read. Please upload it again."
        ) from exc

    image = _decode_image(image_bytes)
    normalized = io.BytesIO()
    image.save(normalized, format="PNG", optimize=True)
    normalized_bytes = normalized.getvalue()
    encoded = base64.b64encode(normalized_bytes).decode("ascii")
    return PreparedImage(
        image_bytes=normalized_bytes,
        preview_data_uri=f"data:image/png;base64,{encoded}",
    )


@lru_cache(maxsize=1)
def _get_ocr_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "Local OCR is unavailable. Install the project requirements and restart."
        ) from exc
    return RapidOCR()


def _readable_ocr_text(results) -> str:
    positioned: list[tuple[float, float, float, float, str]] = []
    unpositioned: list[str] = []
    for result in results:
        line = " ".join(result[1].split())
        if not line:
            continue
        try:
            points = np.asarray(result[0], dtype=float)
        except (TypeError, ValueError):
            points = np.empty((0, 0))
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
            unpositioned.append(line)
            continue
        left = float(points[:, 0].min())
        right = float(points[:, 0].max())
        top = float(points[:, 1].min())
        bottom = float(points[:, 1].max())
        positioned.append(((top + bottom) / 2, left, bottom - top, right, line))

    rows: list[list[tuple[float, float, float, float, str]]] = []
    for item in sorted(positioned, key=lambda value: (value[0], value[1])):
        center, _, height, _, _ = item
        matching_row = next(
            (
                row
                for row in rows
                if abs(center - sum(entry[0] for entry in row) / len(row))
                <= max(height, sum(entry[2] for entry in row) / len(row)) * 0.55
            ),
            None,
        )
        if matching_row is None:
            rows.append([item])
        else:
            matching_row.append(item)

    text_rows = []
    for row in rows:
        line = " ".join(item[4] for item in sorted(row, key=lambda value: value[1]))
        line = re.sub(r"\s+([,.;:!?%)])", r"\1", line)
        line = re.sub(r"([(])\s+", r"\1", line)
        text_rows.append(line)
    text_rows.extend(unpositioned)
    return "\n".join(text_rows).strip()


def extract_instruction_text(image_bytes: bytes) -> str:
    """Read text locally and reject empty or low-confidence OCR results."""
    if not image_bytes:
        raise ImageInputError("Choose a photo before extracting text.")
    image = _decode_image(image_bytes)
    image_array = np.asarray(image)[:, :, ::-1].copy()

    try:
        results, _ = _get_ocr_engine()(image_array)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "Local text extraction could not finish. Try another photo or type "
            "the instructions instead."
        ) from exc

    if not results:
        raise ImageInputError(
            "We could not read clear text in this photo. Try a sharper, well-lit "
            "image with the full instruction visible, or type the text instead."
        )

    recognized_lines: list[str] = []
    confidences: list[float] = []
    for result in results:
        if len(result) < 3 or not isinstance(result[1], str):
            raise RuntimeError(
                "Local text extraction returned an unexpected result. Try again "
                "or type the instructions instead."
            )
        line = result[1].strip()
        try:
            confidence = float(result[2])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Local text extraction returned an unreadable confidence value."
            ) from exc
        if line:
            recognized_lines.append(line)
            confidences.append(confidence)

    text = _readable_ocr_text(results)
    if len(text) < MIN_TEXT_CHARACTERS or not confidences:
        raise ImageInputError(
            "We could not read enough text in this photo. Try a sharper, well-lit "
            "image with the full instruction visible, or type the text instead."
        )
    if (
        sum(confidences) / len(confidences) < MIN_OCR_CONFIDENCE
        or min(confidences) < MIN_LINE_CONFIDENCE
    ):
        raise ImageInputError(
            "Some text in this photo is too unclear to extract reliably. Try a "
            "sharper photo with the full instruction visible, or type the text."
        )
    return text
