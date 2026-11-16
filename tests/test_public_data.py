import json
from pathlib import Path

import pytest

from data.public.download_cochrane import validate_splits
from data.public.cochrane.build_adjudicated_annotations import normalize_facts
from data.public.cochrane.prepare_annotations import select_test_rows

COCHRANE_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "public" / "cochrane"
)


def test_downloaded_cochrane_official_splits_are_doi_disjoint() -> None:
    rows_by_split = {
        split: json.loads((COCHRANE_DIR / f"{split}.json").read_text(encoding="utf-8"))
        for split in ("train", "validation", "test")
    }
    assert {split: len(rows) for split, rows in rows_by_split.items()} == {
        "train": 3568,
        "validation": 411,
        "test": 480,
    }
    validate_splits(rows_by_split)


def test_split_validator_rejects_doi_leakage() -> None:
    with pytest.raises(ValueError, match="DOI leakage"):
        validate_splits(
            {
                "train": [{"doi": "shared"}],
                "test": [{"doi": "shared"}],
            }
        )


def test_cochrane_annotation_sample_is_stable_and_has_no_targets() -> None:
    rows = json.loads((COCHRANE_DIR / "test.json").read_text(encoding="utf-8"))
    first = select_test_rows(rows)
    second = select_test_rows(list(reversed(rows)))
    assert first == second
    assert len(first) == 30
    assert all(set(row) == {"example_id", "doi", "source"} for row in first)


def test_public_gold_facts_must_be_exact_source_spans() -> None:
    facts = normalize_facts(
        '{"number_or_dose":["0.73"]}',
        "The estimated relative risk was 0.73.",
        "example",
    )
    assert facts == {"number_or_dose": ["0.73"]}
    with pytest.raises(ValueError, match="not an exact source span"):
        normalize_facts(
            '{"number_or_dose":["0.74"]}',
            "The estimated relative risk was 0.73.",
            "example",
        )
