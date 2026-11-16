"""Prepare independent, source-only annotation sheets for a fixed test sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
REVISION = "75a92ae445171fa1b7641a229bfe3c77c0d8723d"
SAMPLE_SIZE = 30
FIELDS = (
    "example_id",
    "doi",
    "source",
    "facts_json",
    "complete",
    "reviewer_notes",
)


def select_test_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if len({row.get("doi") for row in rows}) != len(rows):
        raise ValueError("The Cochrane test split must have unique DOI groups.")
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{REVISION}|{row['doi']}".encode("utf-8")
        ).hexdigest(),
    )
    selected = ordered[:SAMPLE_SIZE]
    return [
        {
            "example_id": "cochrane-test-"
            + hashlib.sha256(f"{REVISION}|{row['doi']}".encode("utf-8"))
            .hexdigest()[:12],
            "doi": row["doi"],
            "source": row["source"],
        }
        for row in selected
    ]


def write_review_sheet(path: Path, rows: list[dict[str, str]], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists. It may contain annotations; use --force only "
            "if you intentionally want to replace it."
        )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "facts_json": "",
                    "complete": "",
                    "reviewer_notes": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing reviewer sheets (this deletes their annotations).",
    )
    args = parser.parse_args()

    data_path = DATA_DIR / "test.json"
    if not data_path.is_file():
        raise FileNotFoundError(
            "Cochrane data is missing. Run python data/public/download_cochrane.py first."
        )
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    selected = select_test_rows(rows)
    if len(selected) != SAMPLE_SIZE:
        raise ValueError(f"Expected {SAMPLE_SIZE} selected rows, got {len(selected)}.")

    review_paths = [DATA_DIR / "reviewer_a.csv", DATA_DIR / "reviewer_b.csv"]
    existing = [path for path in review_paths if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Reviewer sheets already exist and may contain annotations: "
            + ", ".join(str(path) for path in existing)
            + ". Use --force only if you intentionally want to replace them."
        )
    for path in review_paths:
        write_review_sheet(path, selected, args.force)
    print(
        f"Created two blank source-only annotation sheets with {SAMPLE_SIZE} "
        f"held-out examples in {DATA_DIR}."
    )


if __name__ == "__main__":
    main()
