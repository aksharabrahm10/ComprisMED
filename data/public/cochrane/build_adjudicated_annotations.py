"""Validate two blind reviews and build gold facts only after consensus/adjudication."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

if __package__:
    from .prepare_annotations import DATA_DIR, select_test_rows
else:
    from prepare_annotations import DATA_DIR, select_test_rows

REVIEW_FIELDS = {
    "example_id",
    "doi",
    "source",
    "facts_json",
    "complete",
    "reviewer_notes",
}
ADJUDICATION_FIELDS = (
    "example_id",
    "final_facts_json",
    "adjudication_complete",
    "notes",
)
ALLOWED_CATEGORIES = {
    "number_or_dose",
    "frequency",
    "duration",
    "date",
    "negation",
}


def normalize_facts(raw: str, source: str, example_id: str) -> dict[str, list[str]]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{example_id}: facts_json must be valid JSON.") from exc
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError(f"{example_id}: annotate at least one source detail.")

    normalized: dict[str, list[str]] = {}
    source_folded = re.sub(r"\s+", " ", source.casefold())
    for category, values in parsed.items():
        if category not in ALLOWED_CATEGORIES:
            raise ValueError(f"{example_id}: unsupported fact category {category!r}.")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{example_id}: {category} must have a non-empty list.")
        clean_values = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{example_id}: facts must be non-empty strings.")
            fact = re.sub(r"\s+", " ", value.strip())
            if fact.casefold() not in source_folded:
                raise ValueError(
                    f"{example_id}: annotated fact {fact!r} is not an exact source span."
                )
            clean_values.append(fact)
        normalized[category] = sorted(set(clean_values), key=str.casefold)
    return dict(sorted(normalized.items()))


def load_review(path: Path, expected: dict[str, dict[str, str]]) -> dict[str, dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.name}. Run prepare_annotations.py and complete both reviews."
        )
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REVIEW_FIELDS.issubset(reader.fieldnames):
            raise ValueError(f"{path.name} is missing required review columns.")
        rows = list(reader)
    by_id = {row["example_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(expected):
        raise ValueError(f"{path.name} must contain exactly the fixed annotation sample.")
    for example_id, row in by_id.items():
        reference = expected[example_id]
        if row["doi"] != reference["doi"] or row["source"] != reference["source"]:
            raise ValueError(f"{path.name}: source/DOI changed for {example_id}.")
        if row["complete"].strip().casefold() != "yes":
            raise ValueError(f"{path.name}: {example_id} is not marked complete=yes.")
        row["normalized_facts"] = normalize_facts(
            row["facts_json"], reference["source"], example_id
        )
    return by_id


def load_adjudications(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(ADJUDICATION_FIELDS).issubset(
            reader.fieldnames
        ):
            raise ValueError(f"{path.name} is missing required adjudication columns.")
        rows = list(reader)
    return {row["example_id"]: row for row in rows}


def write_adjudication_template(
    path: Path,
    disagreements: list[dict[str, str]],
    existing: dict[str, dict[str, str]],
) -> None:
    if path.exists():
        existing_ids = {row["example_id"] for row in existing.values()}
        if existing_ids != {row["example_id"] for row in disagreements}:
            raise ValueError(
                f"{path.name} does not match the current reviewer disagreements."
            )
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ADJUDICATION_FIELDS)
        writer.writeheader()
        writer.writerows(disagreements)


def main() -> None:
    test_path = DATA_DIR / "test.json"
    if not test_path.is_file():
        raise FileNotFoundError(
            "Cochrane data is missing. Run python data/public/download_cochrane.py first."
        )
    test_rows = json.loads(test_path.read_text(encoding="utf-8"))
    selected = select_test_rows(test_rows)
    expected = {row["example_id"]: row for row in selected}
    reviewer_a = load_review(DATA_DIR / "reviewer_a.csv", expected)
    reviewer_b = load_review(DATA_DIR / "reviewer_b.csv", expected)

    adjudication_path = DATA_DIR / "adjudication.csv"
    adjudications = load_adjudications(adjudication_path)
    disagreements: list[dict[str, str]] = []
    output: list[dict[str, object]] = []
    for example_id, reference in expected.items():
        facts_a = reviewer_a[example_id]["normalized_facts"]
        facts_b = reviewer_b[example_id]["normalized_facts"]
        if facts_a == facts_b:
            final_facts = facts_a
            review_status = "double_review_consensus"
        else:
            disagreement = {
                "example_id": example_id,
                "final_facts_json": "",
                "adjudication_complete": "",
                "notes": "",
            }
            disagreements.append(disagreement)
            adjudication = adjudications.get(example_id)
            if not adjudication or adjudication[
                "adjudication_complete"
            ].strip().casefold() != "yes":
                continue
            final_facts = normalize_facts(
                adjudication["final_facts_json"],
                reference["source"],
                example_id,
            )
            review_status = "double_review_adjudicated"
        output.append(
            {
                "example_id": example_id,
                "split": "test",
                "doi": reference["doi"],
                "source": reference["source"],
                "required_facts": final_facts,
                "review_status": review_status,
            }
        )

    write_adjudication_template(adjudication_path, disagreements, adjudications)
    if len(output) != len(expected):
        unresolved = len(expected) - len(output)
        raise ValueError(
            f"{unresolved} Cochrane review(s) need adjudication. Complete "
            f"{adjudication_path.name} and rerun this script."
        )

    destination = DATA_DIR / "adjudicated_annotations.jsonl"
    serialized = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in output
    )
    if destination.exists() and destination.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(
            f"{destination} already exists with different contents; move it aside "
            "only if you intentionally want to replace previously reviewed labels."
        )
    destination.write_text(serialized, encoding="utf-8")
    print(
        f"Validated {len(output)} double-reviewed held-out annotations and wrote "
        f"{destination}."
    )


if __name__ == "__main__":
    main()
