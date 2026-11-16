"""Download and verify the CC BY 4.0 GEM Cochrane simplification dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent / "cochrane"
REVISION = "75a92ae445171fa1b7641a229bfe3c77c0d8723d"
FILES = {
    "train.json": {
        "sha256": "4b52d2e8a09aecb0f9d1fe8894927e49ca019e8cbd588dfa762ef8daab93e119",
    },
    "validation.json": {
        "sha256": "18c883a77ff20f718b71c05251146d9648022f277cd4dc345b43820cd6ba2d5f",
    },
    "test.json": {
        "sha256": "04795eda349393a97817c1e5da65c988b272a686999b683e7838d8f1254a5002",
    },
}
REQUIRED_FIELDS = {"source", "target", "doi"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_dataset_file(path: Path, expected_sha256: str) -> bool:
    if not path.is_file() or sha256(path) != expected_sha256:
        return False
    with path.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} is not a non-empty JSON list.")
    if not all(
        isinstance(row, dict)
        and REQUIRED_FIELDS.issubset(row)
        and all(isinstance(row[field], str) and row[field].strip() for field in REQUIRED_FIELDS)
        for row in rows
    ):
        raise ValueError(f"{path} does not have the expected source/target/doi fields.")
    return True


def validate_splits(rows_by_split: dict[str, list[dict[str, str]]]) -> None:
    doi_sets: dict[str, set[str]] = {}
    for split, rows in rows_by_split.items():
        dois = [row["doi"] for row in rows]
        if len(dois) != len(set(dois)):
            raise ValueError(f"The {split} split contains duplicate DOI groups.")
        doi_sets[split] = set(dois)
    split_names = sorted(doi_sets)
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            overlap = doi_sets[first] & doi_sets[second]
            if overlap:
                raise ValueError(
                    f"DOI leakage between {first} and {second}: "
                    f"{sorted(overlap)[:3]}"
                )


def download_file(filename: str, expected_sha256: str) -> Path:
    ROOT.mkdir(parents=True, exist_ok=True)
    destination = ROOT / filename
    if valid_dataset_file(destination, expected_sha256):
        return destination

    url = (
        "https://huggingface.co/datasets/GEM/cochrane-simplification/"
        f"resolve/{REVISION}/{filename}?download=true"
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        actual_sha256 = sha256(temporary)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Checksum mismatch for {filename}: expected {expected_sha256}, "
                f"received {actual_sha256}."
            )
        temporary.replace(destination)
        if not valid_dataset_file(destination, expected_sha256):
            raise ValueError(f"Downloaded {filename} failed JSON/schema validation.")
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    summary = {}
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    for filename, metadata in FILES.items():
        path = download_file(filename, metadata["sha256"])
        with path.open(encoding="utf-8") as stream:
            rows = json.load(stream)
        rows_by_split[path.stem] = rows
        summary[filename] = {
            "rows": len(rows),
            "bytes": path.stat().st_size,
            "sha256": metadata["sha256"],
        }
    validate_splits(rows_by_split)
    print(json.dumps({"dataset": "GEM/cochrane-simplification", "files": summary}, indent=2))


if __name__ == "__main__":
    main()
