"""Compare ordinary AI simplification with AI plus source-check and one repair."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from comprismed.facts import Check, compare_facts
from comprismed.groq import (
    DEFAULT_MODEL,
    repair_summary_with_groq,
    repair_with_groq,
    simplify_summary_with_groq,
    simplify_with_groq,
)
from comprismed.metrics import checks_by_category, detection_metrics, preservation_metrics

SYNTHETIC_DATA = ROOT / "data" / "synthetic" / "benchmark.jsonl"
COCHRANE_DIR = ROOT / "data" / "public" / "cochrane"
OUTPUT_DIR = ROOT / "results"

Generate = Callable[[str], str]
Repair = Callable[[str, str, dict[str, list[str]]], str]


def load_synthetic_rows(split: str) -> list[dict[str, object]]:
    with SYNTHETIC_DATA.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"No synthetic examples found for split {split!r}.")
    return selected


def load_cochrane_rows(split: str) -> list[dict[str, object]]:
    if split != "test":
        raise ValueError(
            "Cochrane labels are prepared only for the held-out test sample; "
            "use --split test."
        )
    data_path = COCHRANE_DIR / "test.json"
    annotation_path = COCHRANE_DIR / "adjudicated_annotations.jsonl"
    if not data_path.is_file():
        raise FileNotFoundError(
            "Cochrane files are missing. Run python data/public/download_cochrane.py."
        )
    if not annotation_path.is_file():
        raise FileNotFoundError(
            "Cochrane test annotations are not ready. Follow data/public/README.md "
            "to prepare two blind reviews and adjudicate them."
        )
    source_rows = json.loads(data_path.read_text(encoding="utf-8"))
    source_by_doi = {row["doi"]: row for row in source_rows}
    annotations = [
        json.loads(line)
        for line in annotation_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(annotations) != 30:
        raise ValueError("Expected 30 double-reviewed Cochrane test annotations.")
    if len({row.get("doi") for row in annotations}) != len(annotations):
        raise ValueError("Cochrane annotations contain duplicate DOI groups.")
    for row in annotations:
        doi = row.get("doi")
        if (
            row.get("split") != "test"
            or doi not in source_by_doi
            or row.get("source") != source_by_doi[doi]["source"]
            or row.get("review_status")
            not in {"double_review_consensus", "double_review_adjudicated"}
            or not isinstance(row.get("required_facts"), dict)
            or not row["required_facts"]
        ):
            raise ValueError(f"Invalid or unreviewed Cochrane annotation: {row.get('example_id')}")
    return annotations


def run_pair(
    source: str,
    required_facts: dict[str, list[str]],
    generate: Generate,
    repair: Repair,
) -> dict[str, object]:
    baseline = generate(source)
    baseline_checks = compare_facts(source, baseline, required_facts)
    failed_facts: dict[str, list[str]] = defaultdict(list)
    for check in baseline_checks:
        if check.status != "preserved":
            failed_facts[check.category].append(check.source_fact)

    verified = baseline
    if failed_facts:
        verified = repair(source, baseline, dict(failed_facts))
    verified_checks = compare_facts(source, verified, required_facts)
    return {
        "ai_only": {
            "output": baseline,
            "metrics": preservation_metrics(source, baseline, required_facts),
            "checks": [check.__dict__ for check in baseline_checks],
            "checks_by_category": checks_by_category(baseline_checks),
        },
        "ai_plus_verification": {
            "initial_output": baseline,
            "output": verified,
            "repair_attempted": bool(failed_facts),
            "metrics": preservation_metrics(source, verified, required_facts),
            "checks": [check.__dict__ for check in verified_checks],
            "checks_by_category": checks_by_category(verified_checks),
        },
    }


def summarize_condition(records: list[dict[str, object]], condition: str) -> dict[str, object]:
    all_checks: list[Check] = []
    required = preserved = omissions = alterations = 0
    for record in records:
        result = record[condition]
        metrics = result["metrics"]
        required += int(metrics["required_fact_count"])
        preserved += int(metrics["preserved_count"])
        omissions += int(metrics["omission_count"])
        alterations += int(metrics["alteration_count"])
        all_checks.extend(
            Check(**check) for check in result["checks"]
        )
    return {
        "examples": len(records),
        "required_facts": required,
        "preserved": preserved,
        "preservation_rate": preserved / required if required else 0.0,
        "omissions": omissions,
        "omission_rate": omissions / required if required else 0.0,
        "alterations": alterations,
        "alteration_rate": alterations / required if required else 0.0,
        "per_category": checks_by_category(all_checks),
    }


def synthetic_mutation_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    expected: list[bool] = []
    detected: list[bool] = []
    categories: dict[str, tuple[list[bool], list[bool]]] = {}
    for row in rows:
        source = str(row["source"])
        facts = row["required_facts"]
        clean_checks = compare_facts(source, source, facts)
        expected.append(False)
        detected.append(any(check.status != "preserved" for check in clean_checks))
        for category, category_facts in facts.items():
            category_expected, category_detected = categories.setdefault(
                category, ([], [])
            )
            category_checks = compare_facts(
                source, source, {category: category_facts}
            )
            category_expected.append(False)
            category_detected.append(
                any(check.status != "preserved" for check in category_checks)
            )
        for mutation in row.get("mutations", []):
            category = str(mutation["category"])
            mutation_checks = compare_facts(
                source,
                str(mutation["output"]),
                {category: [str(mutation["required_fact"])]},
            )
            expected.append(True)
            detected.append(
                any(check.status != "preserved" for check in mutation_checks)
            )
            category_expected, category_detected = categories.setdefault(
                category, ([], [])
            )
            category_expected.append(True)
            category_detected.append(
                any(check.status != "preserved" for check in mutation_checks)
            )
    return {
        "description": (
            "Deterministic verifier on clean source text and controlled synthetic "
            "mutations; not a Groq result or clinical measure."
        ),
        "overall": detection_metrics(expected, detected),
        "per_category": {
            category: detection_metrics(*pairs)
            for category, pairs in sorted(categories.items())
        },
    }


def build_examples(records: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    successes = []
    failures = []
    ai_only_failures = []
    repaired_successes = []
    for record in records:
        result = record["ai_plus_verification"]
        metrics = result["metrics"]
        example = {
            "id": record["id"],
            "source": record["source"],
            "ai_only_output": record["ai_only"]["output"],
            "ai_only_metrics": record["ai_only"]["metrics"],
            "verified_initial_output": result["initial_output"],
            "verified_final_output": result["output"],
            "repair_attempted": result["repair_attempted"],
            "verified_metrics": metrics,
            "remaining_checks": result["checks"],
        }
        destination = successes if (
            metrics["omission_count"] == 0
            and metrics["alteration_count"] == 0
        ) else failures
        if len(destination) < 3:
            destination.append(example)
        baseline_metrics = record["ai_only"]["metrics"]
        if (
            baseline_metrics["omission_count"]
            or baseline_metrics["alteration_count"]
        ) and len(ai_only_failures) < 3:
            ai_only_failures.append(example)
        if (
            result["repair_attempted"]
            and metrics["omission_count"] == 0
            and metrics["alteration_count"] == 0
            and len(repaired_successes) < 3
        ):
            repaired_successes.append(example)
    return {
        "successes": successes,
        "failures": failures,
        "ai_only_failures": ai_only_failures,
        "repaired_successes": repaired_successes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("synthetic", "cochrane"),
        default="synthetic",
        help="Evaluate fictional instructions (primary) or Cochrane summaries (secondary).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
        help="The test split is the held-out result; do not tune against it.",
    )
    parser.add_argument(
        "--checker-only",
        action="store_true",
        help="Evaluate synthetic clean/mutated pairs without making Groq calls.",
    )
    args = parser.parse_args()

    if args.dataset == "synthetic":
        rows = load_synthetic_rows(args.split)
        if args.checker_only:
            summary = {
                "dataset": "synthetic",
                "split": args.split,
                "controlled_mutation_detection": synthetic_mutation_metrics(rows),
            }
            OUTPUT_DIR.mkdir(exist_ok=True)
            metrics_path = OUTPUT_DIR / (
                f"synthetic_{args.split}_checker_metrics.json"
            )
            metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps({"metrics_file": str(metrics_path), **summary}, indent=2))
            return
        generate: Generate = simplify_with_groq
        repair: Repair = repair_with_groq
    else:
        if args.checker_only:
            parser.error("--checker-only is available only for the synthetic dataset.")
        rows = load_cochrane_rows(args.split)
        generate = simplify_summary_with_groq
        repair = repair_summary_with_groq

    records: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        print(
            f"Running {args.dataset} {args.split} example "
            f"{index}/{len(rows)}: {row.get('id', row.get('example_id'))}",
            flush=True,
        )
        source = str(row["source"])
        facts = row["required_facts"]
        comparison = run_pair(source, facts, generate, repair)
        records.append(
            {
                "id": row.get("id", row.get("example_id")),
                "split": args.split,
                "dataset": args.dataset,
                "source": source,
                "required_facts": facts,
                **comparison,
            }
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    outputs_path = OUTPUT_DIR / f"{args.dataset}_{args.split}_outputs.json"
    outputs_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary: dict[str, object] = {
        "dataset": args.dataset,
        "split": args.split,
        "examples": len(records),
        "model": os.getenv("GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        "comparison": (
            "AI-only output versus the same AI output plus deterministic checks "
            "and at most one repair call"
        ),
        "conditions": {
            condition: summarize_condition(records, condition)
            for condition in ("ai_only", "ai_plus_verification")
        },
        "examples": build_examples(records),
    }
    if args.dataset == "synthetic":
        summary["controlled_mutation_detection"] = synthetic_mutation_metrics(rows)

    metrics_path = OUTPUT_DIR / f"{args.dataset}_{args.split}_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"outputs": str(outputs_path), "metrics_file": str(metrics_path), "summary": summary},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
