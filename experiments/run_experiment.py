"""Run the paired baseline-vs-verification experiment on held-out synthetic cases."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from comprismed.facts import compare_facts
from comprismed.metrics import checks_by_category, detection_metrics, preservation_metrics
from comprismed.groq import simplify_with_groq


DATA = ROOT / "data" / "synthetic" / "benchmark.jsonl"
OUTPUT_DIR = ROOT / "results"


def load_rows(split: str) -> list[dict[str, object]]:
    with DATA.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"No examples found for split {split!r} in {DATA}")
    return selected


def main() -> None:
    split = os.getenv("CARETRACE_SPLIT", "test")
    rows = load_rows(split)
    OUTPUT_DIR.mkdir(exist_ok=True)
    records: list[dict[str, object]] = []
    for row in rows:
        source = str(row["source"])
        facts = row["required_facts"]
        required = [item for values in facts.values() for item in values]
        baseline = simplify_with_groq(source, required_facts=required)
        baseline_checks = compare_facts(source, baseline, facts)
        baseline_metrics = preservation_metrics(source, baseline, facts)

        verified = baseline
        checks = compare_facts(source, verified, facts)
        initial = verified
        if any(check.status != "preserved" for check in checks):
            verified = simplify_with_groq(
                source,
                required_facts=required + [
                    f"preserve exactly: {check.category} {check.source_fact}"
                    for check in checks if check.status != "preserved"
                ],
            )
        verified_metrics = preservation_metrics(source, verified, facts)
        records.append(
            {
                "id": row["id"],
                "split": split,
                "source": source,
                "baseline": baseline,
                "baseline_metrics": baseline_metrics,
                "baseline_checks_by_category": checks_by_category(baseline_checks),
                "verified_initial": initial,
                "verified_final": verified,
                "verified_metrics": verified_metrics,
                "verified_checks_by_category": checks_by_category(checks),
            }
        )

    result_path = OUTPUT_DIR / f"{split}_outputs.json"
    result_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = {}
    for condition in ("baseline", "verified"):
        metric_key = f"{condition}_metrics"
        total = sum(int(record[metric_key]["required_fact_count"]) for record in records)
        omissions = sum(int(record[metric_key]["omission_count"]) for record in records)
        alterations = sum(int(record[metric_key]["alteration_count"]) for record in records)
        summary[condition] = {
            "examples": len(records),
            "required_facts": total,
            "omissions": omissions,
            "omission_rate": omissions / total if total else 0.0,
            "alterations": alterations,
            "alteration_rate": alterations / total if total else 0.0,
        }

    mutation_rows = load_rows("test")
    expected: list[bool] = []
    detected: list[bool] = []
    for row in mutation_rows:
        source = str(row["source"])
        expected.append(False)
        clean_checks = compare_facts(source, source, row["required_facts"])
        detected.append(any(check.status != "preserved" for check in clean_checks))
        for mutation in row.get("mutations", []):
            expected.append(True)
            checks = compare_facts(
                source,
                str(mutation["output"]),
                {mutation["category"]: [mutation["required_fact"]]},
            )
            detected.append(any(check.status != "preserved" for check in checks))
    summary["heldout_mutation_detection"] = detection_metrics(expected, detected)
    summary["per_category"] = {}
    for category in ("number_or_dose", "frequency", "duration", "date", "negation"):
        expected_category: list[bool] = []
        detected_category: list[bool] = []
        for row in mutation_rows:
            source = str(row["source"])
            category_facts = row["required_facts"].get(category, [])
            if category_facts:
                checks = compare_facts(source, source, {category: category_facts})
                expected_category.append(False)
                detected_category.append(any(c.status != "preserved" for c in checks))
            for mutation in row.get("mutations", []):
                if mutation["category"] == category:
                    checks = compare_facts(source, str(mutation["output"]), {category: [mutation["required_fact"]]})
                    expected_category.append(True)
                    detected_category.append(any(c.status != "preserved" for c in checks))
        summary["per_category"][category] = detection_metrics(expected_category, detected_category)
    summary_path = OUTPUT_DIR / f"{split}_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"outputs": str(result_path), "metrics": summary}, indent=2))


if __name__ == "__main__":
    main()
