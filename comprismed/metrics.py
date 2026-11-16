"""Fact-level preservation and verifier metrics."""

from __future__ import annotations

from collections import defaultdict

from comprismed.facts import Check, compare_facts


def preservation_metrics(
    source: str, output: str, required_facts: dict[str, list[str]]
) -> dict[str, int | float]:
    checks = compare_facts(source, output, required_facts)
    total = len(checks)
    omissions = sum(check.status == "omitted" for check in checks)
    alterations = sum(check.status == "altered" for check in checks)
    preserved = sum(check.status == "preserved" for check in checks)
    return {
        "required_fact_count": total,
        "preserved_count": preserved,
        "omission_count": omissions,
        "alteration_count": alterations,
        "preservation_rate": preserved / total if total else 0.0,
        "omission_rate": omissions / total if total else 0.0,
        "alteration_rate": alterations / total if total else 0.0,
    }


def detection_metrics(expected: list[bool], detected: list[bool]) -> dict[str, int | float]:
    if len(expected) != len(detected):
        raise ValueError("Expected labels and detections must have equal length.")
    tp = sum(actual and guess for actual, guess in zip(expected, detected))
    fp = sum(not actual and guess for actual, guess in zip(expected, detected))
    fn = sum(actual and not guess for actual, guess in zip(expected, detected))
    tn = sum(not actual and not guess for actual, guess in zip(expected, detected))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_alarms": fp,
        "true_positives": tp,
        "false_negatives": fn,
        "true_negatives": tn,
    }


def checks_by_category(checks: list[Check]) -> dict[str, dict[str, int | float]]:
    grouped: dict[str, list[Check]] = defaultdict(list)
    for check in checks:
        grouped[check.category].append(check)
    result = {}
    for category, items in sorted(grouped.items()):
        total = len(items)
        omissions = sum(item.status == "omitted" for item in items)
        alterations = sum(item.status == "altered" for item in items)
        preserved = sum(item.status == "preserved" for item in items)
        result[category] = {
            "required_facts": total,
            "preserved": preserved,
            "omissions": omissions,
            "alterations": alterations,
            "preservation_rate": preserved / total if total else 0.0,
            "omission_rate": omissions / total if total else 0.0,
            "alteration_rate": alterations / total if total else 0.0,
        }
    return result
