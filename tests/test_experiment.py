from comprismed.facts import compare_facts
from comprismed.metrics import detection_metrics
from experiments.run_experiment import build_examples, run_pair


def test_verification_repairs_the_same_ai_only_draft_and_rechecks_it() -> None:
    source = "Take 5 mg twice daily for 7 days."
    facts = {
        "number_or_dose": ["5 mg"],
        "frequency": ["twice daily"],
        "duration": ["7 days"],
    }
    generated_sources: list[str] = []
    repair_calls: list[tuple[str, str, dict[str, list[str]]]] = []

    def generate(text: str) -> str:
        generated_sources.append(text)
        return "Take 10 mg twice daily for 7 days."

    def repair(
        original: str, draft: str, failed: dict[str, list[str]]
    ) -> str:
        repair_calls.append((original, draft, failed))
        return "Take 5 mg twice daily for 7 days."

    result = run_pair(source, facts, generate, repair)
    assert generated_sources == [source]
    assert repair_calls == [
        (
            source,
            "Take 10 mg twice daily for 7 days.",
            {"number_or_dose": ["5 mg"]},
        )
    ]
    assert result["ai_only"]["metrics"]["alteration_count"] == 1
    assert result["ai_plus_verification"]["metrics"]["alteration_count"] == 0
    assert result["ai_plus_verification"]["metrics"]["preservation_rate"] == 1.0
    assert all(
        check.status == "preserved"
        for check in compare_facts(
            source,
            result["ai_plus_verification"]["output"],
            facts,
        )
    )


def test_detection_metrics_include_false_alarm_rate() -> None:
    metrics = detection_metrics(
        expected=[True, True, False, False],
        detected=[True, False, True, False],
    )
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["false_alarms"] == 1
    assert metrics["true_negatives"] == 1
    assert metrics["false_alarm_rate"] == 0.5


def test_examples_report_ai_only_failures_and_verified_failures() -> None:
    source = "Take 5 mg twice daily."
    facts = {"number_or_dose": ["5 mg"]}

    repaired = run_pair(
        source,
        facts,
        lambda _: "Take 10 mg twice daily.",
        lambda *_: "Take 5 mg twice daily.",
    )
    unrepaired = run_pair(
        source,
        facts,
        lambda _: "Take 10 mg twice daily.",
        lambda *_: "Take 15 mg twice daily.",
    )
    examples = build_examples(
        [
            {"id": "repaired", "source": source, **repaired},
            {"id": "unrepaired", "source": source, **unrepaired},
        ]
    )

    assert [item["id"] for item in examples["ai_only_failures"]] == [
        "repaired",
        "unrepaired",
    ]
    assert [item["id"] for item in examples["repaired_successes"]] == ["repaired"]
    assert [item["id"] for item in examples["failures"]] == ["unrepaired"]
