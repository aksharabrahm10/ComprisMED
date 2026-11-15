from comprismed.facts import compare_facts, extract_facts


def test_extracts_supported_categories() -> None:
    facts = extract_facts(
        "Take 5 mg twice daily for 7 days. Do not stop before 06/18/2026."
    )
    assert "5 mg" in facts["number_or_dose"]
    assert "twice daily" in facts["frequency"]
    assert "7 days" in facts["duration"]
    assert "06/18/2026" in facts["date"]
    assert "Do not" in facts["negation"]


def test_detects_altered_number_and_frequency_separately() -> None:
    source = "Take 5 mg twice daily."
    required = {"number_or_dose": ["5 mg"], "frequency": ["twice daily"]}
    checks = compare_facts(source, "Take 10 mg three times a day.", required)
    assert [check.status for check in checks] == ["altered", "altered"]


def test_distinguishes_omission_from_alteration() -> None:
    source = "Take 5 mg twice daily."
    required = {"number_or_dose": ["5 mg"], "frequency": ["twice daily"]}
    checks = compare_facts(source, "Take the dose.", required)
    assert [check.status for check in checks] == ["omitted", "omitted"]


def test_recognizes_frequency_variants() -> None:
    facts = extract_facts(
        "Take one capsule two times a day and take another each night."
    )
    assert "two times a day" in facts["frequency"]
    assert "each night" in facts["frequency"]


def test_equivalent_daily_frequency_phrasing_is_preserved() -> None:
    checks = compare_facts(
        "Take the medicine twice daily.",
        "Take the medicine twice each day.",
    )
    assert len(checks) == 1
    assert checks[0].status == "preserved"
