"""Deterministic checks for selected high-salience details in short instructions."""

from __future__ import annotations

import re
from dataclasses import dataclass

NUMBER = r"(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten)"
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}
MEASUREMENT = (
    r"(?:mg|mcg|g|ml|units?|tablets?|capsules?|puffs?|drops?|sprays?|"
    r"readings?|treatments?|(?:marked\s+)?units?)"
)
FREQUENCY = (
    r"\b(?:(?:once|twice|one time|two times|three times|four times|one|two|three|four)\s+"
    r"(?:(?:a|per|each)\s+day|daily)|every\s+\d+\s+hours?|daily|weekly|"
    r"at bedtime|every morning|each morning|each night|every night)\b"
)
DURATION = r"\b(?:for\s+)?\d+\s+(?:minutes?|hours?|days?|weeks?|months?|years?)\b"
DATE = (
    r"\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|"
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}(?:,\s*\d{4})?)\b"
)
NEGATION = r"\b(?:do not|don't|does not|doesn't|not|never|avoid|stop)\b"
MEASUREMENT_RE = re.compile(rf"\b{MEASUREMENT}\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class Check:
    category: str
    source_fact: str
    status: str
    matched_fact: str | None = None


def _normalize(text: str) -> str:
    normalized = text.casefold().replace("’", "'")
    for word, digit in NUMBER_WORDS.items():
        normalized = re.sub(rf"\b{word}\b", digit, normalized)
    normalized = re.sub(
        r"\b(?:once|one time)\s+(?:a|per|each)\s+day\b",
        "1 time per day",
        normalized,
    )
    normalized = re.sub(
        r"\b(?:twice|two times)\s+(?:a|per|each)\s+day\b",
        "2 times per day",
        normalized,
    )
    normalized = re.sub(
        r"\bonce(?:\s+(?:a|per)\s+day|\s+daily)\b",
        "1 time per day",
        normalized,
    )
    normalized = re.sub(
        r"\btwice(?:\s+(?:a|per)\s+day|\s+daily)\b",
        "2 times per day",
        normalized,
    )
    normalized = re.sub(
        r"\b(\d+)\s+times\s+(?:a|per|each)\s+day\b",
        r"\1 times per day",
        normalized,
    )
    return re.sub(r"\s+", " ", normalized).strip()


def extract_facts(text: str) -> dict[str, list[str]]:
    """Extract a deliberately limited set of literal details; not semantic parsing."""
    date_matches = list(re.finditer(DATE, text, flags=re.IGNORECASE))
    duration_matches = list(re.finditer(DURATION, text, flags=re.IGNORECASE))
    frequency_matches = list(re.finditer(FREQUENCY, text, flags=re.IGNORECASE))
    excluded_ranges = [
        match.span()
        for match in date_matches + duration_matches + frequency_matches
    ]
    number_matches = [
        match.group().strip()
        for match in re.finditer(
            rf"\b{NUMBER}(?:\s*{MEASUREMENT})?\b",
            text,
            flags=re.IGNORECASE,
        )
        if not any(
            match.start() < end and start < match.end()
            for start, end in excluded_ranges
        )
    ]
    facts: dict[str, list[str]] = {}
    facts["number_or_dose"] = list(dict.fromkeys(number_matches))
    facts["frequency"] = list(
        dict.fromkeys(match.group().strip() for match in frequency_matches)
    )
    facts["duration"] = list(
        dict.fromkeys(
            re.sub(r"^for\s+", "", match.group().strip(), flags=re.IGNORECASE)
            for match in duration_matches
        )
    )
    facts["date"] = list(
        dict.fromkeys(match.group().strip() for match in date_matches)
    )
    facts["negation"] = list(
        dict.fromkeys(
            match.strip()
            for match in re.findall(NEGATION, text, flags=re.IGNORECASE)
        )
    )
    return facts


def compare_facts(
    source_text: str,
    simplified_text: str,
    required_facts: dict[str, list[str]] | None = None,
) -> list[Check]:
    """Classify each required literal fact as preserved, omitted, or altered."""
    source_facts = required_facts or extract_facts(source_text)
    output_facts = extract_facts(simplified_text)
    checks: list[Check] = []
    for category, facts in source_facts.items():
        candidates = output_facts.get(category, [])
        normalized_candidates = {_normalize(item): item for item in candidates}
        for fact in facts:
            normalized = _normalize(fact)
            exact = re.search(
                rf"(?<!\w){re.escape(normalized)}(?!\w)",
                _normalize(simplified_text),
            )
            if normalized in normalized_candidates:
                checks.append(
                    Check(category, fact, "preserved", normalized_candidates[normalized])
                )
            elif exact:
                checks.append(
                    Check(
                        category,
                        fact,
                        "preserved",
                        normalized_candidates.get(normalized, fact),
                    )
                )
            elif len(facts) == 1 and len(candidates) == 1:
                checks.append(Check(category, fact, "altered", candidates[0]))
            elif category == "number_or_dose":
                fact_unit = MEASUREMENT_RE.search(fact)
                same_unit_required = [
                    item
                    for item in facts
                    if MEASUREMENT_RE.search(item)
                    and MEASUREMENT_RE.search(item).group().casefold()
                    == (fact_unit.group().casefold() if fact_unit else "")
                ]
                same_unit_candidates = [
                    item
                    for item in candidates
                    if fact_unit
                    and MEASUREMENT_RE.search(item)
                    and MEASUREMENT_RE.search(item).group().casefold()
                    == fact_unit.group().casefold()
                ]
                if (
                    fact_unit
                    and len(same_unit_required) == 1
                    and len(same_unit_candidates) == 1
                ):
                    checks.append(
                        Check(category, fact, "altered", same_unit_candidates[0])
                    )
                else:
                    checks.append(Check(category, fact, "omitted"))
            else:
                checks.append(Check(category, fact, "omitted"))
    return checks
