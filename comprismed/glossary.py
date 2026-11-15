"""Small prototype glossary; entries are authored paraphrases, not clinical advice."""

import re

GLOSSARY: dict[str, str] = {
    "administer": "give",
    "adverse effect": "unwanted effect",
    "as needed": "when you need it",
    "benign": "not cancer",
    "contraindicated": "should not be used in this situation",
    "discontinue use": "stop using",
    "discontinue": "stop taking",
    "dosage": "amount to take",
    "edema": "swelling",
    "elevated": "higher than usual",
    "hypertension": "high blood pressure",
    "hypotension": "low blood pressure",
    "inflammation": "swelling and irritation",
    "insomnia": "trouble sleeping",
    "intravenous": "into a vein",
    "monitor": "keep track of",
    "oral": "by mouth",
    "orally": "by mouth",
    "over-the-counter": "available without a prescription",
    "prescription": "medicine ordered by a health professional",
    "prn": "when needed",
    "twice daily": "two times each day",
    "once daily": "one time each day",
    "three times daily": "three times each day",
    "symptoms": "changes or problems you notice",
    "topical": "put on the skin",
    "underlying condition": "health problem that is already present",
}

GLOSSARY_SOURCE = "https://medlineplus.gov/healthdefinitions/"


def find_terms(text: str) -> list[tuple[str, str]]:
    """Return known glossary matches, preferring longer phrases."""
    matches = [
        (term, explanation)
        for term, explanation in GLOSSARY.items()
        if re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE)
    ]
    return sorted(matches, key=lambda match: (-len(match[0]), match[0]))


def simplify_terms(text: str) -> str:
    """Apply conservative phrase substitutions without changing other wording."""
    result = text
    for term, explanation in sorted(
        GLOSSARY.items(), key=lambda item: len(item[0]), reverse=True
    ):
        result = re.sub(
            rf"\b{re.escape(term)}\b",
            explanation,
            result,
            flags=re.IGNORECASE,
        )
    return result
