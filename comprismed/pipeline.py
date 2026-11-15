"""comprisMED simplification and verification workflow."""

from __future__ import annotations

from comprismed.facts import compare_facts, extract_facts
from comprismed.glossary import simplify_terms
from comprismed.groq import simplify_with_groq


def simplify(
    source: str, mode: str = "verified", use_api: bool = True, language: str = "English"
) -> dict[str, object]:
    if not source.strip():
        raise ValueError("Enter the instruction text first.")
    source_facts = extract_facts(source)
    required = [fact for facts in source_facts.values() for fact in facts]
    if use_api:
        initial = simplify_with_groq(source, language)
    else:
        initial = simplify_terms(source)
    initial_checks = compare_facts(source, initial, source_facts)
    if mode == "verified":
        failed = [check for check in initial_checks if check.status != "preserved"]
        if failed and use_api:
            retry = simplify_with_groq(
                source,
                language,
                required + [
                    f"previous output failed to preserve {c.category}: {c.source_fact}"
                    for c in failed
                ],
            )
            retry_checks = compare_facts(source, retry, source_facts)
            output = retry
            checks = retry_checks
        else:
            output = initial
            checks = initial_checks
    else:
        output = initial
        checks = initial_checks
    review_required = any(check.status != "preserved" for check in checks)
    return {
        "source": source,
        "initial_output": initial,
        "output": output,
        "mode": mode,
        "checks": checks,
        "review_required": review_required,
        "is_verified": not review_required,
        "source_facts": source_facts,
        "note": (
            "All supported literal checks passed; this is not proof of clinical correctness."
            if not review_required
            else "A supported detail may be missing or changed. Ask a clinician or pharmacist."
        ),
    }
