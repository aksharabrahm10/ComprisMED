from comprismed.glossary import find_terms, simplify_terms
from comprismed.pipeline import simplify


def test_glossary_finds_terms_and_conservative_substitution() -> None:
    text = "Discontinue oral medication."
    assert {term for term, _ in find_terms(text)} == {"discontinue", "oral"}
    assert simplify_terms(text) == "stop taking by mouth medication."


def test_local_glossary_uses_safe_contextual_phrases() -> None:
    text = "Administer 5 mg orally. Discontinue use if edema develops."
    simplified = simplify_terms(text)
    assert simplified == "give 5 mg by mouth. stop using if swelling develops."


def test_local_simplification_returns_explicit_disclosure() -> None:
    result = simplify(
        "Take 5 mg twice daily for 7 days. Do not stop early.",
        mode="verified",
        use_api=False,
    )
    assert result["output"]
    assert result["mode"] == "verified"
    assert result["is_verified"] is True
    assert "not proof of clinical correctness" in result["note"]
