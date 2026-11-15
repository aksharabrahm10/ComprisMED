"""TF-IDF jargon classifier trained only on the included synthetic examples."""

from __future__ import annotations

import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATA = ROOT / "data" / "jargon_training.csv"


def train_classifier() -> Pipeline:
    """Train the demo classifier; its synthetic seed data is not clinical evidence."""
    with TRAINING_DATA.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or {row["label"] for row in rows} != {"jargon", "plain"}:
        raise ValueError(f"Expected both labels in {TRAINING_DATA}")
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit([row["text"] for row in rows], [row["label"] for row in rows])
    return model


def detect_jargon(text: str, model: Pipeline | None = None) -> dict[str, object]:
    classifier = model or train_classifier()
    probabilities = classifier.predict_proba([text])[0]
    classes = list(classifier.classes_)
    jargon_probability = float(probabilities[classes.index("jargon")])
    from comprismed.glossary import find_terms

    return {
        "label": "jargon" if jargon_probability >= 0.5 else "plain",
        "probability": jargon_probability,
        "term_matches": find_terms(text),
    }
