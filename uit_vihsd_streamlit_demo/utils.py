from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


LABEL_ID_TO_NAME = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}


@dataclass
class ModelParts:
    raw_model: Any
    vectorizer: Optional[Any]  # e.g., TfidfVectorizer
    classifier: Any            # e.g., LinearSVC / SVC / SGDClassifier / LogisticRegression


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return np.array([1.0])
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    if s == 0 or not np.isfinite(s):
        # fallback: uniform
        return np.ones_like(x) / max(1, x.size)
    return e / s


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def extract_parts(model: Any) -> ModelParts:
    """Try to extract vectorizer + classifier from a sklearn Pipeline-like object.

    Supported patterns:
    - Pipeline with steps containing a vectorizer (has transform + get_feature_names_out)
      and a classifier (has predict)
    - Direct classifier that also vectorizes internally (rare) => vectorizer=None
    """
    vectorizer = None
    classifier = model

    steps = None
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.items())
    elif hasattr(model, "steps"):
        steps = list(model.steps)

    if steps:
        # pick last step as classifier
        classifier = steps[-1][1]
        # find the last component that looks like a vectorizer/transformer producing features
        for _, step in steps[:-1]:
            if hasattr(step, "transform") and hasattr(step, "get_feature_names_out"):
                vectorizer = step
        # Some pipelines use ColumnTransformer; try deeper search
        if vectorizer is None:
            for _, step in steps[:-1]:
                if hasattr(step, "transform") and hasattr(step, "get_feature_names_out"):
                    vectorizer = step

    return ModelParts(raw_model=model, vectorizer=vectorizer, classifier=classifier)


def get_confidence(model: Any, X_text: List[str], vectorizer: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (pred_label_ids, confidence_matrix).

    If `model` is a Pipeline, it can handle raw text directly.
    If `model` is a bare classifier, we must transform text -> features using `vectorizer`.
    """
    # Decide what to feed into model.predict / decision_function / predict_proba
    X_in = X_text
    if vectorizer is not None and not hasattr(model, "named_steps"):
        # bare classifier case
        X_in = vectorizer.transform(X_text)

    # Predict
    pred = model.predict(X_in)

    # Probability-like scores
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_in)
        return np.asarray(pred), np.asarray(proba)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X_in))

        if scores.ndim == 1:
            p1 = _sigmoid(scores)
            p0 = 1.0 - p1
            proba2 = np.vstack([p0, p1]).T
            proba = np.hstack([proba2, np.zeros((proba2.shape[0], 1))])
            return np.asarray(pred), proba

        proba = np.vstack([_softmax(row) for row in scores])
        return np.asarray(pred), proba

    # Fallback
    n = len(X_text)
    proba = np.ones((n, 3), dtype=float) / 3.0
    return np.asarray(pred), proba

def top_tfidf_terms(vectorizer: Any, text: str, top_k: int = 15) -> List[Tuple[str, float]]:
    """Top-k terms by TF-IDF weight for a single text."""
    if vectorizer is None:
        return []
    X = vectorizer.transform([text])
    if X.shape[1] == 0:
        return []
    # get_feature_names_out is standard in recent sklearn
    try:
        fn = vectorizer.get_feature_names_out()
    except Exception:
        return []

    row = X.tocoo()
    pairs = [(fn[j], float(v)) for j, v in zip(row.col, row.data)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k]


def top_contributing_terms(parts: ModelParts, text: str, pred_class_index: int, top_k: int = 15) -> List[Tuple[str, float]]:
    """For linear models with coef_, compute contribution = tfidf_value * weight for predicted class.

    Returns top positive contributors.
    """
    if parts.vectorizer is None:
        return []
    clf = parts.classifier
    if not hasattr(clf, "coef_"):
        return []

    X = parts.vectorizer.transform([text])
    try:
        fn = parts.vectorizer.get_feature_names_out()
    except Exception:
        return []

    coef = np.asarray(clf.coef_)
    # Common shapes:
    # - (n_classes, n_features) for multinomial/ovr linear
    # - (1, n_features) for binary
    if coef.ndim != 2 or coef.shape[1] != X.shape[1]:
        return []

    class_row = 0
    if coef.shape[0] >= 3:
        class_row = int(pred_class_index)
    else:
        class_row = 0

    w = coef[class_row]  # (n_features,)
    row = X.tocoo()
    contrib = []
    for j, v in zip(row.col, row.data):
        c = float(v) * float(w[j])
        contrib.append((fn[j], c))

    # sort by contribution descending (positive first)
    contrib.sort(key=lambda x: x[1], reverse=True)
    # keep only positive contributors for interpretability
    contrib = [(t, s) for t, s in contrib if s > 0]
    return contrib[:top_k]
