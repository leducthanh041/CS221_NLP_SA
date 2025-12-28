# UIT-ViHSD Streamlit Demo (SVM)

This project provides a Streamlit web app to demo a 3-class hate-speech classifier for UIT-ViHSD:
- 0: CLEAN
- 1: OFFENSIVE
- 2: HATE

## 1) Environment setup

### Option A — venv (recommended)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Run the app
```bash
streamlit run app.py
```

## 3) Model file
The app expects the model at:
- `./svm_best.joblib`

It is included in this zip.

## 4) Notes on "confidence"
If your model exposes `predict_proba`, the app uses it directly.
If not (common for SVM like `LinearSVC`), the app converts `decision_function` scores to a probability-like
distribution using a softmax. This is useful for UI display, but it is **not** a calibrated probability.

## 5) Explainability
The app provides:
- Confidence scores for all 3 labels
- Top TF-IDF terms in the input (highest TF-IDF weights)
- If the classifier is linear (has `coef_`), top contributing terms for the predicted class
  via `contribution = tfidf_value * class_weight`.

If your saved object is a Pipeline, the app will automatically locate the vectorizer and classifier.
