# app.py
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    LABEL_ID_TO_NAME,
    extract_parts,
    get_confidence,
    top_contributing_terms,
    top_tfidf_terms,
)

st.set_page_config(page_title="UIT-ViHSD Demo", layout="wide")

st.title("UIT-ViHSD Hate Speech Detection Demo")
st.caption("Nhập bình luận tiếng Việt. Mô hình dự đoán 1 trong 3 nhãn: 0=CLEAN, 1=OFFENSIVE, 2=HATE.")

with st.expander("Label definitions (paper-based)", expanded=False):
    st.markdown(
        """- **0 — CLEAN**: Không có công kích/quấy rối.
- **1 — OFFENSIVE**: Có từ thô tục/công kích nhưng không nhắm trực tiếp 1 cá nhân/nhóm cụ thể.
- **2 — HATE**: Công kích nhắm trực tiếp cá nhân/nhóm (theo đặc điểm, tôn giáo, quốc tịch, ...)."""
    )

MODEL_PATH = "svm_best.joblib"


@st.cache_resource(show_spinner=True)
def load_model(path: str):
    return joblib.load(path)


# ===== Load model (and unwrap if dict) =====
try:
    loaded = load_model(MODEL_PATH)
except Exception as e:
    st.error(
        "Không thể load model. Thường do mismatch phiên bản numpy/scikit-learn.\n\n"
        f"Error: {type(e).__name__}: {e}"
    )
    st.stop()

# Optional: quick debug
# st.write("Loaded type:", type(loaded))
# if isinstance(loaded, dict):
#     st.write("Loaded dict keys:", list(loaded.keys()))

vectorizer_override = None

if isinstance(loaded, dict):
    # Try typical keys first
    model = None
    for k in ["model", "clf", "pipeline", "estimator"]:
        if k in loaded:
            model = loaded[k]
            break

    # Otherwise pick any value that has predict()
    if model is None:
        candidates = [v for v in loaded.values() if hasattr(v, "predict")]
        if not candidates:
            st.error(f"Loaded object is dict but no value has .predict(). Keys={list(loaded.keys())}")
            st.stop()
        model = candidates[0]

    # Prefer vectorizer inside dict if exists
    for k in ["vectorizer", "tfidf", "tfidf_vectorizer"]:
        if k in loaded:
            vectorizer_override = loaded[k]
            break
else:
    model = loaded

parts = extract_parts(model)
if vectorizer_override is not None:
    parts.vectorizer = vectorizer_override

# ===== 2-column layout =====
left_col, right_col = st.columns(2, gap="large")

with left_col:
    st.subheader("Input & Output")

    text = st.text_area(
        "Input text",
        value="",
        height=220,
        placeholder="Nhập bình luận tiếng Việt ở đây...",
    )

    c1, c2 = st.columns(2)
    with c1:
        topk_tfidf = st.number_input("Top-K TF-IDF terms", min_value=5, max_value=50, value=15, step=1)
    with c2:
        topk_contrib = st.number_input("Top-K contributing terms", min_value=5, max_value=50, value=15, step=1)

    run = st.button("Predict", type="primary", use_container_width=True)

# ===== Run prediction once =====
pred_id, pred_name, df_conf = None, None, None

if run:
    if not text or not text.strip():
        with left_col:
            st.warning("Please enter a non-empty text.")
    else:
        # IMPORTANT: pass vectorizer for bare classifiers
        pred, conf = get_confidence(model, [text], vectorizer=parts.vectorizer)

        pred_id = int(pred[0])
        pred_name = LABEL_ID_TO_NAME.get(pred_id, str(pred_id))

        conf_row = np.asarray(conf[0], dtype=float)
        if conf_row.size < 3:
            conf_row = np.pad(conf_row, (0, 3 - conf_row.size), constant_values=0.0)
        conf_row = conf_row[:3]

        df_conf = pd.DataFrame(
            {
                "label_id": [0, 1, 2],
                "label_name": [LABEL_ID_TO_NAME[i] for i in [0, 1, 2]],
                "score": conf_row,
            }
        ).sort_values("score", ascending=False)

# ===== Render output (left) =====
with left_col:
    if pred_id is not None:
        st.markdown("### Prediction")
        st.write(f"**Predicted label:** `{pred_id}` — **{pred_name}**")

# ===== Render explainability + confidence (right) =====
with right_col:
    st.subheader("Explainability")

    if pred_id is None:
        st.info("Nhập text và nhấn Predict để xem confidence + giải thích.")
    else:
        st.markdown("### Confidence scores (3 labels)")
        st.dataframe(df_conf, use_container_width=True, hide_index=True)
        st.bar_chart(df_conf.set_index("label_name")["score"])

        st.markdown("### Why this label?")
        st.markdown(
            """Giải thích dựa trên:
- **Top TF-IDF terms**: các token/cụm từ nổi bật nhất trong input theo TF-IDF.
- **Top contributing terms** (nếu model tuyến tính có `coef_`): xấp xỉ đóng góp theo
  `contribution = tfidf_value × class_weight` cho nhãn dự đoán."""
        )

        with st.expander("Top TF-IDF terms in input", expanded=True):
            if parts.vectorizer is None:
                st.info("Vectorizer không được detect. Không thể hiển thị TF-IDF explanation.")
            else:
                tfidf_terms = top_tfidf_terms(parts.vectorizer, text, top_k=int(topk_tfidf))
                if not tfidf_terms:
                    st.info("Không có TF-IDF terms (vocabulary rỗng hoặc vectorizer không tương thích).")
                else:
                    st.dataframe(
                        pd.DataFrame(tfidf_terms, columns=["term", "tfidf_weight"]),
                        use_container_width=True,
                        hide_index=True,
                    )

        with st.expander("Top contributing terms for predicted class", expanded=True):
            contrib_terms = top_contributing_terms(parts, text, pred_class_index=pred_id, top_k=int(topk_contrib))
            if not contrib_terms:
                st.info(
                    "Không có contribution explanation. Lý do thường gặp: "
                    "classifier không tuyến tính (không có coef_) hoặc không detect được vectorizer."
                )
            else:
                st.dataframe(
                    pd.DataFrame(contrib_terms, columns=["term", "contribution"]),
                    use_container_width=True,
                    hide_index=True,
                )

        with st.expander("Model & pipeline info", expanded=False):
            st.write("**Saved object type:**", type(model))
            st.write("**Vectorizer detected:**", type(parts.vectorizer) if parts.vectorizer is not None else None)
            st.write("**Classifier detected:**", type(parts.classifier))
