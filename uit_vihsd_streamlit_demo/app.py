# app.py
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests

from streamlit_lottie import st_lottie

from utils import (
    LABEL_ID_TO_NAME,
    extract_parts,
    get_confidence,
    top_contributing_terms,
    top_tfidf_terms,
)

# =========================
# Page config + CSS
# =========================
st.set_page_config(page_title="UIT-ViHSD Demo", layout="wide")

CSS = """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

.card {
  border: 1px solid rgba(49, 51, 63, 0.15);
  border-radius: 16px;
  padding: 16px 18px;
  background: rgba(255,255,255,0.02);
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
  transition: transform 200ms ease, box-shadow 200ms ease;
}
.card:hover { transform: translateY(-2px); box-shadow: 0 10px 26px rgba(0,0,0,0.10); }

.badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 700;
  letter-spacing: 0.2px;
  animation: pop 260ms ease-out;
}
@keyframes pop { from { transform: scale(0.96); opacity: 0.7; } to { transform: scale(1); opacity: 1; } }

.badge-clean { background: rgba(46, 204, 113, 0.15); color: #1f8f4a; border: 1px solid rgba(46, 204, 113, 0.35); }
.badge-off   { background: rgba(241, 196, 15, 0.15); color: #9a7b00; border: 1px solid rgba(241, 196, 15, 0.35); }
.badge-hate  { background: rgba(231, 76, 60, 0.15); color: #b03a2e; border: 1px solid rgba(231, 76, 60, 0.35); }

.muted { color: rgba(49,51,63,0.68); font-size: 0.92rem; }
.hr { height: 1px; background: rgba(49, 51, 63, 0.12); margin: 12px 0; }
.small { font-size: 0.9rem; color: rgba(49,51,63,0.72); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =========================
# Constants
# =========================
MODEL_PATH = "svm_best.joblib"
RANDOM_STATE = 22

# =========================
# Session state (để UI không reset khi bấm các nút khác)
# =========================
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "did_predict" not in st.session_state:
    st.session_state.did_predict = False
if "last_text" not in st.session_state:
    st.session_state.last_text = ""
if "play_tokens" not in st.session_state:
    st.session_state.play_tokens = False

# =========================
# Helpers
# =========================
def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def label_badge_html(label_id: int, label_name: str):
    klass = "badge-clean" if label_id == 0 else ("badge-off" if label_id == 1 else "badge-hate")
    return f'<span class="badge {klass}">{label_name} ({label_id})</span>'


def try_get_analyzer(vectorizer):
    if vectorizer is None:
        return None
    if hasattr(vectorizer, "build_analyzer"):
        try:
            return vectorizer.build_analyzer()
        except Exception:
            return None
    return None


def compute_scores_from_coef(clf, X_vec):
    """
    Tính score_k = w_k^T x + b_k bằng coef_ nếu có.
    Trả về vector scores shape (n_classes,) hoặc None.
    """
    if clf is None or X_vec is None or (not hasattr(clf, "coef_")):
        return None

    coef = np.asarray(clf.coef_)             # (K, D)
    intercept = np.asarray(getattr(clf, "intercept_", np.zeros((coef.shape[0],), dtype=float)))
    try:
        scores_mat = (X_vec @ coef.T)        # (1, K)
        scores = scores_mat.A1 if hasattr(scores_mat, "A1") else np.asarray(scores_mat).reshape(-1)
        scores = scores + intercept.reshape(-1)
        return scores
    except Exception:
        return None


def get_top2_margin(scores_3):
    s = np.asarray(scores_3, dtype=float).reshape(-1)
    if s.size < 3:
        return None
    idx = np.argsort(s)[::-1]
    return float(s[idx[0]] - s[idx[1]]), int(idx[0]), int(idx[1])


def safe_get_feature_names(vectorizer):
    if vectorizer is None:
        return None
    if hasattr(vectorizer, "get_feature_names_out"):
        try:
            return np.asarray(vectorizer.get_feature_names_out())
        except Exception:
            return None
    if hasattr(vectorizer, "get_feature_names"):
        try:
            return np.asarray(vectorizer.get_feature_names())
        except Exception:
            return None
    return None


def per_term_contributions(vectorizer, clf, X_vec, class_index: int, top_k: int = 20):
    """
    Trả về list (term, tfidf, weight, contribution) cho class_index.
    contribution = tfidf * w_class(term)
    """
    if vectorizer is None or clf is None or X_vec is None:
        return []

    if not hasattr(clf, "coef_"):
        return []

    names = safe_get_feature_names(vectorizer)
    if names is None:
        return []

    coef = np.asarray(clf.coef_)  # (K, D)
    if class_index >= coef.shape[0]:
        return []

    # lấy các chỉ số non-zero của input
    x = X_vec.tocsr()
    row = x[0]
    idxs = row.indices
    vals = row.data

    if idxs.size == 0:
        return []

    w = coef[class_index, idxs]
    contrib = vals * w

    df = pd.DataFrame({
        "term": names[idxs],
        "tfidf": vals.astype(float),
        "weight": w.astype(float),
        "contribution": contrib.astype(float),
    })

    df = df.sort_values("contribution", ascending=False).head(int(top_k))
    return list(df[["term", "tfidf", "weight", "contribution"]].itertuples(index=False, name=None))


def per_term_delta_between_classes(vectorizer, clf, X_vec, class_a: int, class_b: int, top_k: int = 20):
    """
    So sánh lớp A và B trên cùng input:
    delta(t) = (w_A(t) - w_B(t)) * tfidf(t)
    delta dương: token đẩy A mạnh hơn B
    """
    if vectorizer is None or clf is None or X_vec is None:
        return []

    if not hasattr(clf, "coef_"):
        return []

    names = safe_get_feature_names(vectorizer)
    if names is None:
        return []

    coef = np.asarray(clf.coef_)  # (K, D)
    if class_a >= coef.shape[0] or class_b >= coef.shape[0]:
        return []

    x = X_vec.tocsr()
    row = x[0]
    idxs = row.indices
    vals = row.data
    if idxs.size == 0:
        return []

    wa = coef[class_a, idxs]
    wb = coef[class_b, idxs]
    delta = (wa - wb) * vals

    df = pd.DataFrame({
        "term": names[idxs],
        "tfidf": vals.astype(float),
        "w_a": wa.astype(float),
        "w_b": wb.astype(float),
        "delta": delta.astype(float),
    })
    df = df.sort_values("delta", ascending=False).head(int(top_k))
    return list(df[["term", "tfidf", "w_a", "w_b", "delta"]].itertuples(index=False, name=None))


# =========================
# Header
# =========================
st.title("UIT-ViHSD Hate Speech Detection Demo")
st.caption("Nhập bình luận tiếng Việt. Mô hình dự đoán 1 trong 3 nhãn: 0=CLEAN, 1=OFFENSIVE, 2=HATE.")

with st.expander("Label definitions (paper-based)", expanded=False):
    st.markdown(
        """- **0 — CLEAN**: Không có công kích hoặc quấy rối.  
- **1 — OFFENSIVE**: Có từ thô tục hoặc công kích nhưng không nhắm trực tiếp 1 cá nhân hoặc nhóm cụ thể.  
- **2 — HATE**: Công kích nhắm trực tiếp cá nhân hoặc nhóm (theo đặc điểm, tôn giáo, quốc tịch, ...)."""
    )

LOTTIE_LOADING = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_usmfx6bp.json")
LOTTIE_SUCCESS = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_jbrw3hcz.json")

@st.cache_resource(show_spinner=True)
def load_model(path: str):
    return joblib.load(path)

# =========================
# Load model (unwrap dict)
# =========================
try:
    loaded = load_model(MODEL_PATH)
except Exception as e:
    st.error(
        "Không thể load model. Thường do mismatch phiên bản numpy hoặc scikit-learn.\n\n"
        f"Error: {type(e).__name__}: {e}"
    )
    st.stop()

vectorizer_override = None
if isinstance(loaded, dict):
    model = None
    for k in ["model", "clf", "pipeline", "estimator"]:
        if k in loaded:
            model = loaded[k]
            break
    if model is None:
        candidates = [v for v in loaded.values() if hasattr(v, "predict")]
        if not candidates:
            st.error(f"Loaded object is dict but no value has predict. Keys={list(loaded.keys())}")
            st.stop()
        model = candidates[0]
    for k in ["vectorizer", "tfidf", "tfidf_vectorizer"]:
        if k in loaded:
            vectorizer_override = loaded[k]
            break
else:
    model = loaded

parts = extract_parts(model)
if vectorizer_override is not None:
    parts.vectorizer = vectorizer_override

analyzer = try_get_analyzer(parts.vectorizer)

# =========================
# Layout
# =========================
left_col, right_col = st.columns(2, gap="large")

# =========================
# Left: input form
# =========================
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Input & Output")

    with st.form("predict_form", clear_on_submit=False):
        text = st.text_area(
            "Input text",
            value=st.session_state.last_text,
            height=220,
            placeholder="Nhập bình luận tiếng Việt ở đây...",
        )

        c1, c2 = st.columns(2)
        with c1:
            topk_tfidf = st.number_input("Top-K TF-IDF terms", min_value=5, max_value=50, value=15, step=1)
        with c2:
            topk_contrib = st.number_input("Top-K contributing terms", min_value=5, max_value=50, value=15, step=1)

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        slow_mode = st.toggle("Step-by-step (tua chậm)", value=True)
        speed = st.slider("Tốc độ tua (giây / bước)", min_value=1, max_value=4, value=2, step=1)
        reveal_k = st.slider("Số token tua dần (top contributors)", min_value=5, max_value=40, value=15, step=1)

        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Predict and store in session_state
# =========================
if submitted:
    if not text or not text.strip():
        with left_col:
            st.warning("Please enter a non-empty text.")
    else:
        st.session_state.last_text = text

        if LOTTIE_LOADING is not None:
            with left_col:
                st_lottie(LOTTIE_LOADING, height=110, key="loading_anim_submit")

        tokens = None
        if analyzer is not None:
            try:
                tokens = analyzer(text)
            except Exception:
                tokens = None

        X_vec = None
        if parts.vectorizer is not None:
            try:
                X_vec = parts.vectorizer.transform([text])
            except Exception:
                X_vec = None

        # Predict + confidence-like
        pred, conf = get_confidence(model, [text], vectorizer=parts.vectorizer)
        pred_id = int(pred[0])
        pred_name = LABEL_ID_TO_NAME.get(pred_id, str(pred_id))

        conf_row = np.asarray(conf[0], dtype=float)
        if conf_row.size < 3:
            conf_row = np.pad(conf_row, (0, 3 - conf_row.size), constant_values=0.0)
        conf_row = conf_row[:3]

        df_conf = pd.DataFrame(
            {"label_id": [0, 1, 2],
             "label_name": [LABEL_ID_TO_NAME[i] for i in [0, 1, 2]],
             "score": conf_row}
        ).sort_values("score", ascending=False)

        # Raw SVM scores (nếu lấy được từ coef_)
        raw_scores = None
        if X_vec is not None and parts.classifier is not None:
            raw_scores = compute_scores_from_coef(parts.classifier, X_vec)

        margin_info = None
        if raw_scores is not None and np.asarray(raw_scores).size >= 3:
            m, top1_id, top2_id = get_top2_margin(raw_scores[:3])
            margin_info = {"margin": m, "top1_id": top1_id, "top2_id": top2_id, "scores3": raw_scores[:3]}

        # TF-IDF top terms
        tfidf_terms = None
        if parts.vectorizer is not None:
            tfidf_terms = top_tfidf_terms(parts.vectorizer, text, top_k=int(topk_tfidf))

        # Contributions for predicted class (linear)
        contrib_rich = []
        if X_vec is not None and parts.vectorizer is not None and parts.classifier is not None:
            contrib_rich = per_term_contributions(parts.vectorizer, parts.classifier, X_vec, pred_id, top_k=int(topk_contrib))

        # Compare top1 vs top2 class (để “thuyết phục” hơn)
        delta_terms = []
        if margin_info is not None:
            a = int(margin_info["top1_id"])
            b = int(margin_info["top2_id"])
            if X_vec is not None and parts.vectorizer is not None and parts.classifier is not None:
                delta_terms = per_term_delta_between_classes(parts.vectorizer, parts.classifier, X_vec, a, b, top_k=int(topk_contrib))

        st.session_state.analysis = {
            "text": text,
            "tokens": tokens,
            "X_vec": X_vec,
            "pred_id": pred_id,
            "pred_name": pred_name,
            "df_conf": df_conf,
            "raw_scores": raw_scores,
            "margin_info": margin_info,
            "tfidf_terms": tfidf_terms,
            "contrib_rich": contrib_rich,
            "delta_terms": delta_terms,
        }
        st.session_state.did_predict = True
        st.session_state.play_tokens = False  # reset

# =========================
# Left output
# =========================
with left_col:
    A = st.session_state.analysis
    if st.session_state.did_predict and A is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Prediction")
        st.markdown(label_badge_html(int(A["pred_id"]), str(A["pred_name"])), unsafe_allow_html=True)

        df_conf = A["df_conf"]
        top1 = float(df_conf.iloc[0]["score"])
        st.markdown(f'<div class="muted">Top-1 confidence (hiển thị): <b>{top1:.4f}</b></div>', unsafe_allow_html=True)

        mi = A.get("margin_info", None)
        if mi is not None:
            st.markdown(
                f'<div class="muted">SVM margin (top1 - top2): <b>{float(mi["margin"]):.6f}</b></div>',
                unsafe_allow_html=True
            )

        if LOTTIE_SUCCESS is not None:
            st_lottie(LOTTIE_SUCCESS, height=100, key="success_anim_left")

        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Right side tabs
# =========================
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Giải thích SVM thật rõ trên đúng input này")

    A = st.session_state.analysis
    if not st.session_state.did_predict or A is None:
        st.info("Nhập text và nhấn Predict để xem giải thích từng bước.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        tab1, tab2, tab3 = st.tabs(["Confidence", "SVM mechanism (step-by-step)", "Token evidence (animated)"])

        # -------- Tab 1
        with tab1:
            st.markdown("### Confidence scores (3 labels)")
            df_conf = A["df_conf"]
            st.dataframe(df_conf, use_container_width=True, hide_index=True)

            df_plot = df_conf.sort_values("label_id")
            fig = px.bar(
                df_plot,
                x="label_name",
                y="score",
                text="score",
                range_y=[0, max(1.0, float(df_plot["score"].max()) * 1.15)],
            )
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
            st.plotly_chart(fig, use_container_width=True, key="plot_conf")

            st.markdown(
                '<div class="muted">Ghi chú: nếu model không có predict_proba, app chuyển decision_function sang dạng score để hiển thị.</div>',
                unsafe_allow_html=True
            )

        # -------- Tab 2
        with tab2:
            st.markdown("### Cơ chế: từ text -> TF-IDF vector x -> SVM scores -> nhãn cuối")

            text_in = A["text"]
            tokens = A["tokens"]
            X_vec = A["X_vec"]
            raw_scores = A.get("raw_scores", None)
            mi = A.get("margin_info", None)

            prog = st.progress(0, text="Chuẩn bị tua chậm...")
            ph = st.empty()

            def step(i, total, title, body_md=None, df=None):
                prog.progress(int(i / total * 100), text=title)
                with ph.container():
                    st.markdown(f"#### Bước {i} trên {total}: {title}")
                    if body_md:
                        st.markdown(body_md)
                    if df is not None:
                        st.dataframe(df, use_container_width=True, hide_index=True)
                if slow_mode:
                    time.sleep(float(speed))

            total_steps = 4

            # Step 1: tokenize
            if tokens is None:
                token_text = "- Không lấy được tokens từ analyzer."
            else:
                preview = tokens[:80]
                token_text = f"- Số token theo analyzer: {len(tokens)}\n- Preview: `{preview}`"
            step(1, total_steps, "Input và tokens", body_md=f"**Input:** `{text_in}`\n\n{token_text}")

            # Step 2: TF-IDF stats + top terms
            if X_vec is None or parts.vectorizer is None:
                step(2, total_steps, "TF-IDF vector hoá", body_md="- Không vectorize được TF-IDF.")
            else:
                nnz = int(X_vec.nnz)
                n_features = int(X_vec.shape[1])
                l2 = float(np.sqrt(X_vec.multiply(X_vec).sum()))
                sparsity = 1.0 - (nnz / max(1, n_features))
                df_tfidf = pd.DataFrame(A["tfidf_terms"] or [], columns=["term", "tfidf_weight"])

                md2 = (
                    "TF-IDF tạo vector đặc trưng x rất thưa\n\n"
                    f"- n_features = {n_features}\n"
                    f"- nnz = {nnz}\n"
                    f"- sparsity = {sparsity:.6f}\n"
                    f"- norm_2 = {l2:.6f}\n\n"
                    "Top TF-IDF terms:"
                )
                step(2, total_steps, "TF-IDF tạo vector x", body_md=md2, df=df_tfidf)

            # Step 3: scores
            if raw_scores is None or np.asarray(raw_scores).size < 3:
                step(3, total_steps, "SVM tính score", body_md="- Không lấy được score theo coef. (Có thể classifier không có coef_).")
            else:
                s3 = np.asarray(raw_scores).reshape(-1)[:3]
                df_s = pd.DataFrame({
                    "label_id": [0, 1, 2],
                    "label_name": [LABEL_ID_TO_NAME[i] for i in [0, 1, 2]],
                    "svm_score": s3
                }).sort_values("svm_score", ascending=False)

                md3 = (
                    "SVM tuyến tính tính điểm từng lớp theo\n\n"
                    "$$score_k = w_k^T x + b_k$$\n\n"
                    "Bảng score:"
                )
                step(3, total_steps, "Tính score cho 3 lớp", body_md=md3, df=df_s)

            # Step 4: argmax + margin
            pred_id = int(A["pred_id"])
            pred_name = str(A["pred_name"])

            if mi is None:
                md4 = f"Nhãn cuối theo argmax score: **{pred_name} ({pred_id})**"
            else:
                md4 = (
                    "Chọn nhãn cuối theo\n\n"
                    "$$y_hat = argmax_k score_k$$\n\n"
                    f"- Top1: {LABEL_ID_TO_NAME[int(mi['top1_id'])]} ({int(mi['top1_id'])})\n"
                    f"- Top2: {LABEL_ID_TO_NAME[int(mi['top2_id'])]} ({int(mi['top2_id'])})\n"
                    f"- Margin = {float(mi['margin']):.6f}\n\n"
                    f"=> Kết luận: **{pred_name} ({pred_id})**"
                )
            step(4, total_steps, "Argmax -> nhãn cuối", body_md=md4)

            prog.progress(100, text="Hoàn tất.")

        # -------- Tab 3: animated evidence
        with tab3:
            st.markdown("### Evidence theo token (animation)")

            pred_id = int(A["pred_id"])
            pred_name = str(A["pred_name"])
            contrib = A.get("contrib_rich", [])
            delta = A.get("delta_terms", [])
            mi = A.get("margin_info", None)

            colA, colB = st.columns([1, 1])
            with colA:
                play = st.button("Play token reveal", type="primary", use_container_width=True)
                if play:
                    st.session_state.play_tokens = True
            with colB:
                st.markdown('<div class="small">Bước token chạy nhanh gấp 10 lần speed ở cột trái.</div>', unsafe_allow_html=True)

            if not contrib:
                st.info("Không có contribution details. Thường do classifier không có coef_.")
            else:
                st.markdown(
                    "Định nghĩa đóng góp (với lớp dự đoán)\n\n"
                    "$$contribution_t = tfidf_t * w_{y_hat,t}$$"
                )

                k = int(min(reveal_k, len(contrib)))
                fast_sleep = float(speed) / 10.0

                table_ph = st.empty()
                chart_ph = st.empty()
                cum_ph = st.empty()

                # Chuẩn bị data
                run_rows = []
                running = 0.0

                if st.session_state.play_tokens:
                    for i in range(k):
                        term, tfidf_v, w_v, c_v = contrib[i]
                        running += float(c_v)
                        run_rows.append([term, float(tfidf_v), float(w_v), float(c_v), float(running)])

                        df_run = pd.DataFrame(run_rows, columns=["term", "tfidf", "weight", "contribution", "cumulative"])
                        table_ph.dataframe(df_run, use_container_width=True, hide_index=True)

                        # bar chart contribution
                        figc = px.bar(df_run, x="term", y="contribution", text="contribution")
                        figc.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                        figc.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
                        chart_ph.plotly_chart(figc, use_container_width=True, key=f"contrib_bar_{i}")

                        cum_ph.markdown(f"**Cumulative contribution (top {i+1} terms):** {running:.6f}")

                        if slow_mode:
                            time.sleep(fast_sleep)

                    st.session_state.play_tokens = False
                else:
                    # show static top-k
                    df_static = pd.DataFrame(
                        contrib[:k],
                        columns=["term", "tfidf", "weight", "contribution"]
                    )
                    st.dataframe(df_static, use_container_width=True, hide_index=True)

            # Compare top1 vs top2 (rất hữu ích để “giải thích với giáo sư”)
            if mi is not None and delta:
                a = int(mi["top1_id"])
                b = int(mi["top2_id"])
                st.markdown("### So sánh 2 lớp mạnh nhất (top1 vs top2)")
                st.markdown(
                    "Token nào khiến top1 thắng top2 được thể hiện bởi\n\n"
                    "$$delta_t = (w_{top1,t} - w_{top2,t}) * tfidf_t$$"
                )
                df_delta = pd.DataFrame(delta, columns=["term", "tfidf", "w_top1", "w_top2", "delta"])
                st.dataframe(df_delta, use_container_width=True, hide_index=True)

                figd = px.bar(df_delta.head(min(20, len(df_delta))), x="term", y="delta", text="delta")
                figd.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                figd.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(figd, use_container_width=True, key="delta_bar")

            st.markdown(
                f'<div class="muted">Gợi ý thuyết minh: input được gán nhãn <b>{pred_name}</b> vì tổng contribution (và delta so với lớp kế tiếp) '
                'bị chi phối bởi các token có TF-IDF lớn và trọng số w tương ứng kéo score lớp đó lên.</div>',
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)
