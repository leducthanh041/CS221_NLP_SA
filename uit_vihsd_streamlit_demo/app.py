# app.py
import time
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests

from streamlit_lottie import st_lottie
from sklearn.pipeline import Pipeline


# =========================
# Config
# =========================
st.set_page_config(page_title="UIT-ViHSD — TF-IDF + MultinomialNB Demo", layout="wide")

MODEL_PATH = "final_best_mnb_tfidf.joblib"
INFO_PATH = "final_best_mnb_tfidf_info.json"

LABEL_ID_TO_NAME = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}


# =========================
# CSS (card + subtle animations)
# =========================
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


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    if s == 0 or not np.isfinite(s):
        return np.ones_like(x) / max(1, x.size)
    return e / s


def safe_log(x, eps=1e-12):
    return np.log(np.maximum(x, eps))


def extract_vectorizer_and_nb(loaded):
    """
    Support:
      - Pipeline(tfidf, clf)
      - dict contains pipeline/model/vectorizer/clf
      - direct clf (rare)
    Return: (pipeline_or_clf, vectorizer, clf)
    """
    vectorizer = None
    clf = None
    model = loaded

    # unwrap dict
    if isinstance(loaded, dict):
        # pick pipeline/model/clf-like
        for k in ["pipeline", "model", "estimator", "clf"]:
            if k in loaded:
                model = loaded[k]
                break
        # try vectorizer inside dict
        for k in ["vectorizer", "tfidf", "tfidf_vectorizer"]:
            if k in loaded:
                vectorizer = loaded[k]
                break
        # try clf inside dict
        for k in ["clf", "classifier", "mnb", "nb"]:
            if k in loaded and hasattr(loaded[k], "predict"):
                clf = loaded[k]
                break

    # pipeline
    if isinstance(model, Pipeline):
        # try typical step names
        steps = dict(model.named_steps)
        for k in ["tfidf", "vectorizer", "vect"]:
            if k in steps:
                vectorizer = steps[k]
                break
        # classifier
        for k in ["clf", "classifier", "mnb", "nb"]:
            if k in steps:
                clf = steps[k]
                break
        # fallback: last step
        if clf is None:
            clf = model.steps[-1][1]
        return model, vectorizer, clf

    # not pipeline
    if clf is None and hasattr(model, "predict"):
        clf = model

    return model, vectorizer, clf


def tfidf_top_terms(vectorizer, x_vec_row, top_k=15):
    """
    x_vec_row: sparse row (1, n_features)
    Return list of (term, tfidf_value)
    """
    if vectorizer is None or x_vec_row is None:
        return []
    try:
        feat_names = vectorizer.get_feature_names_out()
        row = x_vec_row.tocoo()
        if row.nnz == 0:
            return []
        vals = row.data
        idxs = row.col
        order = np.argsort(vals)[::-1][: int(top_k)]
        out = [(str(feat_names[idxs[i]]), float(vals[i])) for i in order]
        return out
    except Exception:
        return []


def nb_class_term_table(vectorizer, clf, class_index: int, top_k=25):
    """
    Show top terms by P(t|c) (equiv: highest feature_log_prob_ for class c).
    Return dataframe columns: term, log_P_t_given_c, P_t_given_c
    """
    if vectorizer is None or clf is None:
        return pd.DataFrame(columns=["term", "log_P_t_given_c", "P_t_given_c"])
    if not hasattr(clf, "feature_log_prob_"):
        return pd.DataFrame(columns=["term", "log_P_t_given_c", "P_t_given_c"])
    try:
        feat_names = vectorizer.get_feature_names_out()
        logp = np.asarray(clf.feature_log_prob_)[class_index]  # (n_features,)
        order = np.argsort(logp)[::-1][: int(top_k)]
        terms = [str(feat_names[i]) for i in order]
        logps = logp[order]
        ps = np.exp(logps)
        df = pd.DataFrame({"term": terms, "log_P_t_given_c": logps, "P_t_given_c": ps})
        return df
    except Exception:
        return pd.DataFrame(columns=["term", "log_P_t_given_c", "P_t_given_c"])


def nb_explain_log_posterior(vectorizer, clf, x_vec_row, top_k=20):
    """
    Compute per-class:
      log_prior[c]
      log_likelihood[c] = sum_i x_i * log P(w_i|c)
      log_posterior_unnorm[c] = log_prior + log_likelihood

    Also return token-level contributions for each class:
      contrib_i(c) = x_i * logP_i_c
    but we only keep top tokens by |delta| or by contribution for predicted class.
    """
    if vectorizer is None or clf is None or x_vec_row is None:
        return None

    if not hasattr(clf, "feature_log_prob_") or not hasattr(clf, "class_log_prior_"):
        return None

    feat_log_prob = np.asarray(clf.feature_log_prob_)  # (C, V)
    log_prior = np.asarray(clf.class_log_prior_)       # (C,)

    # sparse row
    row = x_vec_row.tocoo()
    if row.nnz == 0:
        # still can compute posterior = prior
        C = int(log_prior.size)
        log_like = np.zeros((C,), dtype=float)
        log_post = log_prior + log_like
        return {
            "log_prior": log_prior,
            "log_like": log_like,
            "log_post": log_post,
            "row": row,
            "token_info": [],
        }

    idxs = row.col
    vals = row.data  # TF-IDF weights (non-negative)

    # compute per class log-likelihood
    # log_like[c] = sum_j vals[j] * feat_log_prob[c, idxs[j]]
    log_like = (feat_log_prob[:, idxs] * vals.reshape(1, -1)).sum(axis=1)
    log_post = log_prior + log_like

    # token-level table for predicted class (and deltas with runner-up)
    try:
        feat_names = vectorizer.get_feature_names_out()
        tokens = [str(feat_names[i]) for i in idxs]
    except Exception:
        tokens = [str(i) for i in idxs]

    token_info = []
    for t, fid, v in zip(tokens, idxs, vals):
        per_class = feat_log_prob[:, fid]  # (C,)
        token_info.append(
            {
                "term": t,
                "tfidf": float(v),
                "logP_given_class": per_class.astype(float).tolist(),
            }
        )

    return {
        "log_prior": log_prior.astype(float),
        "log_like": np.asarray(log_like, dtype=float),
        "log_post": np.asarray(log_post, dtype=float),
        "row": row,
        "token_info": token_info,
    }


def build_token_evidence_tables(vectorizer, clf, x_vec_row, pred_id: int, alt_id: int, top_k=20):
    """
    Return:
      - df_pred: top tokens by contribution to predicted class
      - df_delta: top tokens by delta between pred and alt (why pred beats alt)
    Definitions:
      contribution_pred(t) = tfidf(t) * logP(t|pred)
      delta_pred_alt(t) = tfidf(t) * (logP(t|pred) - logP(t|alt))
    """
    if vectorizer is None or clf is None or x_vec_row is None:
        return pd.DataFrame(), pd.DataFrame()
    if not hasattr(clf, "feature_log_prob_"):
        return pd.DataFrame(), pd.DataFrame()

    feat_log_prob = np.asarray(clf.feature_log_prob_)  # (C, V)
    row = x_vec_row.tocoo()
    if row.nnz == 0:
        return pd.DataFrame(), pd.DataFrame()

    idxs = row.col
    vals = row.data

    try:
        feat_names = vectorizer.get_feature_names_out()
        terms = np.array([str(feat_names[i]) for i in idxs], dtype=object)
    except Exception:
        terms = np.array([str(i) for i in idxs], dtype=object)

    logp_pred = feat_log_prob[pred_id, idxs]
    contrib_pred = vals * logp_pred

    logp_alt = feat_log_prob[alt_id, idxs]
    delta = vals * (logp_pred - logp_alt)

    # top by contrib_pred (descending: less negative is "bigger", but log probs are negative)
    # We want tokens with largest contrib_pred (closest to 0).
    order_pred = np.argsort(contrib_pred)[::-1][: int(top_k)]
    df_pred = pd.DataFrame(
        {
            "term": terms[order_pred],
            "tfidf": vals[order_pred].astype(float),
            "logP(term|pred)": logp_pred[order_pred].astype(float),
            "contribution_pred": contrib_pred[order_pred].astype(float),
        }
    )

    # top by delta (descending)
    order_delta = np.argsort(delta)[::-1][: int(top_k)]
    df_delta = pd.DataFrame(
        {
            "term": terms[order_delta],
            "tfidf": vals[order_delta].astype(float),
            "logP(term|pred)": logp_pred[order_delta].astype(float),
            "logP(term|alt)": logp_alt[order_delta].astype(float),
            "delta_pred_minus_alt": delta[order_delta].astype(float),
        }
    )

    return df_pred, df_delta


# =========================
# Load animations
# =========================
LOTTIE_LOADING = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_usmfx6bp.json")
LOTTIE_SUCCESS = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_jbrw3hcz.json")


# =========================
# Cached loaders
# =========================
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_info(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# =========================
# Header
# =========================
st.title("UIT-ViHSD Demo — TF-IDF + Multinomial Naive Bayes (Step-by-step)")
st.caption("Mục tiêu: phân tích vì sao mô hình dự đoán đúng/sai bằng cách đi qua từng bước TF-IDF → NB likelihood → posterior.")

with st.expander("Label definitions (paper-based)", expanded=False):
    st.markdown(
        """- **0 — CLEAN**: Không có công kích/quấy rối.  
- **1 — OFFENSIVE**: Có từ thô tục/công kích nhưng không nhắm trực tiếp 1 cá nhân/nhóm cụ thể.  
- **2 — HATE**: Công kích nhắm trực tiếp cá nhân/nhóm (theo đặc điểm, tôn giáo, quốc tịch, ...)."""
    )

# show info.json
info = load_info(INFO_PATH)
with st.expander("Model config (from final_best_mnb_tfidf_info.json)", expanded=False):
    if info is None:
        st.info("Không đọc được file info JSON.")
    else:
        st.json(info)


# =========================
# Load model + extract parts
# =========================
try:
    loaded = load_model(MODEL_PATH)
except Exception as e:
    st.error(
        "Không thể load model. Thường do mismatch phiên bản numpy/scikit-learn.\n\n"
        f"Error: {type(e).__name__}: {e}"
    )
    st.stop()

model_obj, vectorizer, clf = extract_vectorizer_and_nb(loaded)

if vectorizer is None or clf is None:
    st.error("Không detect được vectorizer hoặc classifier trong model. Hãy kiểm tra object đã lưu (pipeline/dict).")
    st.stop()

# sanity check: must have NB params
if not hasattr(clf, "feature_log_prob_") or not hasattr(clf, "class_log_prior_"):
    st.error("Classifier không phải MultinomialNB (thiếu feature_log_prob_ / class_log_prior_).")
    st.stop()


# =========================
# Session state to prevent losing results on UI rerun
# =========================
if "analysis" not in st.session_state:
    st.session_state["analysis"] = None
if "last_text" not in st.session_state:
    st.session_state["last_text"] = ""


# =========================
# Layout
# =========================
left_col, right_col = st.columns(2, gap="large")

# -------- Left: Input & controls --------
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Input & Output")

    text = st.text_area(
        "Input text",
        value=st.session_state["last_text"],
        height=220,
        placeholder="Nhập bình luận tiếng Việt ở đây...",
    )

    c1, c2 = st.columns(2)
    with c1:
        topk_tfidf = st.number_input("Top-K TF-IDF terms", min_value=5, max_value=60, value=15, step=1)
    with c2:
        topk_terms_class = st.number_input("Top-K terms per class (P(t|c))", min_value=10, max_value=80, value=25, step=5)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    slow_mode = st.toggle("Step-by-step (tua chậm)", value=True)
    speed = st.slider("Tốc độ tua (giây / bước)", min_value=1, max_value=2, value=1, step=1)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    # Ground-truth optional
    true_label_opt = st.selectbox(
        "Nhãn thật (để phân tích đúng/sai) — chọn nếu bạn biết ground-truth",
        options=["(không chọn)"] + [f"{i} — {LABEL_ID_TO_NAME[i]}" for i in [0, 1, 2]],
        index=0,
    )

    run = st.button("Predict & Explain", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# Run analysis (store in session_state to avoid reset)
# =========================
if run:
    if not text or not text.strip():
        st.warning("Please enter a non-empty text.")
    else:
        st.session_state["last_text"] = text

        if LOTTIE_LOADING is not None:
            with left_col:
                st_lottie(LOTTIE_LOADING, height=110, key="loading_anim_nb")

        # vectorize
        try:
            X_vec = vectorizer.transform([text])
        except Exception as e:
            st.error(f"TF-IDF transform lỗi: {type(e).__name__}: {e}")
            st.stop()

        # predict by pipeline if possible; else by clf on X_vec
        try:
            if isinstance(model_obj, Pipeline):
                pred_id = int(model_obj.predict([text])[0])
            else:
                pred_id = int(clf.predict(X_vec)[0])
        except Exception:
            pred_id = int(clf.predict(X_vec)[0])

        pred_name = LABEL_ID_TO_NAME.get(pred_id, str(pred_id))

        # NB posterior details
        details = nb_explain_log_posterior(vectorizer, clf, X_vec, top_k=int(topk_tfidf))
        if details is None:
            st.error("Không tính được giải thích NB (thiếu thuộc tính hoặc lỗi dữ liệu).")
            st.stop()

        log_prior = np.asarray(details["log_prior"], dtype=float).reshape(-1)
        log_like = np.asarray(details["log_like"], dtype=float).reshape(-1)
        log_post = np.asarray(details["log_post"], dtype=float).reshape(-1)

        # confidence-like by softmax over log_post (not calibrated)
        conf_like = softmax(log_post)

        df_scores = pd.DataFrame(
            {
                "label_id": [0, 1, 2],
                "label_name": [LABEL_ID_TO_NAME[i] for i in [0, 1, 2]],
                "log_prior": log_prior[:3],
                "log_likelihood": log_like[:3],
                "log_posterior_unnorm": log_post[:3],
                "softmax(log_posterior)": conf_like[:3],
            }
        ).sort_values("log_posterior_unnorm", ascending=False)

        # choose alt class (runner-up)
        top_order = df_scores.sort_values("log_posterior_unnorm", ascending=False)["label_id"].astype(int).tolist()
        alt_id = int(top_order[1]) if len(top_order) > 1 else int((pred_id + 1) % 3)

        # TF-IDF top terms
        tfidf_terms = tfidf_top_terms(vectorizer, X_vec, top_k=int(topk_tfidf))
        df_tfidf = pd.DataFrame(tfidf_terms, columns=["term", "tfidf"])

        # Token evidence tables
        df_pred, df_delta_pred_alt = build_token_evidence_tables(
            vectorizer=vectorizer,
            clf=clf,
            x_vec_row=X_vec,
            pred_id=int(pred_id),
            alt_id=int(alt_id),
            top_k=max(10, int(topk_tfidf)),
        )

        # If ground truth is selected, add "why wrong/right" against true label
        true_id = None
        if true_label_opt != "(không chọn)":
            true_id = int(true_label_opt.split("—")[0].strip())

        df_delta_pred_true = pd.DataFrame()
        if true_id is not None and true_id in [0, 1, 2]:
            df_pred_vs_true, df_delta_pred_true = build_token_evidence_tables(
                vectorizer=vectorizer,
                clf=clf,
                x_vec_row=X_vec,
                pred_id=int(pred_id),
                alt_id=int(true_id),
                top_k=max(10, int(topk_tfidf)),
            )
            # reuse df_pred_vs_true if desired (not necessary)

        # Per-class top terms distribution (global learned distribution)
        df_class_terms = {}
        for cid in [0, 1, 2]:
            df_class_terms[cid] = nb_class_term_table(vectorizer, clf, class_index=cid, top_k=int(topk_terms_class))

        st.session_state["analysis"] = {
            "text": text,
            "X_vec": X_vec,  # keep sparse
            "pred_id": pred_id,
            "pred_name": pred_name,
            "alt_id": alt_id,
            "df_scores": df_scores,
            "df_tfidf": df_tfidf,
            "df_pred": df_pred,
            "df_delta_pred_alt": df_delta_pred_alt,
            "true_id": true_id,
            "df_delta_pred_true": df_delta_pred_true,
            "df_class_terms": df_class_terms,
        }


# =========================
# Render Left Output (from session_state)
# =========================
with left_col:
    analysis = st.session_state.get("analysis", None)
    if analysis is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Prediction")

        st.markdown(label_badge_html(int(analysis["pred_id"]), analysis["pred_name"]), unsafe_allow_html=True)

        if LOTTIE_SUCCESS is not None:
            st_lottie(LOTTIE_SUCCESS, height=95, key="success_anim_nb")

        st.markdown(
            '<div class="muted">Ghi chú: softmax(log posterior) chỉ là “score trực quan”, không phải calibrated probability.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)


# =========================
# Right: Explanation tabs
# =========================
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Explainability — TF-IDF → MultinomialNB (step-by-step)")

    analysis = st.session_state.get("analysis", None)
    if analysis is None:
        st.info("Nhập text và nhấn Predict & Explain để xem phân tích chi tiết.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        tab1, tab2, tab3, tab4 = st.tabs(
            ["1) Overview scores", "2) Step-by-step animation", "3) Learned distributions P(t|c)", "4) Token evidence (đúng/sai)"]
        )

        # -------- Tab 1: Overview
        with tab1:
            st.markdown("### Tổng hợp điểm theo từng lớp")

            df_scores = analysis["df_scores"]
            st.dataframe(df_scores, use_container_width=True, hide_index=True)

            df_plot = df_scores.sort_values("label_id")
            fig = px.bar(
                df_plot,
                x="label_name",
                y="softmax(log_posterior)",
                text="softmax(log_posterior)",
                range_y=[0, max(1.0, float(df_plot["softmax(log_posterior)"].max()) * 1.15)],
            )
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
            st.plotly_chart(fig, use_container_width=True, key="plot_nb_conf")

            st.markdown(
                """
**Ý nghĩa các cột**  
- $\\log P(c)$: prior theo lớp  
- $\\log P(x\\mid c)$: likelihood (cộng dồn theo token)  
- $\\log P(c) + \\log P(x\\mid c)$: posterior chưa chuẩn hoá  
- softmax: để nhìn trực quan “lớp nào trội hơn”
                """
            )

            # -------- Tab 2: Step-by-step animation (MANUAL NEXT)
            with tab2:
                st.markdown("### Step-by-step: bấm nút để qua từng bước (không tự động)")

                # init step state
                if "nb_step" not in st.session_state:
                    st.session_state["nb_step"] = 1

                # convenience
                step_now = int(st.session_state["nb_step"])
                total_steps = 5

                # controls row
                cbtn1, cbtn2, cbtn3, cbtn4 = st.columns([1.2, 1.2, 2.2, 2.2])
                with cbtn1:
                    if st.button("Reset", use_container_width=True, key="btn_nb_reset"):
                        st.session_state["nb_step"] = 1
                        st.rerun()
                with cbtn2:
                    if st.button("Next step", type="primary", use_container_width=True, key="btn_nb_next"):
                        st.session_state["nb_step"] = min(total_steps, int(st.session_state["nb_step"]) + 1)
                        st.rerun()
                with cbtn3:
                    if st.button("Prev", use_container_width=True, key="btn_nb_prev"):
                        st.session_state["nb_step"] = max(1, int(st.session_state["nb_step"]) - 1)
                        st.rerun()
                with cbtn4:
                    auto_play = st.toggle("Auto-play (tuỳ chọn)", value=False, key="nb_autoplay_toggle")

                # Optional autoplay (only if you turn it on)
                # It advances ONE step per rerun with a delay, but you can keep it OFF for presentation.
                if auto_play and step_now < total_steps:
                    time.sleep(float(speed))
                    st.session_state["nb_step"] = step_now + 1
                    st.rerun()

                st.markdown(f"**Step hiện tại:** {step_now}/{total_steps}")
                st.progress(int(step_now / total_steps * 100))

                # pull artifacts
                text0 = analysis["text"]
                X_vec = analysis["X_vec"]
                df_tfidf = analysis["df_tfidf"]
                df_scores = analysis["df_scores"]
                pred_name = analysis["pred_name"]
                pred_id = int(analysis["pred_id"])
                alt_id = int(analysis["alt_id"])
                alt_name = LABEL_ID_TO_NAME.get(alt_id, str(alt_id))
                df_pred = analysis["df_pred"].copy()

                # -------- Step 1 --------
                if step_now >= 1:
                    st.markdown("#### Bước 1/5: Nhận input text")
                    st.markdown(f"**Input:** `{text0}`")

                # -------- Step 2 --------
                if step_now >= 2:
                    st.markdown("#### Bước 2/5: TF-IDF biến text thành vector $x$")
                    try:
                        nnz = int(X_vec.nnz)
                        n_features = int(X_vec.shape[1])
                        l2 = float(np.sqrt(X_vec.multiply(X_vec).sum()))
                        sparsity = 1.0 - (nnz / max(1, n_features))
                        st.markdown(
                            f"- $n\\_features = {n_features}$\n"
                            f"- $nnz = {nnz}$\n"
                            f"- $sparsity = {sparsity:.6f}$\n"
                            f"- $\\lVert x \\rVert_2 = {l2:.6f}$\n"
                        )
                    except Exception:
                        st.info("Không tính được thống kê TF-IDF vector.")

                    st.markdown("Top TF-IDF terms trong input:")
                    st.dataframe(df_tfidf, use_container_width=True, hide_index=True)

                # -------- Step 3 --------
                if step_now >= 3:
                    st.markdown("#### Bước 3/5: Naive Bayes đã học gì từ train?")
                    st.markdown(
                        """
    MultinomialNB học 2 nhóm tham số:

    - Prior theo lớp: $\\log P(c)$  
    - Likelihood theo token: $\\log P(t\\mid c)$  

    Các giá trị này được học từ thống kê tần suất token theo lớp (có smoothing).
                        """
                    )
                    st.info("Gợi ý trình bày: mở Tab “Learned distributions P(t|c)” để xem bảng top token theo từng lớp.")

                # -------- Step 4 --------
                if step_now >= 4:
                    st.markdown("#### Bước 4/5: Tính likelihood và posterior cho 3 lớp")
                    st.markdown(
                        """
    Với input hiện tại, NB tính:

    $$
    \\log P(x\\mid c) = \\sum\\_{t \\in input} x_t \\cdot \\log P(t\\mid c)
    $$

    Sau đó:

    $$
    \\log P(c\\mid x) \\propto \\log P(c) + \\log P(x\\mid c)
    $$
                        """
                    )
                    st.dataframe(df_scores, use_container_width=True, hide_index=True)

                    st.markdown(
                        f"**Kết luận hiện tại:** `{pred_id}` — **{pred_name}** (so với runner-up: **{alt_name}**) "
                    )

                # -------- Step 5 --------
                if step_now >= 5:
                    st.markdown("#### Bước 5/5: Token evidence (tua dần theo nút bấm)")

                    st.markdown(
                        f"""
    Ta hiển thị token đóng góp mạnh cho lớp dự đoán **{pred_name}**.

    Định nghĩa:

    $$
    contribution\\_t = x_t \\cdot \\log P(t\\mid \\hat{{y}})
    $$

    Và để thấy vì sao **{pred_name}** thắng **{alt_name}**:

    $$
    delta\\_t = x_t \\cdot (\\log P(t\\mid \\hat{{y}}) - \\log P(t\\mid alt))
    $$
                        """
                    )

                    if df_pred.empty:
                        st.info("Không có token evidence (vector rỗng hoặc không match vocab).")
                    else:
                        # --- Manual reveal inside step 5 ---
                        if "nb_reveal_i" not in st.session_state:
                            st.session_state["nb_reveal_i"] = 0

                        kmax = int(min(25, len(df_pred)))

                        r1, r2, r3, r4 = st.columns([1.2, 1.2, 1.6, 2.0])
                        with r1:
                            if st.button("Reveal +1", type="primary", use_container_width=True, key="btn_reveal1"):
                                st.session_state["nb_reveal_i"] = min(kmax, int(st.session_state["nb_reveal_i"]) + 1)
                                st.rerun()
                        with r2:
                            if st.button("Reveal +5", use_container_width=True, key="btn_reveal5"):
                                st.session_state["nb_reveal_i"] = min(kmax, int(st.session_state["nb_reveal_i"]) + 5)
                                st.rerun()
                        with r3:
                            if st.button("Reset reveal", use_container_width=True, key="btn_reveal_reset"):
                                st.session_state["nb_reveal_i"] = 0
                                st.rerun()
                        with r4:
                            # optional auto for reveal only
                            auto_reveal = st.toggle("Auto reveal (tuỳ chọn)", value=False, key="toggle_auto_reveal")

                        # optional auto reveal
                        if auto_reveal and int(st.session_state["nb_reveal_i"]) < kmax:
                            time.sleep(float(speed) / 10.0)  # vẫn giữ logic “nhanh gấp 10”
                            st.session_state["nb_reveal_i"] = min(kmax, int(st.session_state["nb_reveal_i"]) + 1)
                            st.rerun()

                        shown = int(st.session_state["nb_reveal_i"])
                        st.markdown(f"**Đã reveal:** {shown}/{kmax} token")

                        if shown > 0:
                            df_show = df_pred.head(shown).copy()
                            cum = float(df_show["contribution_pred"].sum())
                            st.dataframe(df_show, use_container_width=True, hide_index=True)
                            st.markdown(f"**Cumulative contribution (top {shown} terms):** {cum:.6f}")
                        else:
                            st.info("Bấm Reveal để hiện dần token.")

                # Important: when user goes back to step < 5, keep reveal but do not show it
                # (no reset unless user presses Reset reveal or Reset step).

        # -------- Tab 3: Learned distributions
        with tab3:
            st.markdown("### Bảng phân phối đã học: $P(t\\mid c)$ (hiển thị top terms)")

            st.markdown(
                """
Ở MultinomialNB, mỗi lớp có một phân phối theo token.  
App hiển thị top token có $\\log P(t\\mid c)$ cao nhất (tương đương $P(t\\mid c)$ cao nhất).  
Đây là “tín hiệu kiểu lớp” mà mô hình học được từ dữ liệu train.
                """
            )

            df_class_terms = analysis["df_class_terms"]
            cA, cB, cC = st.columns(3)

            with cA:
                st.markdown("#### CLEAN")
                st.dataframe(df_class_terms[0], use_container_width=True, hide_index=True)
            with cB:
                st.markdown("#### OFFENSIVE")
                st.dataframe(df_class_terms[1], use_container_width=True, hide_index=True)
            with cC:
                st.markdown("#### HATE")
                st.dataframe(df_class_terms[2], use_container_width=True, hide_index=True)

        # -------- Tab 4: Token evidence (right/wrong)
        with tab4:
            pred_id = int(analysis["pred_id"])
            pred_name = analysis["pred_name"]
            alt_id = int(analysis["alt_id"])
            alt_name = LABEL_ID_TO_NAME.get(alt_id, str(alt_id))

            st.markdown("### Token evidence để giải thích vì sao đúng/sai")

            st.markdown(
                f"""
**Ý tưởng nói với giáo sư (gọn, đúng bản chất):**  
- TF-IDF tạo vector $x$ (token nào nổi bật thì $x_t$ lớn).  
- Naive Bayes có $\\log P(t\\mid c)$ cho từng lớp.  
- Với input này, mô hình cộng dồn $x_t\\cdot\\log P(t\\mid c)$ để ra $\\log P(x\\mid c)$.  
- Lớp nào có $\\log P(c)+\\log P(x\\mid c)$ lớn nhất thì là nhãn dự đoán.
                """
            )

            st.markdown("#### A) Token đóng góp mạnh cho lớp dự đoán")
            df_pred = analysis["df_pred"]
            if df_pred.empty:
                st.info("Không có token evidence (vector rỗng hoặc không match vocab).")
            else:
                st.dataframe(df_pred, use_container_width=True, hide_index=True)

            st.markdown(f"#### B) Vì sao {pred_name} thắng {alt_name} (delta token-level)")
            df_delta = analysis["df_delta_pred_alt"]
            if df_delta.empty:
                st.info("Không dựng được bảng delta.")
            else:
                st.dataframe(df_delta, use_container_width=True, hide_index=True)

            # Wrong/right vs true
            true_id = analysis.get("true_id", None)
            if true_id is None:
                st.markdown(
                    '<div class="muted">Nếu bạn chọn “Nhãn thật” ở cột trái, app sẽ phân tích cụ thể vì sao đúng/sai so với ground-truth.</div>',
                    unsafe_allow_html=True,
                )
            else:
                true_name = LABEL_ID_TO_NAME.get(int(true_id), str(true_id))
                if int(true_id) == int(pred_id):
                    st.success(f"Ground-truth = {true_name}. Mô hình dự đoán ĐÚNG.")
                else:
                    st.error(f"Ground-truth = {true_name}, nhưng mô hình dự đoán = {pred_name}. Đây là lỗi dự đoán.")

                st.markdown(
                    f"""
#### C) Phân tích lỗi so với ground-truth ({true_name})

Ta nhìn $delta\\_t$ giữa lớp dự đoán và lớp đúng:

$$
delta\\_t = x_t \\cdot (\\log P(t\\mid pred) - \\log P(t\\mid true))
$$

- Nếu $delta\\_t$ dương lớn: token khiến mô hình nghiêng về lớp dự đoán hơn lớp đúng  
- Nếu $delta\\_t$ âm: token ủng hộ lớp đúng (nhưng không đủ mạnh để thắng)
                    """
                )

                df_delta_true = analysis.get("df_delta_pred_true", pd.DataFrame())
                if df_delta_true is None or df_delta_true.empty:
                    st.info("Không dựng được bảng delta_pred_minus_true.")
                else:
                    st.dataframe(df_delta_true, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)