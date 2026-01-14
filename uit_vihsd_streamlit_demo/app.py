# app.py
import time
import json
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import requests

from streamlit_lottie import st_lottie
from sklearn.pipeline import Pipeline

# Optional: PyVi tokenizer (word segmentation)
try:
    from pyvi import ViTokenizer
    _HAS_PYVI = True
except Exception:
    _HAS_PYVI = False


# =========================
# Config
# =========================
TRAIN_PATH = ".././UIT-ViHSD-preprocessed/train.csv"  # chỉnh đúng đường dẫn của bạn
st.set_page_config(page_title="UIT-ViHSD — TF-IDF + MultinomialNB Demo", layout="wide")

MODEL_PATH = "final_best_mnb_tfidf.joblib"
INFO_PATH = "final_best_mnb_tfidf_info.json"

LABEL_ID_TO_NAME = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}

# Stopword file (optional). Nếu không có file, app tự bỏ qua bước stopword.
STOPWORD_FILE = "vietnamese-stopwords.txt"


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
def logsumexp(a: np.ndarray, axis=None, keepdims=False):
    a = np.asarray(a, dtype=float)
    a_max = np.max(a, axis=axis, keepdims=True)
    # tránh -inf
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    s = np.sum(np.exp(a - a_max), axis=axis, keepdims=True)
    out = a_max + np.log(np.maximum(s, 1e-300))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def nb_class_logodds_table(vectorizer, clf, class_index: int, top_k=30):
    """
    Bảng token đặc trưng theo LOG-ODDS cho class_index:
      log_odds(t,c) = log P(t|c) - log P(t|not c)

    P(t|not c) được trộn theo prior của các lớp còn lại:
      P(t|not c) = sum_{j!=c} w_j * P(t|j),
      w_j = P(j) / sum_{k!=c} P(k)
    """
    if vectorizer is None or clf is None:
        return pd.DataFrame(columns=["term", "logP_t_given_c", "logP_t_given_notc", "log_odds"])

    if (not hasattr(clf, "feature_log_prob_")) or (not hasattr(clf, "class_log_prior_")):
        return pd.DataFrame(columns=["term", "logP_t_given_c", "logP_t_given_notc", "log_odds"])

    feat_names = vectorizer.get_feature_names_out()
    logP_tc = np.asarray(clf.feature_log_prob_, dtype=float)  # (C, V)
    log_prior = np.asarray(clf.class_log_prior_, dtype=float)  # (C,)

    C, V = logP_tc.shape
    c = int(class_index)
    others = [j for j in range(C) if j != c]
    if len(others) == 0:
        return pd.DataFrame(columns=["term", "logP_t_given_c", "logP_t_given_notc", "log_odds"])

    # log weights for mixture among others: log w_j = log P(j) - logsumexp(log P(others))
    logZ = logsumexp(log_prior[others])
    log_w = log_prior[others] - logZ  # (C-1,)

    # log P(t|not c) = logsumexp_j (log w_j + log P(t|j))
    # build matrix (C-1, V)
    mat = log_w.reshape(-1, 1) + logP_tc[others, :]
    logP_t_notc = logsumexp(mat, axis=0)  # (V,)

    logP_t_c = logP_tc[c, :]  # (V,)
    log_odds = logP_t_c - logP_t_notc  # (V,)

    order = np.argsort(log_odds)[::-1][: int(top_k)]
    df = pd.DataFrame({
        "term": [str(feat_names[i]) for i in order],
        "logP_t_given_c": logP_t_c[order],
        "logP_t_given_notc": logP_t_notc[order],
        "log_odds": log_odds[order],
    })
    return df


@st.cache_data(show_spinner=False)
def load_train_csv_for_stats(path: str):
    df = pd.read_csv(path)
    if "free_text" not in df.columns or "label_id" not in df.columns:
        return None
    df = df.dropna(subset=["free_text", "label_id"]).reset_index(drop=True)
    df["free_text"] = df["free_text"].fillna("").astype(str)
    df["label_id"] = df["label_id"].astype(int)
    return df


def keyword_distribution_stats(train_df: pd.DataFrame, keywords, labels=(0, 1, 2), top_k=None):
    """
    Đếm số CÂU chứa keyword theo từng label.
    keywords: list[str]
    top_k: nếu muốn cắt bớt số token để bảng gọn
    """
    if train_df is None or len(train_df) == 0:
        return pd.DataFrame()

    if not keywords:
        return pd.DataFrame()

    # optional cut
    kws = [str(k) for k in keywords if k and str(k).strip()]
    if top_k is not None:
        kws = kws[: int(top_k)]

    stats = []
    for w in kws:
        row = {"Token": w}
        total = 0
        for lid in labels:
            cnt = train_df[(train_df["label_id"] == int(lid)) & (train_df["free_text"].str.contains(w, regex=False))].shape[0]
            row[f"In {LABEL_ID_TO_NAME[int(lid)]} ({int(lid)})"] = int(cnt)
            total += int(cnt)

        row["Total Count"] = int(total)

        denom = total if total > 0 else 1
        # tỉ lệ thô theo từng lớp (để giải thích “lexical bias”)
        for lid in labels:
            cnt = row[f"In {LABEL_ID_TO_NAME[int(lid)]} ({int(lid)})"]
            row[f"% {LABEL_ID_TO_NAME[int(lid)]}"] = round(cnt / denom * 100.0, 1)

        stats.append(row)

    df_stats = pd.DataFrame(stats)
    # ưu tiên token xuất hiện nhiều
    df_stats = df_stats.sort_values("Total Count", ascending=False).reset_index(drop=True)
    return df_stats


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


# =========================
# Preprocessing (HIDDEN by default)
# =========================
_URL_RE = re.compile(r"(http\S+|www\S+|https\S+)", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+", flags=re.UNICODE)

@st.cache_resource(show_spinner=False)
def load_stopwords(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            sw = set([line.strip() for line in f if line.strip()])
        return sw
    except Exception:
        return None


def preprocess_text(text: str) -> str:
    """
    Pipeline (giống tinh thần notebook):
    1) normalize str + strip
    2) lowercase
    3) remove URL
    4) remove @mention
    5) remove hashtag symbol '#'
    6) collapse spaces
    7) word segmentation (PyVi) nếu có
    8) remove stopwords nếu có file
    """
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""

    # lowercase
    s = s.lower()

    # remove url
    s = _URL_RE.sub(" ", s)

    # remove mention
    s = _MENTION_RE.sub(" ", s)

    # remove hashtag symbol only (giữ lại token)
    s = s.replace("#", " ")

    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip()

    # word segmentation
    if _HAS_PYVI:
        try:
            s = ViTokenizer.tokenize(s)
        except Exception:
            pass

    # remove stopwords (optional)
    sw = load_stopwords(STOPWORD_FILE)
    if sw is not None:
        toks = s.split()
        toks = [t for t in toks if t not in sw]
        s = " ".join(toks).strip()

    return s


# =========================
# Model extraction
# =========================
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
        for k in ["pipeline", "model", "estimator", "clf"]:
            if k in loaded:
                model = loaded[k]
                break
        for k in ["vectorizer", "tfidf", "tfidf_vectorizer"]:
            if k in loaded:
                vectorizer = loaded[k]
                break
        for k in ["clf", "classifier", "mnb", "nb"]:
            if k in loaded and hasattr(loaded[k], "predict"):
                clf = loaded[k]
                break

    if isinstance(model, Pipeline):
        steps = dict(model.named_steps)
        for k in ["tfidf", "vectorizer", "vect"]:
            if k in steps:
                vectorizer = steps[k]
                break
        for k in ["clf", "classifier", "mnb", "nb"]:
            if k in steps:
                clf = steps[k]
                break
        if clf is None:
            clf = model.steps[-1][1]
        return model, vectorizer, clf

    if clf is None and hasattr(model, "predict"):
        clf = model

    return model, vectorizer, clf


def tfidf_top_terms(vectorizer, x_vec_row, top_k=15):
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
        return [(str(feat_names[idxs[i]]), float(vals[i])) for i in order]
    except Exception:
        return []


def nb_class_term_table(vectorizer, clf, class_index: int, top_k=25):
    if vectorizer is None or clf is None:
        return pd.DataFrame(columns=["term", "log_P_t_given_c", "P_t_given_c"])
    if not hasattr(clf, "feature_log_prob_"):
        return pd.DataFrame(columns=["term", "log_P_t_given_c", "P_t_given_c"])
    try:
        feat_names = vectorizer.get_feature_names_out()
        logp = np.asarray(clf.feature_log_prob_)[class_index]
        order = np.argsort(logp)[::-1][: int(top_k)]
        terms = [str(feat_names[i]) for i in order]
        logps = logp[order]
        ps = np.exp(logps)
        return pd.DataFrame({"term": terms, "log_P_t_given_c": logps, "P_t_given_c": ps})
    except Exception:
        return pd.DataFrame(columns=["term", "log_P_t_given_c", "P_t_given_c"])


def nb_explain_log_posterior(vectorizer, clf, x_vec_row):
    if vectorizer is None or clf is None or x_vec_row is None:
        return None
    if not hasattr(clf, "feature_log_prob_") or not hasattr(clf, "class_log_prior_"):
        return None

    feat_log_prob = np.asarray(clf.feature_log_prob_)   # (C, V)
    log_prior = np.asarray(clf.class_log_prior_)        # (C,)

    row = x_vec_row.tocoo()
    idxs = row.col
    vals = row.data

    if row.nnz == 0:
        C = int(log_prior.size)
        log_like = np.zeros((C,), dtype=float)
        log_post = log_prior + log_like
        return {"log_prior": log_prior, "log_like": log_like, "log_post": log_post, "row": row}

    log_like = (feat_log_prob[:, idxs] * vals.reshape(1, -1)).sum(axis=1)
    log_post = log_prior + log_like
    return {"log_prior": log_prior, "log_like": log_like, "log_post": log_post, "row": row}


def build_token_evidence_tables(vectorizer, clf, x_vec_row, pred_id: int, alt_id: int, top_k=20):
    if vectorizer is None or clf is None or x_vec_row is None:
        return pd.DataFrame(), pd.DataFrame()
    if not hasattr(clf, "feature_log_prob_"):
        return pd.DataFrame(), pd.DataFrame()

    feat_log_prob = np.asarray(clf.feature_log_prob_)
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

    order_pred = np.argsort(contrib_pred)[::-1][: int(top_k)]
    df_pred = pd.DataFrame(
        {
            "term": terms[order_pred],
            "tfidf": vals[order_pred].astype(float),
            "logP(term|pred)": logp_pred[order_pred].astype(float),
            "contribution_pred": contrib_pred[order_pred].astype(float),
        }
    )

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

if not hasattr(clf, "feature_log_prob_") or not hasattr(clf, "class_log_prior_"):
    st.error("Classifier không phải MultinomialNB (thiếu feature_log_prob_ / class_log_prior_).")
    st.stop()


# =========================
# Session state (prevent losing results on UI rerun)
# =========================
if "analysis" not in st.session_state:
    st.session_state["analysis"] = None
if "last_text" not in st.session_state:
    st.session_state["last_text"] = ""
if "last_text_proc" not in st.session_state:
    st.session_state["last_text_proc"] = ""


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
    true_label_opt = st.selectbox(
        "Nhãn thật (để phân tích đúng/sai) — chọn nếu bạn biết ground-truth",
        options=["(không chọn)"] + [f"{i} — {LABEL_ID_TO_NAME[i]}" for i in [0, 1, 2]],
        index=0,
    )

    run = st.button("Predict & Explain", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# Run analysis (store in session_state)
# =========================
if run:
    if not text or not text.strip():
        st.warning("Please enter a non-empty text.")
    else:
        st.session_state["last_text"] = text

        # --- PREPROCESS (hidden) ---
        text_proc = preprocess_text(text)
        st.session_state["last_text_proc"] = text_proc

        if LOTTIE_LOADING is not None:
            with left_col:
                st_lottie(LOTTIE_LOADING, height=110, key="loading_anim_nb")

        # vectorize on preprocessed text
        try:
            X_vec = vectorizer.transform([text_proc])
        except Exception as e:
            st.error(f"TF-IDF transform lỗi: {type(e).__name__}: {e}")
            st.stop()

        # predict using preprocessed text
        try:
            if isinstance(model_obj, Pipeline):
                pred_id = int(model_obj.predict([text_proc])[0])
            else:
                pred_id = int(clf.predict(X_vec)[0])
        except Exception:
            pred_id = int(clf.predict(X_vec)[0])

        pred_name = LABEL_ID_TO_NAME.get(pred_id, str(pred_id))

        details = nb_explain_log_posterior(vectorizer, clf, X_vec)
        if details is None:
            st.error("Không tính được giải thích NB (thiếu thuộc tính hoặc lỗi dữ liệu).")
            st.stop()

        log_prior = np.asarray(details["log_prior"], dtype=float).reshape(-1)
        log_like = np.asarray(details["log_like"], dtype=float).reshape(-1)
        log_post = np.asarray(details["log_post"], dtype=float).reshape(-1)

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

        top_order = df_scores.sort_values("log_posterior_unnorm", ascending=False)["label_id"].astype(int).tolist()
        alt_id = int(top_order[1]) if len(top_order) > 1 else int((pred_id + 1) % 3)

        tfidf_terms = tfidf_top_terms(vectorizer, X_vec, top_k=int(topk_tfidf))
        df_tfidf = pd.DataFrame(tfidf_terms, columns=["term", "tfidf"])
        
        train_df_stats = load_train_csv_for_stats(TRAIN_PATH)

        # keywords lấy từ top TF-IDF terms của input (tổng quát)
        keywords_from_input = []
        if df_tfidf is not None and not df_tfidf.empty and "term" in df_tfidf.columns:
            keywords_from_input = df_tfidf["term"].astype(str).tolist()

        df_kw_stats = keyword_distribution_stats(
            train_df=train_df_stats,
            keywords=keywords_from_input,
            labels=(0, 1, 2),
            top_k=10,  # bạn có thể cho user control nếu muốn
        )


        df_pred, df_delta_pred_alt = build_token_evidence_tables(
            vectorizer=vectorizer,
            clf=clf,
            x_vec_row=X_vec,
            pred_id=int(pred_id),
            alt_id=int(alt_id),
            top_k=max(10, int(topk_tfidf)),
        )

        true_id = None
        if true_label_opt != "(không chọn)":
            true_id = int(true_label_opt.split("—")[0].strip())

        df_delta_pred_true = pd.DataFrame()
        if true_id is not None and true_id in [0, 1, 2]:
            _, df_delta_pred_true = build_token_evidence_tables(
                vectorizer=vectorizer,
                clf=clf,
                x_vec_row=X_vec,
                pred_id=int(pred_id),
                alt_id=int(true_id),
                top_k=max(10, int(topk_tfidf)),
            )

        # Per-class log-odds tables (global learned distinctiveness)
        df_class_logodds = {}
        for cid in [0, 1, 2]:
            df_class_logodds[cid] = nb_class_logodds_table(
                vectorizer=vectorizer,
                clf=clf,
                class_index=cid,
                top_k=int(topk_terms_class),
            )


        st.session_state["analysis"] = {
            "text_raw": text,
            "text_proc": text_proc,
            "X_vec": X_vec,
            "pred_id": pred_id,
            "pred_name": pred_name,
            "alt_id": alt_id,
            "df_scores": df_scores,
            "df_tfidf": df_tfidf,
            "df_pred": df_pred,
            "df_delta_pred_alt": df_delta_pred_alt,
            "true_id": true_id,
            "df_delta_pred_true": df_delta_pred_true,
            "df_class_logodds": df_class_logodds,
            "df_kw_stats": df_kw_stats,
        }


# =========================
# Render Left Output
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

        # Debug (ẩn)
        with st.expander("Debug (ẩn): xem preprocessing input", expanded=False):
            st.markdown("**Raw input:**")
            st.code(analysis["text_raw"], language="text")
            st.markdown("**Preprocessed input (đi vào TF-IDF + NB):**")
            st.code(analysis["text_proc"], language="text")
            st.markdown(f"**PyVi available:** {_HAS_PYVI}")
            sw = load_stopwords(STOPWORD_FILE)
            st.markdown(f"**Stopwords loaded:** {sw is not None}")

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
            ["1) Overview scores", "2) Step-by-step (manual)", "3) Learned P(t|c)", "4) Token evidence (đúng/sai)"]
        )

        # -------- Tab 1
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

        # -------- Tab 2 (Manual step)
        with tab2:
            st.markdown("### Step-by-step: bấm nút để qua từng bước (không tự động)")

            if "nb_step" not in st.session_state:
                st.session_state["nb_step"] = 1
            step_now = int(st.session_state["nb_step"])
            total_steps = 5

            cbtn1, cbtn2, cbtn3 = st.columns([1.2, 1.2, 1.2])
            with cbtn1:
                if st.button("Reset", use_container_width=True, key="btn_nb_reset"):
                    st.session_state["nb_step"] = 1
                    st.rerun()
            with cbtn2:
                if st.button("Prev", use_container_width=True, key="btn_nb_prev"):
                    st.session_state["nb_step"] = max(1, step_now - 1)
                    st.rerun()
            with cbtn3:
                if st.button("Next step", type="primary", use_container_width=True, key="btn_nb_next"):
                    st.session_state["nb_step"] = min(total_steps, step_now + 1)
                    st.rerun()

            st.markdown(f"**Step hiện tại:** {step_now}/{total_steps}")
            st.progress(int(step_now / total_steps * 100))

            text_raw = analysis["text_raw"]
            text_proc = analysis["text_proc"]
            X_vec = analysis["X_vec"]
            df_tfidf = analysis["df_tfidf"]
            df_scores = analysis["df_scores"]
            pred_name = analysis["pred_name"]
            pred_id = int(analysis["pred_id"])
            alt_id = int(analysis["alt_id"])
            alt_name = LABEL_ID_TO_NAME.get(alt_id, str(alt_id))

            if step_now >= 1:
                st.markdown("#### Bước 1/5: Nhận input text")
                st.markdown(f"**Input raw:** `{text_raw}`")
                st.markdown("<div class='muted'>Preprocessing được chạy ngầm trước khi vào TF-IDF.</div>", unsafe_allow_html=True)

            if step_now >= 2:
                st.markdown("#### Bước 2/5: Preprocessing (ẩn) → TF-IDF biến text thành vector $x$")
                st.markdown(f"**Text sau preprocessing:** `{text_proc}`")

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

            if step_now >= 3:
                st.markdown("#### Bước 3/5: Naive Bayes đã học gì?")
                st.markdown(
                    """
MultinomialNB học:
- $\\log P(c)$ (prior theo lớp)
- $\\log P(t\\mid c)$ (likelihood theo token cho từng lớp)

Các giá trị này được học từ thống kê tần suất token theo lớp (có smoothing).
                    """
                )
                st.info("Mở tab “Learned P(t|c)” để xem top token theo từng lớp.")

            if step_now >= 4:
                st.markdown("#### Bước 4/5: Tính likelihood và posterior cho 3 lớp")
                st.markdown(
                    """
Với input hiện tại:
$$
\\log P(x\\mid c) = \\sum\\_{t} x_t \\cdot \\log P(t\\mid c)
$$
Sau đó:
$$
\\log P(c\\mid x) \\propto \\log P(c) + \\log P(x\\mid c)
$$
                    """
                )
                st.dataframe(df_scores, use_container_width=True, hide_index=True)
                st.markdown(f"**Kết luận:** `{pred_id}` — **{pred_name}** (runner-up: **{alt_name}**)")

            if step_now >= 5:
                st.markdown("#### Bước 5/5: Token evidence (hiện dần theo nút bấm)")
                df_pred = analysis["df_pred"].copy()

                st.markdown(
                    f"""
Token đóng góp cho lớp dự đoán:
$$
contribution_t = x_t \\cdot \\log P(t\\mid \\hat{{y}})
$$

Vì sao **{pred_name}** thắng **{alt_name}**:
$$
delta_t = x_t \\cdot (\\log P(t\\mid \\hat{{y}}) - \\log P(t\\mid alt))
$$
                    """
                )

                if df_pred.empty:
                    st.info("Không có token evidence (vector rỗng hoặc không match vocab).")
                else:
                    if "nb_reveal_i" not in st.session_state:
                        st.session_state["nb_reveal_i"] = 0

                    kmax = int(min(25, len(df_pred)))

                    r1, r2, r3 = st.columns([1.2, 1.2, 1.2])
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

                    shown = int(st.session_state["nb_reveal_i"])
                    st.markdown(f"**Đã reveal:** {shown}/{kmax} token")

                    if shown > 0:
                        df_show = df_pred.head(shown).copy()
                        cum = float(df_show["contribution_pred"].sum())
                        st.dataframe(df_show, use_container_width=True, hide_index=True)
                        st.markdown(f"**Cumulative contribution (top {shown} terms):** {cum:.6f}")
                    else:
                        st.info("Bấm Reveal để hiện dần token.")

        # -------- Tab 3
        with tab3:
            st.markdown("### Token đặc trưng theo lớp bằng Log-odds")

            st.markdown(
                """
        Thay vì chỉ xem token có $\\log P(t\\mid c)$ cao, ta xem **độ phân biệt** của token cho lớp $c$ so với phần còn lại:

        $$
        log\\_odds(t,c) = \\log P(t\\mid c) - \\log P(t\\mid \\neg c)
        $$

        - log-odds càng cao: token càng “đặc trưng”, giúp phân biệt mạnh cho lớp đó.
        - $P(t\\mid \\neg c)$ được tính bằng cách trộn các lớp còn lại theo prior để tổng quát.
                """
            )

            df_class_logodds = analysis["df_class_logodds"]
            cA, cB, cC = st.columns(3)

            with cA:
                st.markdown("#### CLEAN")
                st.dataframe(df_class_logodds[0], use_container_width=True, hide_index=True)
            with cB:
                st.markdown("#### OFFENSIVE")
                st.dataframe(df_class_logodds[1], use_container_width=True, hide_index=True)
            with cC:
                st.markdown("#### HATE")
                st.dataframe(df_class_logodds[2], use_container_width=True, hide_index=True)

            st.markdown(
                '<div class="muted">Gợi ý trình bày: đây là “từ khóa phân biệt” chứ không chỉ là “từ hay gặp”.</div>',
                unsafe_allow_html=True,
            )


        # -------- Tab 4
        with tab4:
            pred_id = int(analysis["pred_id"])
            pred_name = analysis["pred_name"]
            alt_id = int(analysis["alt_id"])
            alt_name = LABEL_ID_TO_NAME.get(alt_id, str(alt_id))

            st.markdown("### Token evidence để giải thích đúng/sai")

            st.markdown(
                f"""
- TF-IDF tạo vector $x$ (token nào nổi bật thì $x_t$ lớn).  
- Naive Bayes có $\\log P(t\\mid c)$ cho từng lớp.  
- Mô hình cộng dồn $x_t\\cdot\\log P(t\\mid c)$ để ra $\\log P(x\\mid c)$.  
- Lớp nào có $\\log P(c)+\\log P(x\\mid c)$ lớn nhất thì là nhãn dự đoán.
                """
            )

            st.markdown("#### A) Token đóng góp mạnh cho lớp dự đoán")
            df_pred = analysis["df_pred"]
            if df_pred.empty:
                st.info("Không có token evidence.")
            else:
                st.dataframe(df_pred, use_container_width=True, hide_index=True)

            st.markdown(f"#### B) Vì sao {pred_name} thắng {alt_name} (delta)")
            df_delta = analysis["df_delta_pred_alt"]
            if df_delta.empty:
                st.info("Không dựng được bảng delta.")
            else:
                st.dataframe(df_delta, use_container_width=True, hide_index=True)

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

$$
delta_t = x_t \\cdot (\\log P(t\\mid pred) - \\log P(t\\mid true))
$$

- $delta_t$ dương lớn: token kéo mô hình về lớp dự đoán hơn lớp đúng  
- $delta_t$ âm: token ủng hộ lớp đúng nhưng không đủ mạnh
                    """
                )

                df_delta_true = analysis.get("df_delta_pred_true", pd.DataFrame())
                if df_delta_true is None or df_delta_true.empty:
                    st.info("Không dựng được bảng delta_pred_minus_true.")
                else:
                    st.dataframe(df_delta_true, use_container_width=True, hide_index=True)

            st.markdown("#### D) Thống kê phân bố token của input trong tập train")

            df_kw_stats = analysis.get("df_kw_stats", pd.DataFrame())
            if df_kw_stats is None or df_kw_stats.empty:
                st.info("Không có thống kê token-train (kiểm tra TRAIN_PATH hoặc input không có token match).")
            else:
                st.dataframe(df_kw_stats, use_container_width=True, hide_index=True)

                st.markdown(
                    """
            Cách dùng bảng này để giải thích:
            - Nếu một token xuất hiện rất nhiều ở một lớp trong train, nó có thể tạo “lexical bias”.
            - Khi mô hình đoán sai, thường thấy token “kéo” về lớp sai vì token đó có phân bố nghiêng trong train.
                    """
                )


    st.markdown('</div>', unsafe_allow_html=True)