"""
main_web.py  – 
Giao diện Streamlit: SVM Sentiment · TF-IDF Cosine · SVD Collaborative Filtering

  1. Thư mục bảo mật LuuTruDuLieu/ chứa user_accounts.json.
  2. Đăng ký, Đăng nhập, Đăng xuất người dùng thực sự với mã hóa SHA-256.
  3. Quản lý danh sách phim yêu thích cá nhân hóa (Favorite list) giải quyết Cold-start.
  4. Trực quan gợi ý động " Gợi Ý Dành Riêng Cho Bạn" (Netflix-style) trên Trang Chủ.
  5. Nút "Yêu thích" trực tiếp trên từng card phim.
  6. Bảo lưu kết quả tìm kiếm thông qua Streamlit Session State.
  7. Tab " Hồ Sơ Của Tôi" với phân tích gu phim của bạn (AI Taste Profiling) tự động.
"""

import hashlib
import os
import re
import urllib.parse
import json
import datetime
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# CẤU HÌNH TRANG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CineMatch – Gợi Ý Phim AI",
    page_icon="",
    layout="wide",
)

# ---------------------------------------------------------------------------
# QUẢN LÝ TÀI KHOẢN & LƯU TRỮ DỮ LIỆU
# ---------------------------------------------------------------------------
DATA_DIR = "LuuTruDuLieu"
USER_ACCOUNTS_FILE = os.path.join(DATA_DIR, "user_accounts.json")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

def load_user_accounts():
    if os.path.exists(USER_ACCOUNTS_FILE):
        try:
            with open(USER_ACCOUNTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_user_accounts(accounts):
    with open(USER_ACCOUNTS_FILE, "w", encoding="utf-8") as f:
        json.dump(accounts, f, ensure_ascii=False, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

# Khởi tạo các trạng thái session state cho Streamlit
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.favorites = []
    st.session_state.ratings = {}
    st.session_state.reviews = {}
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "search_query_en" not in st.session_state:
    st.session_state.search_query_en = None
if "search_q_vec" not in st.session_state:
    st.session_state.search_q_vec = None
if "search_is_pers" not in st.session_state:
    st.session_state.search_is_pers = False

# ---------------------------------------------------------------------------
# CSS TOÀN CỤC
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Be Vietnam Pro',sans-serif;}
.stApp{background:#0d1117;}

/* Sidebar */
section[data-testid="stSidebar"]>div:first-child{background:#161b22;border-right:1px solid #30363d;}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] .stMarkdown p{color:#c9d1d9 !important;}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label{color:#c9d1d9 !important;}

/* Global text */
.stMarkdown p,.stMarkdown li,.stMarkdown td,.stMarkdown th,.stMarkdown span,.stCheckbox label p,.stCheckbox label span{color:#c9d1d9 !important;}
h1,h2,h3,h4,h5,h6,.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4,.stMarkdown h5,.stMarkdown h6{color:#e6edf3 !important;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:#161b22;border-radius:10px;padding:4px;gap:4px;border:1px solid #30363d;}
.stTabs [data-baseweb="tab"]{border-radius:8px;color:#8b949e;font-weight:600;padding:8px 20px;}
.stTabs [aria-selected="true"]{background:#e50914 !important;color:#fff !important;}

/* Movie card */
.mcard{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:18px 22px;margin-bottom:12px;}
.mcard:hover{border-color:#e50914;transition:border-color .2s;}
.mcard-title{font-size:1.15rem;font-weight:700;color:#e6edf3;margin-bottom:8px;}
.rank{display:inline-block;background:rgba(229,9,20,0.15);color:#e50914;font-weight:700;
      padding:2px 10px;border-radius:6px;margin-right:8px;font-size:1rem;}
.bar-bg{background:#21262d;border-radius:999px;height:7px;margin:6px 0 10px;}
.bar-fill{height:7px;border-radius:999px;background:linear-gradient(90deg,#e50914,#ff6b35);}

/* Chips */
.chip{display:inline-block;background:#21262d;color:#c9d1d9;font-size:0.75rem;
      padding:3px 10px;border-radius:999px;margin:2px;}
.chip-g{background:rgba(46,160,67,0.15);color:#3fb950;}
.chip-b{background:rgba(88,166,255,0.15);color:#58a6ff;}
.chip-y{background:rgba(210,153,34,0.15);color:#d2a520;}

/* Review box – WCAG AA contrast (#b8c0cc on #0d1117 ≈ 6:1) */
.rbox{background:#0d1117;border-left:3px solid #30363d;border-radius:0 8px 8px 0;
      padding:10px 14px;margin:6px 0;color:#b8c0cc;font-size:0.85rem;line-height:1.55;}
.rbox strong{color:#e6edf3;}
.rbox em{color:#8b949e;font-style:normal;}

/* Hero / Metric boxes */
.hero{background:linear-gradient(135deg,#1a0a0a,#2d0808,#1a0505);border:1px solid #3d0f0f;
      border-radius:14px;padding:28px 36px;margin-bottom:20px;}
.hero h1{color:#fff;font-size:1.9rem;font-weight:700;margin:0 0 6px;}
.hero p{color:#c9d1d9;margin:0;font-size:0.93rem;}
.mbox{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;text-align:center;}
.mbox .val{font-size:1.5rem;font-weight:700;color:#e50914;}
.mbox .lbl{font-size:0.78rem;color:#c9d1d9;margin-top:4px;}

/* Alert banners */
.alert-i{background:rgba(88,166,255,0.08);border:1px solid rgba(88,166,255,0.3);
         border-radius:8px;padding:10px 14px;color:#c9d1d9;font-size:0.87rem;margin:8px 0;}
.alert-g{background:rgba(46,160,67,0.08);border:1px solid rgba(46,160,67,0.3);
         border-radius:8px;padding:10px 14px;color:#c9d1d9;font-size:0.87rem;margin:8px 0;}

/* Keyword / theme tags */
.theme-tag{display:inline-block;background:rgba(210,153,34,0.15);color:#e6c94a;
           border:1px solid rgba(210,153,34,0.35);padding:4px 12px;border-radius:999px;
           font-size:0.8rem;margin:3px;font-weight:500;}
.theme-tag-ex{background:rgba(88,166,255,0.12);color:#79b8ff;
              border:1px solid rgba(88,166,255,0.3);padding:3px 10px;
              border-radius:999px;font-size:0.75rem;margin:2px;display:inline-block;}

/* Quick-search buttons */
div[data-testid="column"] .stButton>button{
  background:#1c2128;border:1px solid #444c56;color:#c9d1d9;
  border-radius:999px;font-size:0.82rem;width:100%;transition:all .15s;}
div[data-testid="column"] .stButton>button:hover{
  background:#2d333b;border-color:#e50914;color:#ff6b6b;}

/* Poster placeholder (HTML-rendered fallback) */
.poster-ph{display:flex;flex-direction:column;align-items:center;justify-content:center;
           border-radius:8px;min-height:155px;padding:8px;box-sizing:border-box;}
.poster-initials{font-size:1.9rem;font-weight:700;color:rgba(255,255,255,0.9);letter-spacing:2px;}
.poster-title{font-size:0.65rem;color:rgba(255,255,255,0.55);text-align:center;
              margin-top:6px;word-break:break-all;max-width:90px;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HẰNG SỐ
# ---------------------------------------------------------------------------

TMDB_API_KEY         = "591e629d8c6a47350b7d9a9e74f1b414" 

if os.path.exists("Experiments/v8_svd_500"):
    OUTPUT_DIR = "Experiments/v8_svd_500"
else:
    OUTPUT_DIR = "Experiments/v7_final_750k_svd300"
MAX_USERS_IN_SIDEBAR = 50    # Giới hạn user hiển thị trong selectbox
SNIPPET_LEN          = 150   # Ký tự tối đa trong snippet review
MMR_LAMBDA           = 0.6   # λ trong MMR: cao → ưu tiên relevance; thấp → ưu tiên diversity
                              # Công thức: score = λ·rel - (1-λ)·max_sim_to_selected

# ---------------------------------------------------------------------------
# TẢI MÔ HÌNH
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Đang tải mô hình huấn luyện...")
def load_models():
    """Tải toàn bộ artifact. Dừng app nếu thiếu file."""
    required_files = [
        "model_sentiment_linearsvc.joblib",
        "vectorizer_tfidf.joblib",
        "content_movie_profiles.joblib",
        "content_tfidf_matrix.joblib",
        "web_display_reviews.joblib",
        "collab_svd_model.joblib",
    ]
    missing = [f for f in required_files if not os.path.exists(os.path.join(OUTPUT_DIR, f))]
    if missing:
        st.error(f"Thiếu file mô hình: {missing}\n→ Chạy `train_SVM_cosine_SVD.py` trước!")
        st.stop()
    return tuple(joblib.load(os.path.join(OUTPUT_DIR, f)) for f in required_files)


svm_model, tfidf_vec, movie_profiles, content_matrix, reviews_db, svd_data = load_models()

# ---------------------------------------------------------------------------
# HELPERS – URL
# ---------------------------------------------------------------------------
def build_youtube_url(title: str) -> str:
    """Tạo URL tìm kiếm trailer trên YouTube."""
    return f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(title + ' official trailer')}"


def build_imdb_url(title: str) -> str:
    """Tạo URL tìm kiếm phim trên IMDb."""
    return f"https://www.imdb.com/find?q={urllib.parse.quote_plus(title)}&s=tt"


def rating_to_stars(rating) -> str:
    """Chuyển điểm 1–10 thành chuỗi sao (thang 5 sao)."""
    try:
        n = int(round(float(rating) / 2))
        return "⭐" * n + "☆" * (5 - n)
    except (ValueError, TypeError):
        return ""


def normalize_01(arr: np.ndarray) -> np.ndarray:
    """Min-Max normalize về [0, 1]. Trả về mảng toàn 1 nếu min == max."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.ones_like(arr)
    return (arr - mn) / (mx - mn)


# ---------------------------------------------------------------------------
# HELPERS – POSTER: iTunes → Wikipedia → HTML Placeholder
# ---------------------------------------------------------------------------
def _make_initials(title: str) -> str:
    """Trích xuất chữ cái đầu của mỗi từ trong tên phim, tối đa 3 ký tự."""
    words = re.sub(r"[^a-zA-Z0-9 ]", "", title).split()
    initials = "".join(w[0].upper() for w in words if w)
    return initials[:3] if initials else ""


def _hash_color(title: str) -> str:
    """
    Tạo màu nền từ hash của tên phim.
    Dùng HSL để đảm bảo màu luôn tối (darkmode-friendly) và đủ bão hoà.
    Mỗi tên phim → một màu duy nhất, nhất quán qua các lần reload.
    """
    h = int(hashlib.md5(title.encode()).hexdigest()[:4], 16)
    hue = h % 360           # Hue: 0-359
    return f"hsl({hue}, 55%, 22%)"  # Saturation 55%, Lightness 22% (tối vừa)


def _render_poster_placeholder(title: str) -> None:
    """Render poster ảo dùng HTML: nền màu hash + chữ viết tắt tên phim."""
    bg_color = _hash_color(title)
    initials  = _make_initials(title)
    short     = title[:18] + ("…" if len(title) > 18 else "")
    st.markdown(
        f'<div class="poster-ph" style="background:{bg_color};border:1px solid rgba(255,255,255,0.08);">'
        f'  <div class="poster-initials">{initials}</div>'
        f'  <div class="poster-title">{short}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_poster_url(movie_title: str, tmdb_api_key: str = "") -> str | None:
    """
    Tìm URL poster phim.
    Nếu có TMDB API key → dùng TMDB.
    Nếu không → Trả về None để render HTML placeholder.
    Cache 24 giờ để tránh gọi API liên tục.
    """
    if tmdb_api_key:
        try:
            clean_title = movie_title
            year = ""
            match = re.search(r'\s*\((\d{4})\)', movie_title)
            if match:
                year = match.group(1)
                clean_title = movie_title[:match.start()].strip()

            params_vi = {"api_key": tmdb_api_key, "query": clean_title, "language": "vi-VN"}
            params_en = {"api_key": tmdb_api_key, "query": clean_title}
            if year:
                params_vi["primary_release_year"] = year
                params_en["primary_release_year"] = year

            r = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params=params_vi,
                timeout=3,
            )
            if r.status_code == 200:
                results = r.json().get("results", [])
                if not results:
                    r = requests.get(
                        "https://api.themoviedb.org/3/search/movie",
                        params=params_en,
                        timeout=3,
                    )
                    if r.status_code == 200:
                        results = r.json().get("results", [])
                if results and results[0].get("poster_path"):
                    return f"https://image.tmdb.org/t/p/w500{results[0]['poster_path']}"
            else:
                print(f"[TMDB API Error] HTTP {r.status_code} for '{movie_title}'. Key: {tmdb_api_key}. Response: {r.text}")
        except Exception as e:
            print(f"[TMDB Connection Error] Fail to connect for '{movie_title}': {e}")

    return None  # → Caller dùng _render_poster_placeholder


def render_poster(title: str, tmdb_api_key: str = "") -> None:
    """
    Render poster phim. Nếu không tìm được URL → render placeholder HTML đẹp.
    """
    poster_url = fetch_poster_url(title, tmdb_api_key)
    if poster_url:
        st.image(poster_url, use_container_width=True)
    else:
        _render_poster_placeholder(title)


# ---------------------------------------------------------------------------
# HELPERS – KEYWORDS & EXPLAINABILITY
# ---------------------------------------------------------------------------
# Từ quá chung – loại khỏi keyword hiển thị
_GENERIC_TERMS = {
    "film", "movie", "story", "scene", "character", "plot", "time",
    "watch", "great", "good", "best", "little", "just", "like", "make",
    "really", "made", "one", "also", "even", "first", "well", "much",
    "think", "many", "people", "way", "year", "show", "series", "part",
    "get", "see", "know", "going", "come", "give", "take", "would",
    "could", "never", "still", "another", "pretty", "quite", "better",
}


def get_enriched_keywords(vec_matrix, vectorizer, n: int = 12) -> list[str]:
    """
    Trích xuất từ khoá phong phú từ vector TF-IDF trung bình.
    """
    feature_names = vectorizer.get_feature_names_out()
    avg_vec = np.asarray(vec_matrix.mean(axis=0)).flatten()

    ngram_idx   = [i for i, f in enumerate(feature_names) if " " in f]
    unigram_idx = [
        i for i, f in enumerate(feature_names)
        if " " not in f and f not in _GENERIC_TERMS
    ]

    top_ngrams   = sorted(ngram_idx,   key=lambda i: avg_vec[i], reverse=True)
    top_unigrams = sorted(unigram_idx, key=lambda i: avg_vec[i], reverse=True)

    n_ng  = int(n * 0.6)
    n_uni = n - n_ng
    selected = top_ngrams[:n_ng] + top_unigrams[:n_uni]
    selected.sort(key=lambda i: avg_vec[i], reverse=True)
    return [feature_names[i] for i in selected]


def get_movie_explain_keywords(
    query_vec,
    movie_idx: int,
    vectorizer,
    n: int = 5,
) -> list[str]:
    """
    Explainability: Trích xuất từ khoá giao thoa giữa truy vấn/gu người dùng
    và hồ sơ của phim cụ thể.
    """
    feature_names = vectorizer.get_feature_names_out()
    movie_vec     = np.asarray(content_matrix[movie_idx].todense()).flatten()

    if hasattr(query_vec, "toarray"):
        q_vec = np.asarray(query_vec.toarray()).flatten()
    else:
        q_vec = np.asarray(query_vec).flatten()

    intersection = np.minimum(q_vec, movie_vec)

    valid_idx = [
        i for i, f in enumerate(feature_names)
        if intersection[i] > 0 and f not in _GENERIC_TERMS
    ]
    top_idx = sorted(valid_idx, key=lambda i: intersection[i], reverse=True)[:n]
    return [feature_names[i] for i in top_idx]


# ---------------------------------------------------------------------------
# HELPERS – SVD & PROFILE PERSONALIZATION
# ---------------------------------------------------------------------------
def _is_personalized(user_key: str) -> bool:
    """Kiểm tra xem user_key có phải là user cá nhân hóa hay không."""
    if user_key.startswith("Tài khoản của tôi"):
        return st.session_state.logged_in and len(st.session_state.favorites) > 0
    return user_key != "Khách (Cold-start)" and user_key in svd_data["user_map"]


def _compute_svd_scores(candidate_movies: pd.Series, user_key: str) -> np.ndarray:
    """
    Tính điểm SVD hoặc điểm tương đồng sở thích cho danh sách phim ứng viên.
    """
    scores = np.zeros(len(candidate_movies))
    if user_key.startswith("Tài khoản của tôi"):
        if st.session_state.logged_in and st.session_state.favorites:
            movie_to_idx = {row["movie"]: idx for idx, row in movie_profiles.iterrows()}
            
            # Tính vector gu trung bình có trọng số dựa trên TF-IDF của phim yêu thích
            weights = []
            valid_idxs = []
            for m in st.session_state.favorites:
                if m in movie_to_idx:
                    idx = movie_to_idx[m]
                    valid_idxs.append(idx)
                    rating = 10
                    if "ratings" in st.session_state:
                        rating = st.session_state.ratings.get(m, 10)
                    w = rating / 10.0
                    weights.append(w)
                    
            if valid_idxs:
                sel_matrix = content_matrix[valid_idxs]
                import scipy.sparse as sp
                W_sparse = sp.diags(weights)
                total_vec = W_sparse.dot(sel_matrix).sum(axis=0)
                
                total_w = sum(weights)
                if total_w > 0:
                    user_vec = total_vec / total_w
                else:
                    user_vec = sel_matrix.mean(axis=0)
                
                if hasattr(user_vec, "toarray"):
                    user_vec = user_vec.toarray()
                elif hasattr(user_vec, "A"):
                    user_vec = user_vec.A
                else:
                    user_vec = np.asarray(user_vec)
                
                user_vec = np.asarray(user_vec).reshape(1, -1)
                for i, title in enumerate(candidate_movies):
                    if title in movie_to_idx:
                        m_idx = movie_to_idx[title]
                        cand_vec = content_matrix[m_idx]
                        if hasattr(cand_vec, "toarray"):
                            cand_vec = cand_vec.toarray()
                        elif hasattr(cand_vec, "A"):
                            cand_vec = cand_vec.A
                        else:
                            cand_vec = np.asarray(cand_vec)
                        cand_vec = np.asarray(cand_vec).reshape(1, -1)
                        scores[i] = cosine_similarity(user_vec, cand_vec)[0][0]
    elif _is_personalized(user_key):
        u_vec = svd_data["user_factors"][svd_data["user_map"][user_key]]
        for i, title in enumerate(candidate_movies):
            if title in svd_data["movie_map"]:
                scores[i] = np.dot(u_vec, svd_data["item_factors"][svd_data["movie_map"][title]])
    return scores


def _blend_scores(
    cos_norm: np.ndarray,
    svd_norm: np.ndarray,
    is_pers: bool,
    w_cos: float = 0.6,
) -> np.ndarray:
    """
    Hybrid Scoring: Kết hợp điểm Content-Based và Cá nhân hóa.
    """
    if not is_pers:
        return cos_norm  # 100% Content-Based khi cold-start
    w_svd = 1.0 - w_cos
    return w_cos * cos_norm + w_svd * svd_norm


# ---------------------------------------------------------------------------
# MMR RERANKING
# ---------------------------------------------------------------------------
def mmr_rerank(
    candidates: pd.DataFrame,
    content_matrix_sub,
    top_n: int,
    lmbda: float = MMR_LAMBDA,
) -> pd.DataFrame:
    """
    Maximal Marginal Relevance (MMR) – tái xếp hạng để tăng diversity.
    """
    if len(candidates) <= top_n:
        return candidates.reset_index(drop=True)

    cands = candidates.reset_index(drop=True)
    scores = cands["final"].values

    selected_indices  = []
    remaining_indices = list(range(len(cands)))

    for _ in range(top_n):
        if not remaining_indices:
            break

        best_idx   = None
        best_score = -float("inf")

        for r in remaining_indices:
            relevance = scores[r]

            if selected_indices:
                r_vec    = content_matrix_sub[[r]]
                s_vecs   = content_matrix_sub[selected_indices]
                sims     = cosine_similarity(r_vec, s_vecs).flatten()
                max_sim  = sims.max()
            else:
                max_sim = 0.0

            mmr_score = lmbda * relevance - (1 - lmbda) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx   = r

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    return cands.iloc[selected_indices].reset_index(drop=True)


# ---------------------------------------------------------------------------
# LOGIC GỢI Ý
# ---------------------------------------------------------------------------
def recommend_by_query(
    query_en: str,
    user_key: str,
    top_n: int,
    w_cos: float = 0.6,
    use_mmr: bool = True,
) -> tuple[pd.DataFrame, bool, object]:
    """
    Gợi ý dựa trên truy vấn văn bản (Content-Based + SVD tuỳ chọn + MMR).
    """
    q_vec   = tfidf_vec.transform([query_en])
    cos_raw = cosine_similarity(q_vec, content_matrix).flatten()

    pool_size = max(150, top_n * 15)
    top_idx   = cos_raw.argsort()[-pool_size:][::-1]
    cands     = movie_profiles.iloc[top_idx].copy().reset_index(drop=True)
    cands["cos_raw"]       = cos_raw[top_idx]
    cands["_matrix_idx"]   = top_idx
    cands = cands[cands["cos_raw"] > 0].reset_index(drop=True)

    if cands.empty:
        return pd.DataFrame(), False, q_vec

    is_pers           = _is_personalized(user_key)
    cands["cos_norm"] = normalize_01(cands["cos_raw"].values)
    svd_raw           = _compute_svd_scores(cands["movie"], user_key)
    cands["svd_norm"] = normalize_01(svd_raw)
    cands["final"]    = _blend_scores(
        cands["cos_norm"].values, cands["svd_norm"].values, is_pers, w_cos
    )

    cands = cands.sort_values("final", ascending=False)

    if use_mmr and len(cands) > top_n:
        matrix_idxs = cands["_matrix_idx"].tolist()
        sub_matrix  = content_matrix[matrix_idxs]
        cands = mmr_rerank(cands, sub_matrix, top_n)
    else:
        cands = cands.head(top_n).reset_index(drop=True)

    return cands.drop(columns=["_matrix_idx"], errors="ignore"), is_pers, q_vec


def recommend_by_movies(
    selected_titles: list[str],
    user_key: str,
    top_n: int,
    exclude_selected: bool = True,
    w_cos: float = 0.6,
    use_mmr: bool = True,
) -> tuple[pd.DataFrame, list[str], object]:
    """
    Gợi ý dựa trên danh sách phim đã thích.
    """
    selected_set = set(selected_titles)
    movie_to_idx = {row["movie"]: idx for idx, row in movie_profiles.iterrows()}
    valid_idxs = []
    weights = []
    for m in selected_titles:
        if m in movie_to_idx:
            idx = movie_to_idx[m]
            valid_idxs.append(idx)
            rating = 10
            if st.session_state.logged_in and "ratings" in st.session_state:
                rating = st.session_state.ratings.get(m, 10)
            w = rating / 10.0
            weights.append(w)
            
    if not valid_idxs:
        return pd.DataFrame(), [], None

    sel_matrix = content_matrix[valid_idxs]
    import scipy.sparse as sp
    W_sparse = sp.diags(weights)
    total_vec = W_sparse.dot(sel_matrix).sum(axis=0)
    
    total_w = sum(weights)
    if total_w > 0:
        avg_vec = total_vec / total_w
    else:
        avg_vec = sel_matrix.mean(axis=0)
        
    if hasattr(avg_vec, "toarray"):
        avg_vec = avg_vec.toarray()
    elif hasattr(avg_vec, "A"):
        avg_vec = avg_vec.A
    else:
        avg_vec = np.asarray(avg_vec)
        
    avg_vec = np.asarray(avg_vec).reshape(1, -1)
    cos_raw = cosine_similarity(avg_vec, content_matrix).flatten()

    cands = movie_profiles.copy().reset_index(drop=True)
    cands["cos_raw"]     = cos_raw
    cands["_matrix_idx"] = np.arange(len(cands))

    if exclude_selected:
        cands = cands[~cands["movie"].isin(selected_set)]

    pool_size = max(200, top_n * 20)
    cands = (
        cands[cands["cos_raw"] > 0]
        .sort_values("cos_raw", ascending=False)
        .head(pool_size)
        .reset_index(drop=True)
    )
    if cands.empty:
        return pd.DataFrame(), [], avg_vec

    is_pers           = _is_personalized(user_key)
    cands["cos_norm"] = normalize_01(cands["cos_raw"].values)
    svd_raw           = _compute_svd_scores(cands["movie"], user_key)
    cands["svd_norm"] = normalize_01(svd_raw)
    cands["final"]    = _blend_scores(
        cands["cos_norm"].values, cands["svd_norm"].values, is_pers, w_cos
    )
    cands = cands.sort_values("final", ascending=False)

    if use_mmr and len(cands) > top_n:
        matrix_idxs = cands["_matrix_idx"].tolist()
        sub_matrix  = content_matrix[matrix_idxs]
        cands = mmr_rerank(cands, sub_matrix, top_n)
    else:
        cands = cands.head(top_n).reset_index(drop=True)

    keywords = get_enriched_keywords(sel_matrix, tfidf_vec, n=12)
    return cands.drop(columns=["_matrix_idx"], errors="ignore"), keywords, avg_vec


# ---------------------------------------------------------------------------
# COMPONENTS – RENDER
# ---------------------------------------------------------------------------
def render_movie_card(
    rank: int,
    title: str,
    final: float,
    cos_n: float,
    svd_n: float,
    explain_keywords: list[str] | None = None,
) -> None:
    """
    Render card phim đẹp mắt với thanh gradient và chips.
    """
    pct = final * 100
    explain_html = ""
    if explain_keywords:
        chips = " ".join(
            f'<span class="theme-tag-ex">{kw}</span>'
            for kw in explain_keywords
        )
        explain_html = f'<div style="margin-top:8px;">{chips}</div>'

    rating_html = ""
    if st.session_state.logged_in and "ratings" in st.session_state and title in st.session_state.ratings:
        my_rating = st.session_state.ratings[title]
        stars = rating_to_stars(my_rating)
        rating_html = f'<div style="margin-top:6px; color:#ffc107; font-size:0.9rem; font-weight:600;">Đánh giá của bạn: {stars} ({my_rating}/10)</div>'

    st.markdown(f"""
    <div class="mcard">
      <div><span class="rank">#{rank}</span>
           <span class="mcard-title">{title}</span></div>
      {rating_html}
      <div style="margin-top:6px;">
        <span class="chip chip-b">Cosine {cos_n:.3f}</span>
        <span class="chip chip-g">SVD {svd_n:.3f}</span>
        <span class="chip chip-y">Tổng hợp {pct:.1f}%</span>
      </div>
      <div class="bar-bg"><div class="bar-fill" style="width:{min(pct,100):.1f}%"></div></div>
      {explain_html}
    </div>""", unsafe_allow_html=True)


def render_reviews_section(movie_title: str) -> None:
    """
    Render review và xử lý review spoiler.
    """
    # 1. Hiển thị nhận xét cá nhân của người dùng hiện tại lên đầu tiên
    has_personal = False
    if st.session_state.logged_in:
        my_rating = st.session_state.ratings.get(movie_title)
        my_review = st.session_state.reviews.get(movie_title)
        
        if my_rating is not None or my_review:
            has_personal = True
            stars_str = rating_to_stars(my_rating) if my_rating is not None else ""
            rating_text = f" ({my_rating}/10)" if my_rating is not None else ""
            review_text = my_review if my_review else "(Không có nội dung nhận xét)"
            
            st.markdown(
                f'<div class="rbox" style="border-left-color: #58a6ff; background-color: rgba(56, 139, 253, 0.1);">'
                f'<strong style="color: #58a6ff;">[ĐÁNH GIÁ CỦA BẠN]</strong> {stars_str} {rating_text}<br>'
                f'{review_text}</div>',
                unsafe_allow_html=True
            )
            st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    sub         = reviews_db[reviews_db["movie"] == movie_title]
    top_reviews = sub.sort_values(["spoiler_tag", "rating"], ascending=[True, False]).head(3)

    if top_reviews.empty:
        if not has_personal:
            st.info("Chưa có đánh giá lưu trong hệ thống.")
        return

    for _, rv in top_reviews.iterrows():
        reviewer  = rv.get("reviewer", "Ẩn danh")
        rating    = rv.get("rating", "?")
        text      = str(rv.get("review", ""))
        is_spoil  = int(rv.get("spoiler_tag", 0)) == 1
        stars_str = rating_to_stars(rating)
        has_more  = len(text) > SNIPPET_LEN
        snippet   = text[:SNIPPET_LEN] + ("..." if has_more else "")

        if is_spoil:
            with st.expander(f"Review của {reviewer} – SPOILER (nhấn để xem)"):
                st.markdown(
                    f'<div class="rbox"><strong>{reviewer}</strong> {stars_str} '
                    f'<em>({rating}/10)</em><br>{text}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f'<div class="rbox"><strong>{reviewer}</strong> {stars_str} '
                f'<em>({rating}/10)</em><br>{snippet}</div>',
                unsafe_allow_html=True,
            )
            if has_more:
                with st.expander("Xem đầy đủ review"):
                    st.markdown(
                        f'<div class="rbox" style="border-left-color:#58a6ff">{text}</div>',
                        unsafe_allow_html=True,
                    )


def render_links_section(movie_title: str, context: str = "fav") -> None:
    """Render nút trailer YouTube, link IMDb và nút Thích phim."""
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.link_button(" Trailer YouTube", build_youtube_url(movie_title), use_container_width=True)
    with c2:
        st.link_button(" Xem trên IMDb",  build_imdb_url(movie_title),   use_container_width=True)
    with c3:
        if st.session_state.logged_in:
            is_fav = movie_title in st.session_state.favorites
            btn_label = " Bỏ thích" if is_fav else " Thích phim"
            btn_type = "secondary" if is_fav else "primary"
            if st.button(btn_label, key=f"fav_{context}_{movie_title}_{btn_label}", type=btn_type, use_container_width=True):
                accounts = load_user_accounts()
                user = st.session_state.username
                if is_fav:
                    st.session_state.favorites.remove(movie_title)
                    st.toast(f"Đã xoá '{movie_title}' khỏi danh sách yêu thích!")
                else:
                    st.session_state.favorites.append(movie_title)
                    st.toast(f"Đã thêm '{movie_title}' vào yêu thích!")
                accounts[user]["favorites"] = st.session_state.favorites
                save_user_accounts(accounts)
                st.rerun()
        else:
            st.button(" Đăng nhập để chọn phim yêu thích cho riêng bạn", key=f"fav_anon_{context}_{movie_title}", disabled=True, use_container_width=True)

    if st.session_state.logged_in:
        st.write("---")
        st.markdown("##### Đánh giá & Nhận xét của bạn")
        
        current_rating = st.session_state.ratings.get(movie_title, 5)
        current_review = st.session_state.reviews.get(movie_title, "")
        
        with st.form(key=f"rating_form_{context}_{movie_title}"):
            rating_val = st.slider(
                "Chọn số điểm (1 - 10 sao)",
                min_value=1,
                max_value=10,
                value=int(current_rating),
                key=f"slider_val_{context}_{movie_title}"
            )
            
            stars_preview = rating_to_stars(rating_val)
            st.markdown(f"**Điểm số chọn:** {stars_preview} ({rating_val}/10)")
            
            review_val = st.text_area(
                "Nhận xét của bạn",
                value=current_review,
                placeholder="Nhập nhận xét ngắn về bộ phim này...",
                key=f"review_val_{context}_{movie_title}"
            )
            
            submit_btn = st.form_submit_button("Lưu Đánh Giá", type="primary")
            if submit_btn:
                accounts = load_user_accounts()
                user = st.session_state.username
                
                st.session_state.ratings[movie_title] = rating_val
                st.session_state.reviews[movie_title] = review_val
                
                if movie_title not in st.session_state.favorites:
                    st.session_state.favorites.append(movie_title)
                    st.toast(f"Đã thêm '{movie_title}' vào danh sách yêu thích!")
                
                accounts[user]["ratings"] = st.session_state.ratings
                accounts[user]["reviews"] = st.session_state.reviews
                accounts[user]["favorites"] = st.session_state.favorites
                
                save_user_accounts(accounts)
                st.toast("Đã lưu đánh giá của bạn thành công!")
                st.rerun()
    else:
        st.write("---")
        st.info("Hãy đăng nhập để đánh giá và nhận xét bộ phim này.")


def render_result_list(
    results: pd.DataFrame,
    query_vec=None,
    tmdb_api_key: str = "",
    context: str = "result",
) -> None:
    """
    Render danh sách kết quả gợi ý.
    """
    _movie_to_matrix_idx = {
        row["movie"]: idx for idx, row in movie_profiles.iterrows()
    }

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        title = row["movie"]

        explain_kws = []
        if query_vec is not None and title in _movie_to_matrix_idx:
            m_idx = _movie_to_matrix_idx[title]
            explain_kws = get_movie_explain_keywords(query_vec, m_idx, tfidf_vec, n=4)

        col_poster, col_main = st.columns([1, 4], gap="medium")

        with col_poster:
            render_poster(title, tmdb_api_key)

        with col_main:
            render_movie_card(
                rank, title, row["final"], row["cos_norm"], row["svd_norm"],
                explain_keywords=explain_kws,
            )
            tab_reviews, tab_links = st.tabs(["Đánh giá cộng đồng", "Trailer & Tương tác"])
            with tab_reviews:
                render_reviews_section(title)
            with tab_links:
                render_links_section(title, context=f"{context}_{rank}")

        st.markdown("---")


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### CineMatch")
    st.caption("Hệ thống Gợi Ý Phim – SVM + TF-IDF + Cosine + SVD")
    st.divider()

    # --- WIDGET QUẢN LÝ TÀI KHOẢN ---
    st.markdown("#### Tài Khoản Người Dùng")
    accounts = load_user_accounts()

    if not st.session_state.logged_in:
        acc_tab_login, acc_tab_register = st.tabs(["Đăng nhập", "Đăng ký"])
        with acc_tab_login:
            with st.form("login_form"):
                login_user = st.text_input("Tên đăng nhập", key="sidebar_login_user")
                login_pass = st.text_input("Mật khẩu", type="password", key="sidebar_login_pass")
                submit_login = st.form_submit_button("Đăng nhập", type="primary", use_container_width=True)
                if submit_login:
                    if not login_user or not login_pass:
                        st.error("Vui lòng điền đủ thông tin!")
                    elif login_user in accounts and accounts[login_user]["password"] == hash_password(login_pass):
                        st.session_state.logged_in = True
                        st.session_state.username = login_user
                        st.session_state.favorites = accounts[login_user].get("favorites", [])
                        st.session_state.ratings = accounts[login_user].get("ratings", {})
                        st.session_state.reviews = accounts[login_user].get("reviews", {})
                        st.toast(f"Đăng nhập thành công! Chào {login_user}")
                        st.rerun()
                    else:
                        st.error("Sai tài khoản hoặc mật khẩu!")
        with acc_tab_register:
            with st.form("register_form"):
                reg_user = st.text_input("Tên đăng ký", key="sidebar_reg_user")
                reg_pass = st.text_input("Mật khẩu mới", type="password", key="sidebar_reg_pass")
                reg_pass_confirm = st.text_input("Xác nhận mật khẩu", type="password", key="sidebar_reg_pass_confirm")
                submit_register = st.form_submit_button("Đăng ký tài khoản", type="primary", use_container_width=True)
                if submit_register:
                    if not reg_user or not reg_pass:
                        st.error("Vui lòng điền đầy đủ thông tin!")
                    elif reg_pass != reg_pass_confirm:
                        st.error("Mật khẩu xác nhận không khớp!")
                    elif reg_user in accounts:
                        st.error("Tên tài khoản đã tồn tại!")
                    else:
                        accounts[reg_user] = {
                            "password": hash_password(reg_pass),
                            "favorites": [],
                            "ratings": {},
                            "reviews": {},
                            "created_at": datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                        }
                        save_user_accounts(accounts)
                        st.success("Đăng ký thành công! Hãy đăng nhập.")
    else:
        st.markdown(f"Chào mừng, **{st.session_state.username}**!")
        st.caption(f"Số phim đã thích: **{len(st.session_state.favorites)}**")
        
        if st.button("Đăng xuất", type="secondary", use_container_width=True, key="btn_logout_sidebar"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.favorites = []
            st.session_state.ratings = {}
            st.session_state.reviews = {}
            st.session_state.search_results = None
            st.session_state.search_query_en = None
            st.toast("Đã đăng xuất!")
            st.rerun()
    st.divider()

    top_n = st.slider("Số phim gợi ý", min_value=3, max_value=10, value=5)

    st.markdown("**Hồ sơ người dùng (Cá nhân hóa SVD)**")
    all_users = list(svd_data["user_map"].keys())
    
    options = ["Khách (Cold-start)"]
    if st.session_state.logged_in:
        options.append(f"Tài khoản của tôi ({st.session_state.username})")
    options += all_users[:MAX_USERS_IN_SIDEBAR]
    
    selected_user = st.selectbox(
        "Người dùng:",
        options=options,
        label_visibility="collapsed",
    )
    st.divider()

    tmdb_api_key = TMDB_API_KEY

    # Hybrid weight slider
    st.markdown("**Tỉ lệ Hybrid Scoring**")
    w_cosine = st.slider(
        "Content (Cosine) ↔ Collab (SVD)",
        min_value=0.3, max_value=1.0, value=0.6, step=0.05,
        help="0.6 = 60% Content-Based + 40% Collaborative. Cold-start luôn dùng 100% Content.",
        format="%.2f",
    )
    w_svd_display = 1.0 - w_cosine
    st.caption(f"Cosine: **{w_cosine:.0%}** | SVD/Gu: **{w_svd_display:.0%}**")
    st.divider()

    use_mmr = st.checkbox(
        "Bật MMR Diversity",
        value=True,
        help="Maximal Marginal Relevance – giúp kết quả đa dạng hơn, tránh gợi ý các phần của cùng 1 series.",
    )
    st.divider()

    with st.expander("Về Hybrid Scoring & MMR"):
        st.markdown(f"""
**Hybrid Scoring:**  
`Score = {w_cosine:.0%} × Cosine + {w_svd_display:.0%} × SVD/Gu`

Cả hai điểm đều được **normalize về [0,1]** trước khi kết hợp.

**MMR (Maximal Marginal Relevance):**  
`MMR = λ × Relevance − (1−λ) × MaxSim(selected)`  
Phạt điểm phim quá giống phim đã chọn → kết quả đa dạng hơn.
        """)

# ---------------------------------------------------------------------------
# TABS CHÍNH
# ---------------------------------------------------------------------------
tab_home, tab_search, tab_exp, tab_profile = st.tabs([
    " Trang Chủ",
    " Gợi Ý Theo Từ Khoá",
    " Trải Nghiệm Xem Phim",
    " Hồ Sơ Của Tôi",
])

# ══════════════════════════════ TAB HOME ════════════════════════════════════
with tab_home:
    st.markdown("""
    <div class="hero">
      <h1>CineMatch – Gợi Ý Phim Thông Minh</h1>
      <p>Kết hợp Phân tích Cảm xúc (SVM), Độ tương đồng nội dung (TF-IDF + Cosine)
         và Lọc cộng tác (SVD) để gợi ý phim phù hợp nhất với bạn.</p>
    </div>""", unsafe_allow_html=True)

    # Đọc SVM Accuracy từ file metrics
    _metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.txt")
    _svm_accuracy = "N/A"
    if os.path.exists(_metrics_path):
        with open(_metrics_path, encoding="utf-8") as _f:
            for _line in _f:
                if "Accuracy" in _line and "%" in _line:
                    try:
                        _svm_accuracy = _line.split("(")[1].split(")")[0].strip()
                    except IndexError:
                        pass
                    break

    ev = svd_data.get("explained_variance", 0)
    metrics = [
        (f"{len(movie_profiles):,}", "Phim trong hệ thống"),
        (f"{len(svd_data['user_map']):,}", "Người dùng"),
        (_svm_accuracy, "SVM Accuracy"),
        (f"{ev*100:.1f}%", "SVD Explained Var."),
    ]
    for col, (val, lbl) in zip(st.columns(4), metrics):
        col.markdown(
            f'<div class="mbox"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- NETFLIX-STYLE PERSONALIZED RECOMMENDATION ON HOME TAB ---
    if st.session_state.logged_in and st.session_state.favorites:
        st.markdown("### 🎬 Gợi Ý Dành Riêng Cho Bạn")
        st.markdown(
            '<div class="alert-g">AI phân tích gu phim của bạn từ danh sách yêu thích và đề xuất các phim tương đồng tốt nhất:</div>',
            unsafe_allow_html=True
        )
        
        results_home, _, avg_vec_home = recommend_by_movies(
            st.session_state.favorites, 
            f"Tài khoản của tôi ({st.session_state.username})", 
            top_n=5,
            exclude_selected=True,
            w_cos=w_cosine,
            use_mmr=use_mmr
        )
        
        if not results_home.empty:
            render_result_list(results_home, query_vec=avg_vec_home, tmdb_api_key=tmdb_api_key, context="home")
        else:
            st.info("Chưa tìm thấy phim tương tự. Hãy thích thêm phim để AI học gu của bạn.")
        st.markdown("---")

    st.markdown("### Kiến Trúc Hybrid Pipeline")
    st.markdown(f"""
| Tầng | Kỹ thuật | Vai trò |
|---|---|---|
| 1 | **Dịch thuật tự động** | Hỗ trợ truy vấn tiếng Việt |
| 2 | **TF-IDF + Cosine Similarity** | Tìm phim có review tương đồng với truy vấn |
| 3 | **Cá nhân hóa nâng cao** | SVD Collaborative (cho user cũ) hoặc Personalized Content-Based (cho tài khoản mới qua danh sách yêu thích) |
| 4 | **MMR Reranking** | Đa dạng hóa kết quả, tránh gợi ý lặp |
| 5 | **Lọc Spoiler** | Ẩn review tiết lộ nội dung phim |

**Hybrid Weight hiện tại:** {w_cosine:.0%} Cosine + {w_svd_display:.0%} SVD/Gu  
→ Điều chỉnh tỉ lệ này trong Sidebar để thay đổi hành vi gợi ý.
    """)

# ══════════════════════════════ TAB SEARCH ══════════════════════════════════
with tab_search:
    st.markdown("#### Nhập mô tả nội dung phim bạn muốn xem")

    with st.form(key="search_form"):
        col_q, col_btn = st.columns([5, 1])
        with col_q:
            query = st.text_input(
                "query", label_visibility="collapsed",
                placeholder="VD: Phim kinh dị siêu nhiên, ma quái trong ngôi nhà bỏ hoang...",
            )
        with col_btn:
            do_search = st.form_submit_button("Tìm phim", type="primary", use_container_width=True)

    QUICK_SEARCHES = [
        "Kinh dị ma quái", "Tình cảm lãng mạn",
        "Viễn tưởng không gian", "Tội phạm trinh thám", "Hoạt hình gia đình",
    ]
    for col, label in zip(st.columns(len(QUICK_SEARCHES)), QUICK_SEARCHES):
        if col.button(label, key=f"quick_{label}"):
            query     = label
            do_search = True

    if do_search and not query:
        st.warning("Vui lòng nhập mô tả phim!")

    elif do_search and query:
        with st.status("Đang phân tích...", expanded=True) as status:
            st.write("Dịch thuật sang tiếng Anh...")
            try:
                from deep_translator import GoogleTranslator
                query_en = GoogleTranslator(source="auto", target="en").translate(query)
            except Exception:
                query_en = query
            st.write(f'→ **"{query_en}"**')
            st.write("Content-Based Filtering...")
            st.write("Cá nhân hóa...")
            st.write("MMR Reranking...")
            
            results, is_pers, q_vec = recommend_by_query(
                query_en, selected_user, top_n,
                w_cos=w_cosine, use_mmr=use_mmr,
            )
            
            # Lưu trữ kết quả tìm kiếm vào session state
            st.session_state.search_results = results
            st.session_state.search_query_en = query_en
            st.session_state.search_q_vec = q_vec
            st.session_state.search_is_pers = is_pers
            status.update(label="Hoàn tất!", state="complete", expanded=False)

    # Hiển thị kết quả tìm kiếm từ session state (đảm bảo không bị mất khi bấm Thích phim)
    if st.session_state.search_results is not None:
        mode_label = (
            f"Cá nhân hóa: **{selected_user}**"
            if st.session_state.search_is_pers else "**Cold-start** (100% Content-Based)"
        )
        mmr_label = "MMR bật" if use_mmr else "MMR tắt"
        st.markdown(
            f'<div class="alert-i">{mode_label} &nbsp;|&nbsp; {mmr_label} &nbsp;|&nbsp; '
            f'Truy vấn: <em>"{st.session_state.search_query_en}"</em></div>',
            unsafe_allow_html=True,
        )

        if st.session_state.search_results.empty:
            st.error("Không tìm thấy phim phù hợp. Thử từ khoá khác.")
        else:
            render_result_list(
                st.session_state.search_results, 
                query_vec=st.session_state.search_q_vec, 
                tmdb_api_key=tmdb_api_key,
                context="search"
            )

# ══════════════════════════════ TAB EXPERIENCE ══════════════════════════════
with tab_exp:
    st.markdown("#### Gợi Ý Dựa Trên Phim Bạn Đã Xem")
    st.markdown("""
    <div class="alert-i">
    Chọn các phim bạn đã xem và thích. Hệ thống sẽ phân tích <strong>điểm chung</strong>
    giữa các phim đó (dựa trên nội dung review TF-IDF) rồi tìm những phim tương tự bạn chưa xem.
    </div>""", unsafe_allow_html=True)

    all_movie_names = sorted(movie_profiles["movie"].tolist())

    col_sel, col_cfg = st.columns([3, 1])
    with col_sel:
        selected_movies = st.multiselect(
            "Chọn phim đã xem (gõ để tìm kiếm):",
            options=all_movie_names,
            placeholder="Tìm và chọn phim...",
            max_selections=10,
        )
    with col_cfg:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        do_exp = st.button("Gợi ý phim tương tự", type="primary", use_container_width=True)

    if selected_movies:
        st.markdown("**Phim đã chọn:**")
        tags_html = " ".join(
            f'<span class="theme-tag">{m}</span>' for m in selected_movies
        )
        st.markdown(tags_html, unsafe_allow_html=True)
        st.markdown("")

    if do_exp and not selected_movies:
        st.warning("Vui lòng chọn ít nhất 1 phim!")

    elif do_exp and selected_movies:
        with st.status("Đang phân tích gu phim của bạn...", expanded=True) as status:
            st.write(f"Phân tích {len(selected_movies)} phim đã chọn...")
            st.write("Tính vector 'gu' trung bình...")
            st.write("MMR Reranking...")
            results_exp, keywords, avg_vec = recommend_by_movies(
                selected_movies, selected_user, top_n,
                w_cos=w_cosine, use_mmr=use_mmr,
            )
            status.update(label="Hoàn tất!", state="complete", expanded=False)

        if results_exp.empty:
            st.error("Không tìm thấy phim phù hợp.")
        else:
            st.markdown("---")
            st.markdown("### Điểm Chung Giữa Các Phim Đã Chọn")
            st.caption("Các từ khoá nổi bật trong review – đây là 'gu' phim của bạn:")
            kw_html = " ".join(f'<span class="theme-tag">{k}</span>' for k in keywords)
            st.markdown(kw_html, unsafe_allow_html=True)

            is_pers_exp = _is_personalized(selected_user)
            mode_label  = (
                f"Cá nhân hóa: **{selected_user}**"
                if is_pers_exp else "Cold-start (100% Content-Based)"
            )
            mmr_label = "MMR bật" if use_mmr else "MMR tắt"
            st.markdown(
                f'<div class="alert-g" style="margin-top:12px;">{mode_label}'
                f' &nbsp;|&nbsp; {mmr_label}'
                f' &nbsp;|&nbsp; Tìm thấy <strong>{len(results_exp)}</strong> phim phù hợp</div>',
                unsafe_allow_html=True,
            )

            st.markdown("### Phim Gợi Ý Cho Bạn")
            render_result_list(results_exp, query_vec=avg_vec, tmdb_api_key=tmdb_api_key, context="exp")

# ══════════════════════════════ TAB PROFILE ══════════════════════════════════
with tab_profile:
    st.markdown("### Hồ Sơ Cá Nhân CineMatch")
    
    if not st.session_state.logged_in:
        st.info("Vui lòng đăng nhập hoặc đăng ký tài khoản ở Sidebar bên trái để truy cập hồ sơ cá nhân và quản lý sở thích.")
    else:
        # Tải thông tin tài khoản hiện hành
        accounts = load_user_accounts()
        user_info = accounts.get(st.session_state.username, {})
        created_at = user_info.get("created_at", "N/A")
        
        col_info, col_pwd = st.columns([2, 1], gap="large")
        
        with col_info:
            st.markdown(f"#### Xin chào, **{st.session_state.username}**")
            st.markdown(f"""
            * **Ngày đăng ký tài khoản:** {created_at}
            * **Tổng số phim yêu thích:** {len(st.session_state.favorites)} bộ phim
            """)
            
            # --- PHẦN PHÂN TÍCH GU PHIM CỦA BẠN ---
            st.markdown("---")
            st.markdown("#### Phân tích gu phim của bạn")
            
            if len(st.session_state.favorites) == 0:
                st.info("Danh sách phim yêu thích của bạn đang trống. Hãy tìm kiếm phim và nhấn nút để thêm phim yêu thích, AI sẽ tự động phân tích gu phim của bạn tại đây!")
            else:
                movie_to_idx = {row["movie"]: idx for idx, row in movie_profiles.iterrows()}
                idxs = [movie_to_idx[m] for m in st.session_state.favorites if m in movie_to_idx]
                
                if idxs:
                    sel_matrix = content_matrix[idxs]
                    # Trích xuất top 8 từ khóa đại diện cho gu phim
                    keywords = get_enriched_keywords(sel_matrix, tfidf_vec, n=8)
                    st.markdown("Dựa trên các bộ phim bạn đã thích, AI phân tích thấy gu phim của bạn nổi bật với các từ khóa và chủ đề sau:")
                    
                    kw_html = " ".join(f'<span class="theme-tag">{k}</span>' for k in keywords)
                    st.markdown(kw_html, unsafe_allow_html=True)
                    st.caption("Các từ khóa được trích xuất tự động bằng phân tích TF-IDF từ các review tích cực của những bộ phim bạn đã thích.")
                else:
                    st.warning("Không tìm thấy dữ liệu đặc trưng cho các phim trong danh sách thích.")
            
            # --- QUẢN LÝ DANH SÁCH THÍCH TẬP TRUNG ---
            st.markdown("---")
            st.markdown("#### Danh sách phim yêu thích")
            
            # Tiện ích thêm phim trực tiếp từ Hồ sơ
            st.markdown("**Thêm nhanh phim yêu thích mới:**")
            all_movie_names_profile = sorted(movie_profiles["movie"].tolist())
            unliked_movies = [m for m in all_movie_names_profile if m not in st.session_state.favorites]
            
            with st.form("add_favorite_form", clear_on_submit=True):
                col_add_sel, col_add_btn = st.columns([4, 1])
                with col_add_sel:
                    new_fav_movie = st.selectbox(
                        "Chọn phim để thêm:",
                        options=[" "] + unliked_movies,
                        label_visibility="collapsed",
                        key="select_new_fav_profile"
                    )
                with col_add_btn:
                    submit_add = st.form_submit_button("Thêm", use_container_width=True)
                
                if submit_add and new_fav_movie != " ":
                    st.session_state.favorites.append(new_fav_movie)
                    accounts[st.session_state.username]["favorites"] = st.session_state.favorites
                    save_user_accounts(accounts)
                    st.toast(f"Đã thêm '{new_fav_movie}' vào yêu thích!")
                    st.rerun()
            
            st.markdown("---")
            if not st.session_state.favorites:
                st.write("Chưa có phim nào trong danh sách yêu thích của bạn.")
            else:
                for m_fav in st.session_state.favorites:
                    c_fav_name, c_fav_btn = st.columns([5, 1])
                    with c_fav_name:
                        st.markdown(f"🎥 **{m_fav}**")
                    with c_fav_btn:
                        if st.button("Xóa", key=f"del_fav_tab_{m_fav}", use_container_width=True):
                            st.session_state.favorites.remove(m_fav)
                            accounts[st.session_state.username]["favorites"] = st.session_state.favorites
                            save_user_accounts(accounts)
                            st.toast(f"Đã xoá '{m_fav}'!")
                            st.rerun()

            # --- QUẢN LÝ ĐÁNH GIÁ & NHẬN XÉT ---
            st.markdown("---")
            st.markdown("#### Phim bạn đã đánh giá & nhận xét")
            
            rated_movies = list(st.session_state.ratings.keys())
            if not rated_movies:
                st.write("Bạn chưa đánh giá bộ phim nào.")
            else:
                for m_rated in rated_movies:
                    c_rated_info, c_rated_btn = st.columns([5, 1])
                    with c_rated_info:
                        m_r = st.session_state.ratings[m_rated]
                        m_rev = st.session_state.reviews.get(m_rated, "")
                        stars = rating_to_stars(m_r)
                        st.markdown(f"**{m_rated}** – {stars} ({m_r}/10)")
                        if m_rev:
                            st.caption(f"*Nhận xét:* {m_rev}")
                    with c_rated_btn:
                        if st.button("Xóa", key=f"del_rate_tab_{m_rated}", use_container_width=True):
                            if m_rated in st.session_state.ratings:
                                del st.session_state.ratings[m_rated]
                            if m_rated in st.session_state.reviews:
                                del st.session_state.reviews[m_rated]
                            if m_rated in st.session_state.favorites:
                                st.session_state.favorites.remove(m_rated)
                            
                            accounts[st.session_state.username]["ratings"] = st.session_state.ratings
                            accounts[st.session_state.username]["reviews"] = st.session_state.reviews
                            accounts[st.session_state.username]["favorites"] = st.session_state.favorites
                            save_user_accounts(accounts)
                            st.toast(f"Đã xoá đánh giá của phim '{m_rated}'!")
                            st.rerun()
        
        with col_pwd:
            st.markdown("#### Đổi mật khẩu")
            with st.form("change_password_form", clear_on_submit=True):
                old_pwd = st.text_input("Mật khẩu cũ", type="password", key="change_old_pwd")
                new_pwd = st.text_input("Mật khẩu mới", type="password", key="change_new_pwd")
                new_pwd_conf = st.text_input("Xác nhận mật khẩu mới", type="password", key="change_new_pwd_conf")
                submit_change_pwd = st.form_submit_button("Cập nhật mật khẩu", type="primary", use_container_width=True)
                
                if submit_change_pwd:
                    if not old_pwd or not new_pwd or not new_pwd_conf:
                        st.error("Vui lòng điền đủ thông tin!")
                    elif new_pwd != new_pwd_conf:
                        st.error("Mật khẩu xác nhận không khớp!")
                    elif accounts[st.session_state.username]["password"] != hash_password(old_pwd):
                        st.error("Mật khẩu cũ không đúng!")
                    else:
                        accounts[st.session_state.username]["password"] = hash_password(new_pwd)
                        save_user_accounts(accounts)
                        st.success("Đổi mật khẩu thành công!")
