"""
main_web.py  –  CineMatch v5
Giao diện Streamlit: SVM Sentiment · TF-IDF Cosine · SVD Collaborative Filtering

Cải tiến v5:
  1. MMR (Maximal Marginal Relevance) reranking – tăng diversity gợi ý
  2. Hybrid weight slider – người dùng tự điều chỉnh Cosine/SVD ratio
  3. Poster: iTunes API → Wikipedia API → Placeholder HTML đẹp (hash color + initials)
  4. Explainability: hiển thị từ khoá giao thoa (intersection keywords) theo từng phim
  5. Màu chữ WCAG AA, enriched keywords, snippet review + expander
  6. Loading states, error handling chuyên nghiệp
"""

import hashlib
import os
import re
import urllib.parse

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
    page_icon="🎬",
    layout="wide",
)

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
OUTPUT_DIR           = "Outputs"
MAX_USERS_IN_SIDEBAR = 50    # Giới hạn user hiển thị trong selectbox
SNIPPET_LEN          = 150   # Ký tự tối đa trong snippet review
MMR_LAMBDA           = 0.6   # λ trong MMR: cao → ưu tiên relevance; thấp → ưu tiên diversity
                              # Công thức: score = λ·rel - (1-λ)·max_sim_to_selected

# ---------------------------------------------------------------------------
# TẢI MÔ HÌNH
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="⏳ Đang tải mô hình AI...")
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
        st.error(f"❌ Thiếu file mô hình: {missing}\n→ Chạy `train_SVM_cosine_SVD.py` trước!")
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
    """Chuyển điểm 1–10 thành chuỗi sao ★☆ (thang 5 sao)."""
    try:
        n = int(round(float(rating) / 2))
        return "★" * n + "☆" * (5 - n)
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
    return initials[:3] if initials else "🎬"


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
def fetch_poster_url(movie_title: str) -> str | None:
    """
    Tìm URL poster phim (không cần API key).
    Fallback chain: iTunes Search API → Wikipedia REST API → None.
    Cache 24 giờ để tránh gọi API liên tục.
    """
    # 1. iTunes Search API – chất lượng tốt, không cần key
    try:
        r = requests.get(
            "https://itunes.apple.com/search",
            params={"term": movie_title, "media": "movie", "limit": 1, "entity": "movie"},
            timeout=3,
        )
        results = r.json().get("results", [])
        if results:
            url = results[0].get("artworkUrl100", "")
            if url:
                return url.replace("100x100bb", "300x300bb")
    except Exception:
        pass

    # 2. Wikipedia REST API – dự phòng
    try:
        r = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(movie_title)}",
            timeout=3,
        )
        if r.status_code == 200:
            url = r.json().get("thumbnail", {}).get("source")
            if url:
                return url
    except Exception:
        pass

    return None  # → Caller dùng _render_poster_placeholder


def render_poster(title: str) -> None:
    """
    Render poster phim. Nếu không tìm được URL → render placeholder HTML đẹp.
    Placeholder dùng màu hash từ tên phim + chữ viết tắt.
    """
    poster_url = fetch_poster_url(title)
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
    - Lọc từ quá chung (_GENERIC_TERMS)
    - Ưu tiên bigram/trigram (60%) trước unigram (40%)
    - Trả về n từ khoá đa dạng
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

    Cách tính:
      - Lấy min(query_vec, movie_vec) theo từng feature → phần giao thoa
      - Top N features trong phần giao thoa = lý do gợi ý

    Ý nghĩa: "Phim này được gợi ý vì cùng nói về [keyword1], [keyword2]..."
    """
    feature_names = vectorizer.get_feature_names_out()
    movie_vec     = np.asarray(content_matrix[movie_idx].todense()).flatten()

    # Nếu query_vec là sparse matrix, chuyển thành dense
    if hasattr(query_vec, "toarray"):
        q_vec = np.asarray(query_vec.toarray()).flatten()
    else:
        q_vec = np.asarray(query_vec).flatten()

    # Giao thoa = element-wise minimum (cả hai phải có giá trị > 0)
    intersection = np.minimum(q_vec, movie_vec)

    # Lọc từ quá chung và chỉ lấy n-gram hoặc unigram có nghĩa
    valid_idx = [
        i for i, f in enumerate(feature_names)
        if intersection[i] > 0 and f not in _GENERIC_TERMS
    ]
    top_idx = sorted(valid_idx, key=lambda i: intersection[i], reverse=True)[:n]
    return [feature_names[i] for i in top_idx]


# ---------------------------------------------------------------------------
# HELPERS – SVD PERSONALIZATION
# ---------------------------------------------------------------------------
def _is_personalized(user_key: str) -> bool:
    """Kiểm tra xem user_key có trong SVD model không."""
    return user_key != "🎭 Khách (Cold-start)" and user_key in svd_data["user_map"]


def _compute_svd_scores(candidate_movies: pd.Series, user_key: str) -> np.ndarray:
    """
    Tính điểm SVD cho danh sách phim ứng viên.
    Điểm = dot product(user_factor, item_factor).
    Trả về mảng 0 nếu cold-start.
    """
    scores = np.zeros(len(candidate_movies))
    if _is_personalized(user_key):
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
    Hybrid Scoring:
      Score = w_cos × Cosine_norm + (1 - w_cos) × SVD_norm   [nếu có user]
      Score = Cosine_norm                                       [cold-start]

    Tham số w_cos được truyền từ UI slider → người dùng tự điều chỉnh.
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

    Thuật toán:
      1. Khởi tạo: selected = {}; remaining = tất cả candidates
      2. Vòng lặp (n lần):
         a. Với mỗi phim r trong remaining, tính:
            MMR_score(r) = λ × final_score(r) − (1−λ) × max_sim(r, selected)
         b. Chọn phim có MMR_score cao nhất vào selected
         c. Loại phim đó khỏi remaining

    Ý nghĩa của λ (MMR_LAMBDA):
      λ = 1.0 → giống sort bình thường (không rerank)
      λ = 0.5 → cân bằng relevance và diversity
      λ = 0.0 → tối đa hóa diversity hoàn toàn

    Tại sao không dùng sort thông thường?
      - Sort thông thường có thể gợi ý 5 phần Harry Potter liên tiếp
      - MMR chọn phim Harry Potter 1, rồi phim tiếp theo phải vừa liên quan
        vừa khác biệt → có thể chọn 1 phim fantasy khác thay vì HP2

    Returns:
      DataFrame đã được rerank theo MMR
    """
    if len(candidates) <= top_n:
        return candidates.reset_index(drop=True)

    # Reset index để dùng positional indexing
    cands = candidates.reset_index(drop=True)
    scores = cands["final"].values

    selected_indices  = []  # indices đã chọn vào kết quả
    remaining_indices = list(range(len(cands)))

    for _ in range(top_n):
        if not remaining_indices:
            break

        best_idx   = None
        best_score = -float("inf")

        for r in remaining_indices:
            relevance = scores[r]

            # Tính độ tương đồng với các phim đã chọn
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

    Returns:
      results  : DataFrame top_n phim với cột cos_norm, svd_norm, final
      is_pers  : True nếu có cá nhân hóa SVD
      q_vec    : vector truy vấn TF-IDF (dùng cho explainability)
    """
    q_vec   = tfidf_vec.transform([query_en])
    cos_raw = cosine_similarity(q_vec, content_matrix).flatten()

    # Lấy top-150 ứng viên để MMR có đủ "pool" để chọn
    pool_size = max(150, top_n * 15)
    top_idx   = cos_raw.argsort()[-pool_size:][::-1]
    cands     = movie_profiles.iloc[top_idx].copy().reset_index(drop=True)
    cands["cos_raw"]       = cos_raw[top_idx]
    cands["_matrix_idx"]   = top_idx          # lưu index gốc để lookup matrix
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

    # Sắp xếp theo final score trước khi MMR
    cands = cands.sort_values("final", ascending=False)

    if use_mmr and len(cands) > top_n:
        # Xây dựng sub-matrix từ các candidates
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
    Gợi ý dựa trên danh sách phim đã xem.
    Tính vector "gu" trung bình → Cosine với catalogue → Hybrid + MMR.

    Returns:
      results  : DataFrame top_n phim gợi ý
      keywords : từ khoá đặc trưng của phim đã chọn
      avg_vec  : vector "gu" trung bình (dùng cho explainability)
    """
    selected_set = set(selected_titles)
    idxs = [i for i, m in enumerate(movie_profiles["movie"]) if m in selected_set]
    if not idxs:
        return pd.DataFrame(), [], None

    sel_matrix = content_matrix[idxs]
    avg_vec    = np.asarray(sel_matrix.mean(axis=0)).reshape(1, -1)  # (1, n_features)
    cos_raw    = cosine_similarity(avg_vec, content_matrix).flatten()

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
    Render card phim với:
    - Thanh điểm số (gradient)
    - Chips Cosine / SVD / Tổng hợp
    - (Tuỳ chọn) Keywords giải thích tại sao phim được gợi ý
    """
    pct = final * 100
    explain_html = ""
    if explain_keywords:
        chips = " ".join(
            f'<span class="theme-tag-ex">🔑 {kw}</span>'
            for kw in explain_keywords
        )
        explain_html = f'<div style="margin-top:8px;">{chips}</div>'

    st.markdown(f"""
    <div class="mcard">
      <div><span class="rank">#{rank}</span>
           <span class="mcard-title">🎬 {title}</span></div>
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
    Render 3 review hàng đầu:
    - Non-spoiler: snippet SNIPPET_LEN ký tự + expander "Xem đầy đủ"
    - Spoiler: ẩn trong expander có cảnh báo
    """
    sub         = reviews_db[reviews_db["movie"] == movie_title]
    top_reviews = sub.sort_values(["spoiler_tag", "rating"], ascending=[True, False]).head(3)

    if top_reviews.empty:
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
            with st.expander(f"⚠️ Review của {reviewer} – SPOILER (nhấn để xem)"):
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


def render_links_section(movie_title: str) -> None:
    """Render nút trailer YouTube và link IMDb."""
    c1, c2 = st.columns(2)
    with c1:
        st.link_button("▶️ Trailer YouTube", build_youtube_url(movie_title), use_container_width=True)
    with c2:
        st.link_button("🎞️ Xem trên IMDb",  build_imdb_url(movie_title),   use_container_width=True)
    st.caption("💡 Nhấn để mở tab mới.")


def render_result_list(
    results: pd.DataFrame,
    query_vec=None,
) -> None:
    """
    Render danh sách kết quả gợi ý:
    - Cột trái: Poster (iTunes → Wiki → Placeholder HTML màu hash)
    - Cột phải: Card phim (có Explainability keywords nếu có query_vec)
                + Tab Đánh giá / Trailer

    Tham số query_vec: vector truy vấn hoặc vector "gu" người dùng.
    Dùng để tính intersection keywords (giải thích tại sao phim được gợi ý).
    """
    # Lập index: tên phim → vị trí trong content_matrix
    _movie_to_matrix_idx = {
        row["movie"]: idx for idx, row in movie_profiles.iterrows()
    }

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        title = row["movie"]

        # Lấy keywords giải thích (chỉ khi có query vector)
        explain_kws = []
        if query_vec is not None and title in _movie_to_matrix_idx:
            m_idx = _movie_to_matrix_idx[title]
            explain_kws = get_movie_explain_keywords(query_vec, m_idx, tfidf_vec, n=4)

        col_poster, col_main = st.columns([1, 4], gap="medium")

        with col_poster:
            render_poster(title)

        with col_main:
            render_movie_card(
                rank, title, row["final"], row["cos_norm"], row["svd_norm"],
                explain_keywords=explain_kws,
            )
            tab_reviews, tab_links = st.tabs(["📝 Đánh giá cộng đồng", "▶️ Trailer & Links"])
            with tab_reviews:
                render_reviews_section(title)
            with tab_links:
                render_links_section(title)

        st.markdown("---")


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎬 CineMatch")
    st.caption("Hệ thống Gợi Ý Phim – SVM + TF-IDF + Cosine + SVD")
    st.divider()

    top_n = st.slider("🎯 Số phim gợi ý", min_value=3, max_value=10, value=5)

    st.markdown("**👤 Hồ sơ người dùng (SVD)**")
    all_users     = list(svd_data["user_map"].keys())
    selected_user = st.selectbox(
        "Người dùng:",
        options=["🎭 Khách (Cold-start)"] + all_users[:MAX_USERS_IN_SIDEBAR],
        label_visibility="collapsed",
    )
    st.divider()

    # Hybrid weight slider – cải tiến v5
    st.markdown("**⚖️ Tỉ lệ Hybrid Scoring**")
    w_cosine = st.slider(
        "Content (Cosine) ↔ Collab (SVD)",
        min_value=0.3, max_value=1.0, value=0.6, step=0.05,
        help="0.6 = 60% Content-Based + 40% Collaborative. Cold-start luôn dùng 100% Content.",
        format="%.2f",
    )
    w_svd_display = 1.0 - w_cosine
    st.caption(f"Cosine: **{w_cosine:.0%}** | SVD: **{w_svd_display:.0%}**")
    st.divider()

    # MMR toggle
    use_mmr = st.checkbox(
        "🔀 Bật MMR Diversity",
        value=True,
        help="Maximal Marginal Relevance – giúp kết quả đa dạng hơn, tránh gợi ý các phần của cùng 1 series.",
    )
    st.divider()

    with st.expander("⚙️ Về Hybrid Scoring & MMR"):
        st.markdown(f"""
**Hybrid Scoring:**  
`Score = {w_cosine:.0%} × Cosine + {w_svd_display:.0%} × SVD`

Cả hai điểm đều được **normalize về [0,1]** trước khi kết hợp.

**MMR (Maximal Marginal Relevance):**  
`MMR = λ × Relevance − (1−λ) × MaxSim(selected)`  
Phạt điểm phim quá giống phim đã chọn → kết quả đa dạng hơn.
        """)

# ---------------------------------------------------------------------------
# TABS CHÍNH
# ---------------------------------------------------------------------------
tab_home, tab_search, tab_exp = st.tabs([
    "🏠  Trang Chủ",
    "🔍  Gợi Ý Theo Từ Khoá",
    "🎭  Trải Nghiệm Xem Phim",
])

# ══════════════════════════════ TAB HOME ════════════════════════════════════
with tab_home:
    st.markdown("""
    <div class="hero">
      <h1>🎬 CineMatch – Gợi Ý Phim Thông Minh</h1>
      <p>Kết hợp Phân tích Cảm xúc (SVM), Độ tương đồng nội dung (TF-IDF + Cosine)
         và Lọc cộng tác (SVD) để gợi ý phim phù hợp nhất với bạn.</p>
    </div>""", unsafe_allow_html=True)

    # Đọc SVM Accuracy từ file metrics (tên chuẩn)
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
    st.markdown("### 🏗️ Kiến Trúc Hybrid Pipeline")
    st.markdown(f"""
| Tầng | Kỹ thuật | Vai trò |
|---|---|---|
| 1 | **Dịch thuật tự động** | Hỗ trợ truy vấn tiếng Việt |
| 2 | **TF-IDF + Cosine Similarity** | Tìm phim có review tương đồng với truy vấn |
| 3 | **SVD (Collaborative Filtering)** | Cá nhân hóa theo lịch sử người dùng |
| 4 | **MMR Reranking** | Đa dạng hóa kết quả, tránh gợi ý lặp |
| 5 | **Lọc Spoiler** | Ẩn review tiết lộ nội dung phim |

**Hybrid Weight hiện tại:** {w_cosine:.0%} Cosine + {w_svd_display:.0%} SVD  
→ Điều chỉnh tỉ lệ này trong Sidebar để thay đổi hành vi gợi ý.
    """)

# ══════════════════════════════ TAB SEARCH ══════════════════════════════════
with tab_search:
    st.markdown("#### 🔍 Nhập mô tả nội dung phim bạn muốn xem")

    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input(
            "query", label_visibility="collapsed",
            placeholder="VD: Phim kinh dị siêu nhiên, ma quái trong ngôi nhà bỏ hoang...",
        )
    with col_btn:
        do_search = st.button("Tìm phim", type="primary", use_container_width=True)

    QUICK_SEARCHES = [
        "Kinh dị ma quái", "Tình cảm lãng mạn",
        "Viễn tưởng không gian", "Tội phạm trinh thám", "Hoạt hình gia đình",
    ]
    for col, label in zip(st.columns(len(QUICK_SEARCHES)), QUICK_SEARCHES):
        if col.button(label, key=f"quick_{label}"):
            query     = label
            do_search = True

    if do_search and not query:
        st.warning("⚠️ Vui lòng nhập mô tả phim!")

    elif do_search and query:
        with st.status("🤖 Đang phân tích...", expanded=True) as status:
            st.write("🌐 Dịch thuật sang tiếng Anh...")
            try:
                from deep_translator import GoogleTranslator
                query_en = GoogleTranslator(source="auto", target="en").translate(query)
            except Exception:
                query_en = query
            st.write(f'→ **"{query_en}"**')
            st.write("🔍 Content-Based Filtering...")
            st.write("🧠 SVD Collaborative Filtering...")
            st.write("🔀 MMR Reranking...")
            results, is_pers, q_vec = recommend_by_query(
                query_en, selected_user, top_n,
                w_cos=w_cosine, use_mmr=use_mmr,
            )
            status.update(label="✅ Hoàn tất!", state="complete", expanded=False)

        mode_label = (
            f"👤 Cá nhân hóa: **{selected_user}**"
            if is_pers else "🎭 **Cold-start** (100% Content-Based)"
        )
        mmr_label = "🔀 MMR bật" if use_mmr else "📋 MMR tắt"
        st.markdown(
            f'<div class="alert-i">{mode_label} &nbsp;|&nbsp; {mmr_label} &nbsp;|&nbsp; '
            f'Truy vấn: <em>"{query_en}"</em></div>',
            unsafe_allow_html=True,
        )

        if results.empty:
            st.error("Không tìm thấy phim phù hợp. Thử từ khoá khác.")
        else:
            render_result_list(results, query_vec=q_vec)

# ══════════════════════════════ TAB EXPERIENCE ══════════════════════════════
with tab_exp:
    st.markdown("#### 🎭 Gợi Ý Dựa Trên Phim Bạn Đã Xem")
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
        do_exp = st.button("🔍 Gợi ý phim tương tự", type="primary", use_container_width=True)

    if selected_movies:
        st.markdown("**✅ Phim đã chọn:**")
        tags_html = " ".join(
            f'<span class="theme-tag">🎬 {m}</span>' for m in selected_movies
        )
        st.markdown(tags_html, unsafe_allow_html=True)
        st.markdown("")

    if do_exp and not selected_movies:
        st.warning("⚠️ Vui lòng chọn ít nhất 1 phim!")

    elif do_exp and selected_movies:
        with st.status("🧠 Đang phân tích gu phim của bạn...", expanded=True) as status:
            st.write(f"📌 Phân tích {len(selected_movies)} phim đã chọn...")
            st.write("🔍 Tính vector 'gu' trung bình...")
            st.write("🔀 MMR Reranking...")
            results_exp, keywords, avg_vec = recommend_by_movies(
                selected_movies, selected_user, top_n,
                w_cos=w_cosine, use_mmr=use_mmr,
            )
            status.update(label="✅ Hoàn tất!", state="complete", expanded=False)

        if results_exp.empty:
            st.error("Không tìm thấy phim phù hợp.")
        else:
            st.markdown("---")
            st.markdown("### 🔑 Điểm Chung Giữa Các Phim Đã Chọn")
            st.caption("Các từ khoá nổi bật trong review – đây là 'gu' phim của bạn:")
            kw_html = " ".join(f'<span class="theme-tag">{k}</span>' for k in keywords)
            st.markdown(kw_html, unsafe_allow_html=True)

            is_pers_exp = _is_personalized(selected_user)
            mode_label  = (
                f"👤 Cá nhân hóa: **{selected_user}**"
                if is_pers_exp else "🎭 Cold-start (100% Content-Based)"
            )
            mmr_label = "🔀 MMR bật" if use_mmr else "📋 MMR tắt"
            st.markdown(
                f'<div class="alert-g" style="margin-top:12px;">✅ {mode_label}'
                f' &nbsp;|&nbsp; {mmr_label}'
                f' &nbsp;|&nbsp; Tìm thấy <strong>{len(results_exp)}</strong> phim phù hợp</div>',
                unsafe_allow_html=True,
            )

            st.markdown("### 🎯 Phim Gợi Ý Cho Bạn")
            render_result_list(results_exp, query_vec=avg_vec)

