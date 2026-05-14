import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import urllib.parse
from deep_translator import GoogleTranslator
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CineMatch – Gợi Ý Phim AI", page_icon="🎬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Be Vietnam Pro',sans-serif;}
.stApp{background:#0d1117;}

/* Sidebar */
section[data-testid="stSidebar"]>div:first-child{background:#161b22;border-right:1px solid #30363d;}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] .stMarkdown p {color:#c9d1d9 !important;}
section[data-testid="stSidebar"] .stSelectbox label, section[data-testid="stSidebar"] .stSlider label {color:#c9d1d9 !important;}

/* Global text */
.stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {color:#c9d1d9;}
h1,h2,h3,h4{color:#e6edf3;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:#161b22;border-radius:10px;padding:4px;gap:4px;border:1px solid #30363d;}
.stTabs [data-baseweb="tab"]{border-radius:8px;color:#8b949e;font-weight:600;padding:8px 20px;}
.stTabs [aria-selected="true"]{background:#e50914 !important;color:#fff !important;}

/* Cards */
.mcard{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px 24px;margin-bottom:16px;}
.mcard:hover{border-color:#e50914;}
.mcard-title{font-size:1.2rem;font-weight:700;color:#e6edf3;margin-bottom:8px;}
.rank{display:inline-block;background:rgba(229,9,20,0.15);color:#e50914;font-weight:700;padding:2px 10px;border-radius:6px;margin-right:8px;font-size:1rem;}
.bar-bg{background:#21262d;border-radius:999px;height:7px;margin:6px 0 10px;}
.bar-fill{height:7px;border-radius:999px;background:linear-gradient(90deg,#e50914,#ff6b35);}
.chip{display:inline-block;background:#21262d;color:#8b949e;font-size:0.75rem;padding:3px 10px;border-radius:999px;margin:2px;}
.chip-g{background:rgba(46,160,67,0.15);color:#3fb950;}
.chip-b{background:rgba(88,166,255,0.15);color:#58a6ff;}
.chip-y{background:rgba(210,153,34,0.15);color:#d2a520;}
.rbox{background:#0d1117;border-left:3px solid #30363d;border-radius:0 8px 8px 0;padding:10px 14px;margin:6px 0;color:#8b949e;font-size:0.85rem;}
.rbox strong{color:#c9d1d9;}
.hero{background:linear-gradient(135deg,#1a0a0a,#2d0808,#1a0505);border:1px solid #3d0f0f;border-radius:14px;padding:28px 36px;margin-bottom:20px;}
.hero h1{color:#fff;font-size:1.9rem;font-weight:700;margin:0 0 6px;}
.hero p{color:#8b949e;margin:0;font-size:0.93rem;}
.mbox{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;text-align:center;}
.mbox .val{font-size:1.5rem;font-weight:700;color:#e50914;}
.mbox .lbl{font-size:0.78rem;color:#8b949e;margin-top:4px;}
.alert-i{background:rgba(88,166,255,0.08);border:1px solid rgba(88,166,255,0.3);border-radius:8px;padding:10px 14px;color:#58a6ff;font-size:0.87rem;margin:8px 0;}
.alert-g{background:rgba(46,160,67,0.08);border:1px solid rgba(46,160,67,0.3);border-radius:8px;padding:10px 14px;color:#3fb950;font-size:0.87rem;margin:8px 0;}
.theme-tag{display:inline-block;background:rgba(210,153,34,0.15);color:#d2a520;border:1px solid rgba(210,153,34,0.3);padding:4px 12px;border-radius:999px;font-size:0.8rem;margin:3px;}
div[data-testid="column"] .stButton>button{background:#21262d;border:1px solid #30363d;color:#8b949e;border-radius:999px;font-size:0.8rem;width:100%;}
div[data-testid="column"] .stButton>button:hover{border-color:#e50914;color:#e50914;}
</style>
""", unsafe_allow_html=True)

OUTPUT_DIR = "Outputs"

@st.cache_resource(show_spinner="⏳ Đang tải mô hình AI...")
def load_models():
    try:
        return (
            joblib.load(os.path.join(OUTPUT_DIR, 'model_sentiment_linearsvc.joblib')),
            joblib.load(os.path.join(OUTPUT_DIR, 'vectorizer_tfidf.joblib')),
            joblib.load(os.path.join(OUTPUT_DIR, 'content_movie_profiles.joblib')),
            joblib.load(os.path.join(OUTPUT_DIR, 'content_tfidf_matrix.joblib')),
            joblib.load(os.path.join(OUTPUT_DIR, 'web_display_reviews.joblib')),
            joblib.load(os.path.join(OUTPUT_DIR, 'collab_svd_model.joblib')),
        )
    except Exception as e:
        st.error(f"❌ Không tải được mô hình: {e} — Chạy train_SVM_cosine_SVD.py trước!")
        st.stop()

svm_model, tfidf_vec, movie_profiles, content_matrix, reviews_db, svd_data = load_models()

# ── Helpers ──────────────────────────────────────────────────────
def yt_url(t): return f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(t+' official trailer')}"
def imdb_url(t): return f"https://www.imdb.com/find?q={urllib.parse.quote_plus(t)}&s=tt"

def stars(r):
    try: n=int(round(float(r)/2)); return "★"*n+"☆"*(5-n)
    except: return ""

def get_reviews(movie, n=3):
    sub = reviews_db[reviews_db['movie']==movie]
    return sub.sort_values(['spoiler_tag','rating'],ascending=[True,False]).head(n)

def normalize_01(arr):
    mn,mx = arr.min(),arr.max()
    return np.ones_like(arr) if mx-mn<1e-9 else (arr-mn)/(mx-mn)

def get_top_keywords(vec_matrix, vectorizer, n=8):
    """Trích xuất từ khoá nổi bật từ vector TF-IDF."""
    feat = vectorizer.get_feature_names_out()
    avg  = np.asarray(vec_matrix.mean(axis=0)).flatten()
    top_idx = avg.argsort()[-n:][::-1]
    return [feat[i] for i in top_idx]

def recommend_by_query(query_en, user_key, top_n):
    q_vec = tfidf_vec.transform([query_en])
    cos_raw = cosine_similarity(q_vec, content_matrix).flatten()
    top_idx = cos_raw.argsort()[-100:][::-1]
    cands = movie_profiles.iloc[top_idx].copy().reset_index(drop=True)
    cands['cos_raw'] = cos_raw[top_idx]
    cands = cands[cands['cos_raw']>0].reset_index(drop=True)
    cands['cos_norm'] = normalize_01(cands['cos_raw'].values)

    svd_scores = np.zeros(len(cands))
    is_pers = user_key!="🎭 Khách (Cold-start)" and user_key in svd_data['user_map']
    if is_pers:
        u_vec = svd_data['user_factors'][svd_data['user_map'][user_key]]
        for i,m in enumerate(cands['movie']):
            if m in svd_data['movie_map']:
                svd_scores[i] = np.dot(u_vec, svd_data['item_factors'][svd_data['movie_map'][m]])

    svd_norm = normalize_01(svd_scores)
    w = 0.6 if is_pers else 1.0
    cands['final']    = w*cands['cos_norm'] + (1-w)*svd_norm
    cands['svd_norm'] = svd_norm
    return cands.sort_values('final',ascending=False).head(top_n), is_pers

def recommend_by_movies(selected_titles, user_key, top_n, exclude=True):
    """
    Gợi ý dựa trên danh sách phim đã chọn:
    1. Lấy vector TF-IDF trung bình của các phim đã chọn
    2. Tính cosine với toàn bộ ma trận phim
    3. Kết hợp SVD nếu có người dùng
    """
    idxs = [i for i,m in enumerate(movie_profiles['movie']) if m in selected_titles]
    if not idxs:
        return pd.DataFrame(), np.array([]), []

    # Vector trung bình của các phim đã chọn
    import scipy.sparse as sp
    sel_matrix = content_matrix[idxs]
    avg_vec = np.asarray(sel_matrix.mean(axis=0)).reshape(1, -1)  # (1, n_features)

    cos_raw = cosine_similarity(avg_vec, content_matrix).flatten()

    cands = movie_profiles.copy().reset_index(drop=True)
    cands['cos_raw'] = cos_raw

    # Loại phim đã chọn khỏi gợi ý
    if exclude:
        cands = cands[~cands['movie'].isin(selected_titles)]

    cands = cands[cands['cos_raw']>0].sort_values('cos_raw',ascending=False).head(150).reset_index(drop=True)
    cands['cos_norm'] = normalize_01(cands['cos_raw'].values)

    svd_scores = np.zeros(len(cands))
    is_pers = user_key!="🎭 Khách (Cold-start)" and user_key in svd_data['user_map']
    if is_pers:
        u_vec = svd_data['user_factors'][svd_data['user_map'][user_key]]
        for i,m in enumerate(cands['movie']):
            if m in svd_data['movie_map']:
                svd_scores[i] = np.dot(u_vec, svd_data['item_factors'][svd_data['movie_map'][m]])

    svd_norm = normalize_01(svd_scores)
    w = 0.6 if is_pers else 1.0
    cands['final']    = w*cands['cos_norm'] + (1-w)*svd_norm
    cands['svd_norm'] = svd_norm

    # Trích xuất từ khoá chung
    keywords = get_top_keywords(sel_matrix, tfidf_vec, n=10)
    return cands.sort_values('final',ascending=False).head(top_n), svd_norm, keywords

def render_movie_card(rank, title, final, cos_n, svd_n):
    pct = final*100
    st.markdown(f"""
    <div class="mcard">
      <div><span class="rank">#{rank}</span><span class="mcard-title">🎬 {title}</span></div>
      <div style="margin-top:6px;">
        <span class="chip chip-b">Cosine {cos_n:.3f}</span>
        <span class="chip chip-g">SVD {svd_n:.3f}</span>
        <span class="chip chip-y">Tổng hợp {pct:.1f}%</span>
      </div>
      <div class="bar-bg"><div class="bar-fill" style="width:{min(pct,100):.1f}%"></div></div>
    </div>""", unsafe_allow_html=True)

def render_reviews_tab(title):
    reviews = get_reviews(title)
    if reviews.empty:
        st.info("Chưa có đánh giá lưu trong hệ thống.")
        return
    for _,rv in reviews.iterrows():
        reviewer = rv.get('reviewer','Ẩn danh')
        rating   = rv.get('rating','?')
        text     = str(rv.get('review',''))[:320]
        is_spoil = int(rv.get('spoiler_tag',0))==1
        s = stars(rating)
        if is_spoil:
            with st.expander(f"⚠️ Review của {reviewer} – SPOILER"):
                st.write(f"{s} {rating}/10 | {text}...")
        else:
            st.markdown(f'<div class="rbox"><strong>{reviewer}</strong> {s} <em>({rating}/10)</em><br>{text}{"..." if len(str(rv.get("review","")))>320 else ""}</div>', unsafe_allow_html=True)

def render_links_tab(title):
    c1,c2 = st.columns(2)
    with c1: st.link_button("▶️ Trailer YouTube", yt_url(title), use_container_width=True)
    with c2: st.link_button("🎞️ Xem trên IMDb",  imdb_url(title), use_container_width=True)
    st.caption("💡 YouTube không cho nhúng trực tiếp. Nhấn để mở tab mới.")

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎬 CineMatch")
    st.caption("Hệ thống Gợi Ý Phim – SVM + TF-IDF + Cosine + SVD")
    st.divider()
    top_n = st.slider("🎯 Số phim gợi ý", 3, 10, 5)
    st.markdown("**👤 Hồ sơ người dùng (SVD)**")
    all_users = list(svd_data['user_map'].keys())
    selected_user = st.selectbox("Người dùng:", ["🎭 Khách (Cold-start)"]+all_users[:50], label_visibility="collapsed")
    st.divider()
    with st.expander("⚙️ Về cá nhân hóa SVD"):
        st.markdown("""
Điểm Cosine và SVD đều được **normalize về \[0,1\]** rồi kết hợp:  
**60% Cosine + 40% SVD** (nếu có user) hoặc **100% Cosine** (Cold-start).

Mỗi user có vector riêng → chọn user khác → thứ hạng phim thay đổi.
        """)

# ── Tabs ─────────────────────────────────────────────────────────
tab_home, tab_search, tab_exp = st.tabs([
    "🏠  Trang Chủ",
    "🔍  Gợi Ý Theo Từ Khoá",
    "🎭  Trải Nghiệm Xem Phim",
])

# ════════════════════════════════ TAB HOME ════════════════════════
with tab_home:
    st.markdown("""
    <div class="hero">
      <h1>🎬 CineMatch – Gợi Ý Phim Thông Minh</h1>
      <p>Kết hợp Phân tích Cảm xúc (SVM), Độ tương đồng nội dung (TF-IDF + Cosine) và Lọc cộng tác (SVD) để gợi ý phim phù hợp nhất với bạn.</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    ev = svd_data.get('explained_variance',0)
    for col,val,lbl in zip([c1,c2,c3,c4],
        [f"{len(movie_profiles):,}", f"{len(svd_data['user_map']):,}", "92.15%", f"{ev*100:.1f}%"],
        ["Phim trong hệ thống","Người dùng","SVM Accuracy","SVD Explained Var."]):
        col.markdown(f'<div class="mbox"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏗️ Kiến Trúc Hybrid Pipeline")
    st.markdown("""
| Tầng | Kỹ thuật | Vai trò |
|---|---|---|
| 1 | **Dịch thuật tự động** | Hỗ trợ truy vấn tiếng Việt |
| 2 | **TF-IDF + Cosine Similarity** | Tìm phim có review tương đồng với truy vấn |
| 3 | **SVD (Collaborative Filtering)** | Cá nhân hóa theo lịch sử người dùng |
| 4 | **Lọc Spoiler** | Ẩn review tiết lộ nội dung phim |

**Tab "Trải Nghiệm Xem Phim":** Chọn những phim bạn đã xem → hệ thống phân tích điểm chung → gợi ý phim mới phù hợp gu của bạn.
    """)

# ════════════════════════════════ TAB SEARCH ═════════════════════
with tab_search:
    st.markdown("#### 🔍 Nhập mô tả nội dung phim bạn muốn xem")
    col_q,col_b = st.columns([5,1])
    with col_q:
        query = st.text_input("q", label_visibility="collapsed",
            placeholder="VD: Phim kinh dị siêu nhiên, ma quái trong ngôi nhà bỏ hoang...")
    with col_b:
        do_search = st.button("Tìm phim", type="primary", use_container_width=True)

    QUICK = ["Kinh dị ma quái","Tình cảm lãng mạn","Viễn tưởng không gian","Tội phạm trinh thám","Hoạt hình gia đình"]
    qcols = st.columns(5)
    for i,(qc,ql) in enumerate(zip(qcols,QUICK)):
        if qc.button(ql, key=f"q{i}"):
            query=ql; do_search=True

    if do_search and not query:
        st.warning("⚠️ Vui lòng nhập mô tả phim!")
    elif do_search and query:
        with st.status("🤖 Đang phân tích...", expanded=True) as status:
            st.write("🌐 Dịch thuật...")
            try: query_en = GoogleTranslator(source='auto',target='en').translate(query)
            except: query_en = query
            st.write(f"→ **\"{query_en}\"**")
            st.write("🔍 Content-Based + 🧠 SVD...")
            results, is_pers = recommend_by_query(query_en, selected_user, top_n)
            status.update(label="✅ Hoàn tất!", state="complete", expanded=False)

        mode = f"👤 Cá nhân hóa: **{selected_user}**" if is_pers else "🎭 **Cold-start** (100% Content-Based)"
        st.markdown(f'<div class="alert-i">{mode} &nbsp;|&nbsp; Truy vấn: <em>"{query_en}"</em></div>', unsafe_allow_html=True)

        if results.empty:
            st.error("Không tìm thấy phim phù hợp. Thử từ khoá khác.")
        else:
            for rank,(_,row) in enumerate(results.iterrows(),1):
                render_movie_card(rank, row['movie'], row['final'], row['cos_norm'], row['svd_norm'])
                r_tab,l_tab = st.tabs(["📝 Đánh giá cộng đồng","▶️ Trailer & Links"])
                with r_tab: render_reviews_tab(row['movie'])
                with l_tab: render_links_tab(row['movie'])
                st.markdown("---")

# ════════════════════════════════ TAB EXPERIENCE ═════════════════
with tab_exp:
    st.markdown("#### 🎭 Gợi Ý Dựa Trên Phim Bạn Đã Xem")
    st.markdown("""
    <div class="alert-i">
    Chọn các phim bạn đã xem và thích. Hệ thống sẽ phân tích <strong>điểm chung</strong> giữa các phim đó
    (dựa trên nội dung review TF-IDF) rồi tìm những phim tương tự bạn chưa xem.
    </div>""", unsafe_allow_html=True)

    all_movie_names = sorted(movie_profiles['movie'].tolist())

    col_sel, col_cfg = st.columns([3,1])
    with col_sel:
        selected_movies = st.multiselect(
            "Chọn phim đã xem (gõ để tìm kiếm):",
            options=all_movie_names,
            placeholder="Tìm và chọn phim...",
            max_selections=10
        )
    with col_cfg:
        st.markdown("<br>", unsafe_allow_html=True)
        do_exp = st.button("🔍 Gợi ý phim tương tự", type="primary", use_container_width=True)

    if selected_movies:
        # Hiển thị phim đã chọn
        st.markdown("**✅ Phim đã chọn:**")
        tags = " ".join([f'<span class="theme-tag">🎬 {m}</span>' for m in selected_movies])
        st.markdown(tags, unsafe_allow_html=True)
        st.markdown("")

    if do_exp and not selected_movies:
        st.warning("⚠️ Vui lòng chọn ít nhất 1 phim!")

    elif do_exp and selected_movies:
        with st.status("🧠 Đang phân tích gu phim của bạn...", expanded=True) as status:
            st.write(f"📌 Phân tích {len(selected_movies)} phim đã chọn...")
            results_exp, svd_arr, keywords = recommend_by_movies(selected_movies, selected_user, top_n)
            status.update(label="✅ Hoàn tất!", state="complete", expanded=False)

        if results_exp.empty:
            st.error("Không tìm thấy phim phù hợp.")
        else:
            # Hiển thị điểm chung tìm được
            st.markdown("---")
            st.markdown("### 🔑 Điểm Chung Giữa Các Phim Đã Chọn")
            st.caption("Các từ khoá nổi bật trong review của những phim bạn thích – đây là 'gu' phim của bạn:")
            kw_html = " ".join([f'<span class="theme-tag">{k}</span>' for k in keywords])
            st.markdown(kw_html, unsafe_allow_html=True)

            mode = f"👤 Cá nhân hóa: **{selected_user}**" if (selected_user!="🎭 Khách (Cold-start)") else "🎭 Cold-start (100% Content-Based)"
            st.markdown(f'<div class="alert-g" style="margin-top:12px;">✅ {mode} &nbsp;|&nbsp; Tìm thấy <strong>{len(results_exp)}</strong> phim phù hợp</div>', unsafe_allow_html=True)

            st.markdown("### 🎯 Phim Gợi Ý Cho Bạn")
            for rank,(_,row) in enumerate(results_exp.iterrows(),1):
                render_movie_card(rank, row['movie'], row['final'], row['cos_norm'], row['svd_norm'])
                r_tab,l_tab = st.tabs(["📝 Đánh giá cộng đồng","▶️ Trailer & Links"])
                with r_tab: render_reviews_tab(row['movie'])
                with l_tab: render_links_tab(row['movie'])
                st.markdown("---")

# ── Footer ────────────────────────────────────────────────────────
st.markdown("""
<hr style="border:none;border-top:1px solid #21262d;margin-top:20px;">
<p style="text-align:center;color:#484f58;font-size:0.78rem;">
CineMatch · SVM + TF-IDF + Cosine Similarity + SVD · Đồ án KPDL 2026
</p>""", unsafe_allow_html=True)