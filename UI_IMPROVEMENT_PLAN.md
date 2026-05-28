# 🎨 Kế Hoạch Cải Thiện Giao Diện & Tính Năng – CineMatch

> **Trạng thái:** v1 – Phân tích (21/05/2026) | **v2 – Triển khai một phần (21/05/2026)**  
> **Phạm vi:** `main_web.py` (giao diện)

> [!NOTE]
> **Đã hoàn thành (v2):** Fix màu chữ · Cải thiện từ khoá · Snippet review · Poster layout (chờ API key)

---

## 0. Đánh Giá Tổng Quan Hiện Trạng

Hệ thống hiện tại hoạt động **đúng về mặt chức năng** nhưng còn một số vấn đề về UX và
tính thẩm mỹ cần khắc phục. Đánh giá dưới đây mang tính **khách quan và khắt khe**,
tập trung vào chất lượng thực sự.

| Hạng mục | Trước v2 | Sau v2 | Ghi chú |
|---|---|---|---|
| Chức năng cốt lõi | ✅ Tốt | ✅ Tốt | Không thay đổi |
| Giao diện tổng thể | ⚠️ 65/100 | ✅ 80/100 | Màu chữ, button, card |
| Thông tin hiển thị | ❌ 50/100 | ⚠️ 65/100 | Poster chờ API key |
| Từ khoá kết quả | ❌ 40/100 | ✅ 75/100 | Đã dùng `get_enriched_keywords` |
| Review snippet | ⚠️ 60/100 | ✅ 85/100 | 150 ký tự + expander xem đầy đủ |

---

## 1. Lỗi SVM Accuracy Hiển Thị "N/A"

### Nguyên nhân xác định

File `evaluation_metrics.txt` **không tồn tại** trong thư mục `Outputs/`.  
Các file hiện có là:
- `evaluation_metrics_onlypart3_100.txt`
- `evaluation_metrics_part1-3.txt`
- `evaluation_metrics_part1-3_ver2.txt`

Code mới (v4) tìm đúng tên `evaluation_metrics.txt` (do pipeline khi chạy lại mới tạo ra
file này) nhưng file đó chưa được tạo vì chưa chạy lại training hoàn chỉnh.

### Giải pháp

**Cách 1 – Ngắn hạn (không cần retrain):**  
Sửa code đọc file: ưu tiên `evaluation_metrics.txt`, nếu không có thì quét tất cả
`evaluation_metrics*.txt` trong Outputs, lấy file mới nhất.

```python
# Thay vì chỉ tìm 1 tên cố định:
_metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.txt")

# Quét pattern rộng hơn:
import glob as _glob
_candidates = sorted(_glob.glob(os.path.join(OUTPUT_DIR, "evaluation_metrics*.txt")))
_metrics_path = _candidates[-1] if _candidates else None
```

**Cách 2 – Dài hạn (đúng nhất):**  
Chạy lại `train_SVM_cosine_SVD.py` để tạo `evaluation_metrics.txt` với tên chuẩn.
Kết quả tốt nhất hiện có: **92.26%** (từ `evaluation_metrics_part1-3_ver2.txt`).

### Độ ưu tiên: 🔴 Cao – Lỗi hiển thị rõ ràng, sửa nhanh trong vài dòng code

---

## 2. Màu Chữ Chưa Hợp Giao Diện

### Vấn đề quan sát được từ ảnh

| Vị trí | Vấn đề | Mức độ |
|---|---|---|
| Nút quick search (Kinh dị ma quái...) | Chữ tối (#8b949e) trên nền tối (#21262d), tương phản thấp | ⚠️ Trung bình |
| Caption "YouTube không cho nhúng..." | Màu mặc định Streamlit, không khớp theme | ⚠️ Nhẹ |
| Tab "Đánh giá cộng đồng / Trailer" | Màu tab active đỏ rõ, nhưng tab inactive gần như không đọc được | ⚠️ Trung bình |
| Text trong `.rbox` (review) | `#8b949e` trên `#0d1117` – tương phản chưa đủ WCAG AA | ❌ Cần sửa |
| Sidebar selectbox label | Đã có `!important` nhưng vẫn có thể bị override bởi Streamlit | ⚠️ Nhẹ |

### Ngưỡng tương phản chuẩn (WCAG AA)

- **Text thường:** tối thiểu 4.5:1
- `#8b949e` trên `#0d1117` ≈ 4.1:1 → **chưa đạt**
- `#c9d1d9` trên `#0d1117` ≈ 10.5:1 → **đạt tốt**

### Khuyến nghị palette nhất quán

```css
/* Text chính          */ color: #c9d1d9;   /* contrast 10.5:1 */
/* Text phụ (caption)  */ color: #8b949e;   /* dùng cho metadata, không phải nội dung */
/* Text nổi bật        */ color: #e6edf3;   /* tiêu đề, tên phim */
/* Accent đỏ           */ color: #e50914;   /* brand color */
/* Link / info         */ color: #58a6ff;   /* xanh nhạt */
/* Success             */ color: #3fb950;   /* xanh lá */
```

### Độ ưu tiên: 🟡 Trung bình – Ảnh hưởng aesthetics và accessibility

> [!NOTE]
> **✅ ĐÃ XỬ LÝ (v2 – 21/05/2026)**  
> - `.rbox` text: `#8b949e` → `#b8c0cc` (tương phản ~6:1, đạt WCAG AA)  
> - `.rbox strong`: `#c9d1d9` → `#e6edf3` (reviewer name nổi bật hơn)  
> - `.hero p`: `#8b949e` → `#c9d1d9` (mô tả hero section dễ đọc hơn)  
> - Alert banners: từ màu accent (`#58a6ff`/`#3fb950`) → `#c9d1d9` (body text nhất quán)  
> - Quick-search buttons: `#8b949e`→`#c9d1d9`, nền `#21262d`→`#1c2128`, border `#30363d`→`#444c56`  
> - Chip labels: `#8b949e` → `#c9d1d9` cho chip mặc định

---

## 3. Poster Phim Trong Kết Quả Gợi Ý

### Yêu cầu
Hiển thị poster phim bên cạnh card kết quả ở Tab 2 và Tab 3.

### Phân tích khả thi

#### Lựa chọn A – TMDb API ⭐ (Khuyến nghị)
- **Mô tả:** The Movie Database API, miễn phí cho mục đích phi thương mại
- **Endpoint:** `GET /3/search/movie?query={title}&api_key={key}`
- **Trả về:** URL poster dạng `https://image.tmdb.org/t/p/w300/{poster_path}`
- **Giới hạn:** 40 requests/10 giây (đủ dùng)
- **Cần:** Đăng ký API key miễn phí tại themoviedb.org
- **Cache:** Lưu `{title: poster_url}` vào `.json` để không gọi API liên tục
- **Khả thi:** ✅ Cao – thư viện `requests` có sẵn, code đơn giản ~20 dòng
- **Rủi ro:** Tên phim không khớp (đặc biệt phim có dấu hoặc tên nước ngoài)

#### Lựa chọn B – OMDb API
- Tương tự TMDb, miễn phí 1000 req/ngày
- Chất lượng poster thấp hơn, API phức tạp hơn
- **Không khuyến nghị** so với TMDb

#### Lựa chọn C – Placeholder thông minh
- Nếu không có API: generate placeholder theo màu hash từ tên phim
- Hiển thị icon 🎬 + tên phim viết tắt
- **Khả thi:** ✅ Cao, không cần API, nhưng trải nghiệm kém hơn nhiều

### Cách triển khai (TMDb)

```python
import requests, hashlib, os, json

TMDB_API_KEY = "YOUR_KEY_HERE"  # đặt trong st.secrets hoặc .env
POSTER_CACHE_PATH = "Outputs/poster_cache.json"

@st.cache_data(ttl=3600*24)  # cache 24 giờ
def fetch_poster_url(movie_title: str) -> str | None:
    """Tìm poster từ TMDb. Trả về URL hoặc None nếu không tìm thấy."""
    cache = _load_cache()
    if movie_title in cache:
        return cache[movie_title]
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"query": movie_title, "api_key": TMDB_API_KEY},
            timeout=3
        )
        results = r.json().get("results", [])
        if results and results[0].get("poster_path"):
            url = f"https://image.tmdb.org/t/p/w300{results[0]['poster_path']}"
            _save_cache(movie_title, url)
            return url
    except Exception:
        pass
    return None
```

**Card với poster (layout 2 cột):**
```python
col_poster, col_info = st.columns([1, 3])
with col_poster:
    poster = fetch_poster_url(title)
    if poster:
        st.image(poster, width=120)
    else:
        st.markdown("🎬", unsafe_allow_html=False)  # fallback
with col_info:
    render_movie_card(rank, title, final, cos_n, svd_n)
```

### Độ ưu tiên: 🟡 Trung bình – Cải thiện UX đáng kể, cần API key

> [!IMPORTANT]
> **✅ ĐÃ TRIỂN KHAI (v2.1 – 21/05/2026) – Không Cần API Key**  
> - Áp dụng chiến lược **Fallback**: `iTunes Search API` -> `Wikipedia REST API`  
> - Không cần đăng ký, không cần số điện thoại, dùng trực tiếp  
> - Cả hai API đều miễn phí và công khai, tự động cache 24h  
> - Layout 2 cột (poster | card+tabs) đã được kích hoạt  
> - Khi không tìm thấy: hiển thị placeholder 🎬 đẹp thay vì lỗi

---

## 4. Nhúng Trailer Video Trực Tiếp

### Yêu cầu
Hiển thị video trailer ngay trong app thay vì chỉ có nút link ngoài.

### Phân tích kỹ thuật

#### Vì sao hiện tại không nhúng được?
Streamlit dùng `st.markdown()` với iframe HTML bị chặn vì:
- YouTube có header `X-Frame-Options: SAMEORIGIN` với URL `youtube.com/watch`
- Tuy nhiên URL **embed** (`youtube.com/embed/{id}`) **cho phép nhúng** trong iframe

#### Lựa chọn A – `st.components.v1.html()` với YouTube Embed ⭐ (Khả thi nhất)
```python
import streamlit.components.v1 as components

def render_youtube_embed(query: str, height: int = 200):
    # Dùng YouTube embed search (không cần API key)
    encoded = urllib.parse.quote_plus(query + " official trailer")
    embed_url = f"https://www.youtube.com/embed?listType=search&list={encoded}"
    components.html(
        f'<iframe width="100%" height="{height}" src="{embed_url}" '
        f'frameborder="0" allowfullscreen></iframe>',
        height=height + 10
    )
```
- **Ưu điểm:** Không cần API key, hoạt động với Streamlit
- **Nhược điểm:** YouTube Search Embed không ổn định (YouTube có thể thay đổi policy)
- **Rủi ro:** Kết quả tìm kiếm tự động có thể không đúng phim
- **Khả thi:** ✅ Có thể hoạt động, nhưng **không đáng tin cậy về lâu dài**

#### Lựa chọn B – YouTube Data API v3 + `st.components.v1.html()`
- Tìm `video_id` chính xác bằng API: `GET /search?q={title} trailer&type=video`
- Embed bằng `youtube.com/embed/{video_id}`
- **Quota:** 10,000 đơn vị/ngày miễn phí, 1 search = 100 đơn vị → **100 lần tìm/ngày**
- **Khả thi:** ✅ Cao, nhưng quota rất hạn chế → cần cache `{title: video_id}`
- **Đề xuất nếu dùng:** Cache video_id sau lần tìm đầu, lưu vào file JSON

#### Lựa chọn C – Không nhúng, cải thiện nút link
- Thêm thumbnail preview (ảnh tĩnh) lấy từ `https://img.youtube.com/vi/{id}/hqdefault.jpg`
- Khi click thumbnail → mở YouTube tab mới
- **Khả thi:** ✅ Rất cao, không cần quota, UX tốt hơn nút text thô

### Đánh giá thực tế

> ⚠️ **Khuyến nghị trung thực:** YouTube embed search (`listType=search`) hoạt động nhưng
> YouTube có thể vô hiệu hóa tính năng này bất cứ lúc nào (đã từng xảy ra).
> Với phạm vi đồ án, **Lựa chọn C (thumbnail + link)** là cân bằng tốt nhất giữa
> tính ổn định và trải nghiệm người dùng.
> Nếu muốn video thật sự, cần YouTube API v3 với caching tích cực.

### Độ ưu tiên: 🟢 Thấp-Trung – Cải thiện nice-to-have, phức tạp và có rủi ro

---

## 5. Từ Khoá Nổi Bật Phong Phú Hơn (Tab 3)

### Vấn đề hiện tại

Các từ khoá hiện tại (`captain america`, `avengers`, `captain`, `chris evans`...) là
**top TF-IDF terms** – phản ánh tần suất trong văn bản nhưng có vấn đề:

1. **Quá nhiều tên riêng (actor/character):** `chris evans`, `evans`, `iron man` → không
   mô tả **nội dung** phim mà mô tả **đối tượng**
2. **Từ trùng nghĩa lặp:** `captain` và `captain america` cùng xuất hiện
3. **Không phân loại:** không biết đâu là thể loại, đâu là tone, đâu là setting
4. **Toàn unigram/bigram nhàm:** thiếu cụm từ ý nghĩa hơn

### Giải pháp theo mức độ phức tạp

#### Cấp độ 1 – Cải thiện ngay trong TF-IDF (không cần thư viện mới)
```python
# Lọc từ khoá sau khi lấy từ TF-IDF:
# 1. Bỏ các token là tên riêng rõ ràng (viết hoa hết)
# 2. Ưu tiên bigram/trigram hơn unigram
# 3. Loại trừ whitelist các từ quá chung: film, movie, story, scene...

GENERIC_TERMS = {"film", "movie", "story", "scene", "character", "plot", "time",
                 "watch", "great", "good", "best", "little", "just", "like", "make"}

def get_enriched_keywords(vec_matrix, vectorizer, n=12):
    feature_names = vectorizer.get_feature_names_out()
    avg_vec = np.asarray(vec_matrix.mean(axis=0)).flatten()
    
    # Tách unigram và n-gram
    ngram_indices = [i for i, f in enumerate(feature_names) if " " in f]
    unigram_indices = [i for i, f in enumerate(feature_names) if " " not in f]
    
    # Ưu tiên n-gram (top 60%), bổ sung unigram (top 40%)
    top_ngrams = sorted(ngram_indices, key=lambda i: avg_vec[i], reverse=True)
    top_unigrams = sorted(unigram_indices, key=lambda i: avg_vec[i], reverse=True)
    top_unigrams = [i for i in top_unigrams if feature_names[i] not in GENERIC_TERMS]
    
    n_ngram = int(n * 0.6)
    n_uni = n - n_ngram
    selected = top_ngrams[:n_ngram] + top_unigrams[:n_uni]
    return [feature_names[i] for i in selected]
```
- **Khả thi:** ✅ Rất cao, sửa 1 hàm trong `main_web.py`

#### Cấp độ 2 – KeyBERT (kết quả tốt hơn nhiều, cần thêm thư viện)
```bash
pip install keybert sentence-transformers  # ~500MB lần đầu download model
```
```python
from keybert import KeyBERT
kw_model = KeyBERT()  # dùng model nhỏ "all-MiniLM-L6-v2" mặc định

def get_keybert_keywords(text: str, n=10) -> list[str]:
    keywords = kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 3), top_n=n,
        diversity=0.5  # MMR diversity để tránh lặp
    )
    return [kw for kw, score in keywords]
```
- **Ưu điểm:** Từ khoá ngữ nghĩa, đa dạng hơn nhiều
- **Nhược điểm:** Chậm (~1-3s/lần gọi), cần download model lớn
- **Khả thi:** ✅ Trung bình – phù hợp nếu cache kết quả

#### Cấp độ 3 – Phân loại từ khoá theo nhóm
```python
# Phân loại keyword theo danh mục (rule-based)
GENRE_WORDS   = {"action", "comedy", "horror", "drama", "thriller", "romance", ...}
TONE_WORDS    = {"dark", "funny", "emotional", "inspiring", "suspenseful", ...}
SETTING_WORDS = {"space", "war", "school", "city", "future", "historical", ...}
```
- Hiển thị chips theo màu: 🎭 Thể loại | 🎨 Tone | 🌍 Bối cảnh
- **Khả thi:** ✅ Trung bình – cần xây dựng từ điển phân loại thủ công

### Độ ưu tiên: 🟡 Trung bình – Cải thiện đáng kể trải nghiệm Tab 3

> [!NOTE]
> **✅ ĐÃ XỬ LÝ CẤP 1 (v2 – 21/05/2026)**  
> - Thay `get_top_keywords` bằng `get_enriched_keywords` (lọc `_GENERIC_TERMS`, ưu tiên n-gram)  
> - Số lượng từ khoá: 10 → **12** để phong phú hơn  
> - Từ khoá chứa bigram/trigram (cụm từ có nghĩa) được ưu tiên hiển thị trước  
> **Cấp 2 (KeyBERT) và Cấp 3 (phân loại màu):** chưa triển khai, xem xét khi có thời gian

---

## 6. Sinh Văn Bản Giải Thích Kết Quả (Tab 2 & 3)

### Yêu cầu
"Tại sao các phim này được gợi ý? / Tại sao chúng có điểm chung?"

### Phân tích trung thực – Đây là bài toán khó

#### Cấp độ 1 – Template-based (Không cần AI/API) ✅ Khả thi ngay
Dùng từ khoá đã extract để điền vào mẫu câu:
```python
def generate_explanation_template(keywords: list[str], movie_titles: list[str]) -> str:
    kw_str = ", ".join(f'**{k}**' for k in keywords[:5])
    return (
        f"Các phim bạn chọn đều có chung điểm nhấn về {kw_str}. "
        f"Hệ thống đã phân tích {len(movie_titles)} phim để tìm ra {len(recommended)} "
        f"tác phẩm tương tự mà bạn chưa xem, dựa trên nội dung review của cộng đồng."
    )
```
- **Ưu điểm:** Không cần API, chạy ngay, ổn định
- **Nhược điểm:** Giải thích cứng nhắc, lặp pattern, không thực sự thông minh
- **Đánh giá:** Chấp nhận được ở mức đồ án, **không ấn tượng** trong thực tế

#### Cấp độ 2 – Local LLM qua Ollama (Không cần API key)
```bash
# Cài đặt Ollama (tool local LLM)
# Chạy model nhỏ: ollama run gemma2:2b hoặc phi3:mini
pip install ollama
```
```python
import ollama
def generate_explanation_llm(keywords, titles):
    prompt = f"""Các phim: {', '.join(titles[:3])}
    Từ khoá chung: {', '.join(keywords[:8])}
    Viết 2 câu giải thích ngắn gọn tại sao những phim này có điểm chung."""
    response = ollama.chat(model="gemma2:2b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
```
- **Ưu điểm:** Giải thích tự nhiên, thông minh, không cần internet
- **Nhược điểm:** Cần cài Ollama (~4GB), chậm 3-8s/lần, cần GPU để mượt
- **Khả thi:** ⚠️ Trung bình – phụ thuộc phần cứng máy chạy demo

#### Cấp độ 3 – Google Gemini API (miễn phí tier)
- Free tier: 15 req/phút, 1500 req/ngày
- Chất lượng cao nhất, cần API key
- **Khả thi:** ✅ Cao về mặt kỹ thuật, nhưng **vi phạm ràng buộc đồ án** "không dùng API ngoài"

### Đánh giá thực tế về giải thích kết quả

> ⚠️ **Nhận xét khắt khe:** Giải thích kiểu template-based thực chất **không thêm thông tin mới**
> so với việc hiển thị từ khoá. Người dùng thông minh sẽ thấy ngay đây là câu văn được ghép
> máy móc. Nếu không dùng LLM thực sự, tốt nhất là **hiển thị từ khoá một cách trực quan**
> (phân loại, có màu, có tooltip giải thích nguồn gốc từ khoá) thay vì sinh câu văn giả tạo.
> 
> **Khuyến nghị:** Chỉ thêm giải thích văn bản nếu dùng LLM thực sự. Nếu không, tập trung
> vào việc làm từ khoá trực quan và phong phú hơn (Mục 5, Cấp độ 1+3).

### Độ ưu tiên: 🟢 Thấp – Chỉ thêm nếu dùng LLM thực; template-based không đủ thuyết phục

---

## 7. Các Vấn Đề Giao Diện Khác Phát Hiện Từ Ảnh

### 7.1 Sidebar – Slider & Selectbox
- **Vấn đề:** Label "Số phim gợi ý" và "Hồ sơ người dùng (SVD)" màu trắng ổn, nhưng
  value display (`8`, `eydokia-21277`) màu đỏ của slider thumb không rõ trên nền tối
- **Fix:** Thêm CSS target `[data-testid="stSlider"]` để style nhất quán

### 7.2 Nút Quick Search (Tab 2)
- **Vấn đề:** 5 nút nằm trên 1 hàng, text màu `#8b949e` trên `#21262d` – tương phản thấp
- **Quan sát từ ảnh:** Nút trông như text box, thiếu visual cue là nút bấm
- **Fix:** Thêm hover effect rõ hơn, có thể thêm icon trước mỗi nút

### 7.3 Card Kết Quả – Thông Tin Quá Ít
- **Vấn đề:** Card chỉ có tên phim, 3 chip số, thanh progress bar
- **Thiếu:** Năm phát hành, thể loại, số review, rating trung bình
- **Fix:** Bổ sung metadata từ `reviews_db` hoặc TMDb API

### 7.4 Tab "Đánh giá cộng đồng" – Review Tiếng Anh Không Dịch
- **Vấn đề:** Toàn bộ review hiển thị bằng tiếng Anh, giao diện tiếng Việt → không nhất quán
- **Quan sát từ ảnh:** Review dài, khó đọc, không có tóm tắt
- **Fix ngắn hạn:** Hiển thị snippet 150 ký tự + toggle "Xem thêm"
- **Fix lý tưởng:** Dịch snippet đầu (1-2 câu) sang tiếng Việt bằng GoogleTranslator

### 7.5 Progress Bar Màu Không Trực Quan
- **Vấn đề:** Thanh đỏ-cam (`#e50914 → #ff6b35`) không phân biệt được 60% vs 90%
- **Cải thiện:** Thêm gradient màu theo điểm: thấp = cam, trung bình = vàng, cao = xanh lá
- Hoặc: dùng emoji rating `🔥🔥🔥` thay thế để trực quan hơn

### 7.6 Tab "Trải Nghiệm Xem Phim" – Sau Khi Có Kết Quả
- **Vấn đề:** Section "Phim đã chọn" (tags vàng) biến mất sau khi hiển thị kết quả
  → người dùng quên mình đã chọn phim gì
- **Fix:** Giữ tags đã chọn luôn hiển thị ở trên cùng dạng sticky/collapsed section

---

## 8. Kế Hoạch Triển Khai Ưu Tiên

### 🔴 Ưu tiên 1 – Sửa ngay ✅ HOÀN THÀNH
1. ~~**Fix SVM N/A**~~ *(bỏ qua theo yêu cầu – cần retrain để có file đúng tên)*
2. ✅ **Fix màu chữ**: Tăng tương phản `.rbox`, quick search buttons, alert banners
3. ✅ **Cải thiện từ khoá (Cấp 1)**: `get_enriched_keywords` lọc generic terms, ưu tiên n-gram
4. ✅ **Snippet review**: 150 ký tự + expander "Xem đầy đủ review"
5. ✅ **Poster layout**: Đã đổi sang **iTunes API + Wikipedia API** (không cần đăng ký)

### 🟡 Ưu tiên 2 – Phát triển (cần thời gian)
5. **Poster phim (TMDb API)**: Layout 2 cột, cache JSON
6. **Thumbnail trailer**: Ảnh tĩnh YouTube → click mở tab mới
7. **Metadata card**: Thêm năm, genre từ TMDb
8. **Phân loại từ khoá (Cấp 3)**: Chips có màu theo thể loại/tone

### 🟢 Ưu tiên 3 – Nâng cao (tùy điều kiện)
9. **Nhúng video trailer**: YouTube embed search hoặc Data API v3 + cache
10. **Giải thích kết quả**: Chỉ khi dùng local LLM hoặc Gemini API
11. **Dịch review snippet**: GoogleTranslator trên 1-2 câu đầu

---

## 9. Đánh Giá Tổng Thể Theo Tiêu Chí Đồ Án

| Tiêu chí | Hiện tại | Sau Ưu tiên 1 | Sau Ưu tiên 2 |
|---|---|---|---|
| Chức năng đúng | ✅ | ✅ | ✅ |
| Giao diện trực quan | ⚠️ 65/100 | ✅ 80/100 | ✅ 90/100 |
| Thông tin đầy đủ | ❌ 50/100 | ⚠️ 65/100 | ✅ 85/100 |
| Khả năng giải thích | ❌ 30/100 | ⚠️ 55/100 | ✅ 75/100 |
| Tính nhất quán UI | ⚠️ 70/100 | ✅ 85/100 | ✅ 90/100 |

> **Nhận xét thẳng thắn:** Hệ thống hiện tại đủ điều kiện demo nhưng chưa đạt
> mức "hoàn thiện". Khoảng cách lớn nhất là **thiếu thông tin thị giác** (poster, trailer)
> và **giải thích kết quả** – đây là 2 thứ người dùng thực tế luôn muốn biết đầu tiên.
> Ưu tiên 1 có thể hoàn thành trong 2-3 giờ. Ưu tiên 2 cần thêm 1-2 ngày và API key.

---

*CineMatch UI Improvement Plan – 21/05/2026*
