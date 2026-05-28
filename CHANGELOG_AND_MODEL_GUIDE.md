# 📄 Tài Liệu Cập Nhật – Hệ Thống Gợi Ý Phim Hybrid

> **Dự án:** KPDL_DoAn · SVM + TF-IDF + Cosine Similarity + SVD  
> **Cập nhật:** 14/05/2026 (v2) | 20/05/2026 (v3) | 20/05/2026 (v4) | 21/05/2026 (v5 – UI) | 21/05/2026 (v6 – Algorithm Improvements)

---

## 1. Kết Quả Huấn Luyện Mô Hình (Phiên bản mới nhất)

```
Accuracy              : 0.9215  (92.15%)
Precision (Weighted)  : 0.9204
Recall    (Weighted)  : 0.9215
F1-Score  (Weighted)  : 0.9202
Precision (Macro)     : 0.9095
Recall    (Macro)     : 0.8837
F1-Score  (Macro)     : 0.8955
ROC-AUC               : 0.9672

Confusion Matrix:
[[ 5003  1220]   ← Negative: 5003 đúng, 1220 nhầm thành Positive
 [  639 16816]]  ← Positive: 639 nhầm thành Negative, 16816 đúng

CV F1-Weighted (3-fold): 0.9213 ± 0.0013
```

**Nhận xét tổng quan:**  
Mô hình đạt chất lượng **rất tốt**. ROC-AUC 96.7% và CV ổn định ±0.13% cho thấy mô hình học thực sự chứ không overfit. Lớp Negative có Recall thấp hơn (0.80) do dữ liệu gốc vẫn còn lệch. **Phiên bản v2** áp dụng cân bằng 1.5:1 để cải thiện Recall lớp Negative.

---

## 2. Giải Thích Chi Tiết Từng Chỉ Số

### 2.1 Accuracy – 92.15%

```
Accuracy = (TP + TN) / Tổng = (16816 + 5003) / 23678 = 92.15%
```

**Ý nghĩa:** Tỷ lệ dự đoán đúng trên toàn bộ tập test.  
**Lưu ý:** Bị ảnh hưởng bởi mất cân bằng. Lớp Positive chiếm 73.7% → model đoán nhiều Positive hơn → Accuracy cao nhưng Recall Negative bị kéo xuống.

---

### 2.2 Precision – 0.9204 (Weighted)

```
Precision = TP / (TP + FP)
Negative: 5003 / (5003 + 639)  = 0.887
Positive: 16816 / (16816 + 1220) = 0.932
```

**Ý nghĩa:** Trong số phim được dự đoán là Positive, 93.2% đúng thực sự Positive → ít gợi ý phim dở.

---

### 2.3 Recall – 0.8837 (Macro)

```
Recall = TP / (TP + FN)
Negative: 5003 / (5003 + 1220) = 0.804  ← điểm yếu
Positive: 16816 / (16816 + 639) = 0.963
```

**Ý nghĩa:** Mô hình bỏ sót 19.6% phim thực sự dở (đoán nhầm là hay). Đây là điểm cần cải thiện.

---

### 2.4 F1-Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Negative : 0.887 × 0.804 / (0.887 + 0.804) × 2 = 0.843
Positive : 0.932 × 0.963 / (0.932 + 0.963) × 2 = 0.947
Macro F1 : (0.843 + 0.947) / 2 = 0.895
Weighted F1 : trọng số theo số mẫu = 0.920
```

**Ý nghĩa:** Chỉ số tổng hợp quan trọng nhất với dữ liệu mất cân bằng. Macro F1 0.895 cho thấy cả hai lớp đều được học tốt, nhưng Negative vẫn yếu hơn.

---

### 2.5 ROC-AUC – 0.9672

| Giá trị | Diễn giải |
|---|---|
| 0.50 | Ngẫu nhiên (vô dụng) |
| 0.70–0.80 | Trung bình |
| 0.80–0.90 | Tốt |
| **0.9672** | **Xuất sắc** ✅ |

**Ý nghĩa:** Mô hình phân biệt Positive/Negative với xác suất 96.7% chính xác khi chọn ngẫu nhiên 1 mẫu từ mỗi lớp.

---

### 2.6 Confusion Matrix

```
              Dự đoán NEG    Dự đoán POS
Thực tế NEG  [  5003   1220  ]
Thực tế POS  [   639  16816  ]
```

| | Tên | Ý nghĩa | Ảnh hưởng |
|---|---|---|---|
| **TP = 16816** | True Positive | Phim hay → đoán đúng ✅ | Gợi ý đúng |
| **TN = 5003**  | True Negative | Phim dở → đoán đúng ✅ | Loại đúng |
| **FP = 1220**  | False Positive | Phim dở → đoán nhầm là hay ❌ | Gợi ý sai |
| **FN = 639**   | False Negative | Phim hay → đoán nhầm là dở ❌ | Bỏ lỡ phim tốt |

---

### 2.7 Cross-Validation – 0.9213 ± 0.0013

```
3-fold Stratified CV:
  Fold 1: F1 = 0.9202
  Fold 2: F1 = 0.9218
  Fold 3: F1 = 0.9219
  Mean: 0.9213 | Std: 0.0013
```

**Ý nghĩa:**  
- Mean cao (0.921) → Model học tốt trên dữ liệu huấn luyện  
- Std thấp (0.001) → Kết quả **ổn định**, **không overfit**, có thể triển khai thực tế

---

## 3. Giải Thích: Tại Sao Chọn User Khác Nhau Cho Kết Quả Giống Nhau?

### Nguyên nhân (phiên bản cũ)

```python
# Code cũ – SAI
svd_bonus = max(0, dot * 0.15)
final_score = cosine_score + svd_bonus
```

**Vấn đề:**
- `cosine_score` nằm trong khoảng [0, 0.3]
- `dot product SVD` nằm trong khoảng [-50, 50] → sau `× 0.15` còn [-7.5, 7.5]
- `max(0, ...)` cắt bỏ hầu hết giá trị SVD
- Kết quả: SVD bonus ≈ 0.001 → quá nhỏ, không thay đổi thứ hạng

### Giải pháp (phiên bản mới)

```python
# Code mới – ĐÚNG
cosine_norm = normalize_01(cosine_raw)    # [0, 1]
svd_norm    = normalize_01(svd_scores)    # [0, 1]

# Kết hợp theo trọng số cân bằng
final = 0.60 * cosine_norm + 0.40 * svd_norm
```

**Kết quả:** Khi chọn user khác nhau:
- Vector SVD `u_vec` khác nhau → `dot(u_vec, item_vec)` khác nhau
- Sau normalize về [0,1], SVD đóng góp **40% điểm số**
- Thứ hạng phim thay đổi rõ rệt giữa các người dùng

**Lưu ý về giới hạn:** SVD chỉ giải thích **20.5% phương sai** (100 components) do ma trận user-item rất thưa (sparse). Đây là thách thức chung của Collaborative Filtering với dữ liệu review.

---

## 4. Thay Đổi `main_web.py` (Cập nhật mới nhất)

| Vấn đề / Yêu cầu | Giải pháp |
|---|---|
| SVD không ảnh hưởng thứ hạng | Normalize cả hai về [0,1], kết hợp 60% Cosine + 40% SVD |
| Không xem trailer trực tiếp | Bỏ iframe (YouTube chặn X-Frame), dùng `st.link_button` |
| Màu chữ tối, khó đọc trên nền tối | Thêm CSS toàn cục: `color:#c9d1d9`, sidebar label `color:#c9d1d9` |
| Slider ngưỡng tương đồng thừa | Đã xoá, lấy top-100 ứng viên tự động |
| Tab Đánh Giá / Về Hệ Thống thừa | Gộp vào sidebar expander, giữ 3 tab chính |
| Chưa có tính năng theo phim đã xem | Thêm tab **🎭 Trải Nghiệm Xem Phim** |

### Cấu trúc 3 Tab mới

```
🏠 Trang Chủ         →  Tổng quan, metrics hệ thống, kiến trúc pipeline
🔍 Gợi Ý Theo Từ Khoá →  Tìm kiếm text, quick search, kết quả breakdown
🎭 Trải Nghiệm Xem Phim → Chọn phim đã xem → phân tích điểm chung → gợi ý
```

---

## 4b. Tính Năng Mới: Tab "Trải Nghiệm Xem Phim"

### Ý tưởng
Thay vì nhập từ khoá, người dùng **chọn các phim đã xem và thích** (multiselect tối đa 10 phim). Hệ thống tìm điểm chung giữa chúng và gợi ý phim tương tự chưa xem.

### Cơ chế kỹ thuật (`recommend_by_movies`)

```python
# 1. Lấy TF-IDF vector của từng phim đã chọn
sel_matrix = content_matrix[idxs]   # shape: (n_selected, n_features)

# 2. Trung bình vector → "vector gu" của người dùng
avg_vec = sel_matrix.mean(axis=0)   # shape: (1, n_features)

# 3. Cosine similarity với toàn bộ phim
cos_raw = cosine_similarity(avg_vec, content_matrix).flatten()

# 4. Normalize + kết hợp SVD (nếu có user)
final = 0.6 * cos_norm + 0.4 * svd_norm
```

### Trích xuất điểm chung (từ khoá)

```python
def get_top_keywords(sel_matrix, vectorizer, n=10):
    avg = np.asarray(sel_matrix.mean(axis=0)).flatten()
    top_idx = avg.argsort()[-n:][::-1]
    return [feature_names[i] for i in top_idx]
```

Hiển thị các từ khoá nổi bật như: `action`, `thriller`, `suspense`, `plot twist`... → Giúp người dùng hiểu tại sao các phim đó được gợi ý.

### Ưu điểm
- Không cần nhập từ khoá – trực quan hơn cho người dùng thông thường
- Kết hợp sở thích từ nhiều phim → profile gu chính xác hơn
- Vẫn trong phạm vi TF-IDF + Cosine + SVD (không cần API ngoài)

---

## 5. Thay Đổi `train_SVM_cosine_SVD.py`

| Hạng mục | Thay đổi |
|---|---|
| **Cân bằng dữ liệu (v2)** | **Undersample POS để pos:neg ≤ 1.5:1 (từ 3:1 cũ); RandomOverSampler nếu vẫn lệch** |
| Cụm từ đặc biệt | Thêm POSITIVE_PHRASES / NEGATIVE_PHRASES trước TF-IDF |
| TF-IDF | `max_features=30000`, `ngram=(1,3)`, `max_df=0.90`, `min_df=3` |
| Model | `LinearSVC` + `CalibratedClassifierCV` (xuất `predict_proba`) |
| SVD | `n_components=100`, `n_iter=10` |
| Profile phim | Chỉ giữ phim có ≥ 3 review tích cực |
| Đánh giá | Thêm ROC-AUC, Macro F1, Cross-Validation |

### Chi Tiết Cân Bằng Dữ Liệu 1.5:1 (v2 – 14/05/2026)

**Vấn đề:** Sau khi lọc phim 1980–2021 từ dữ liệu IMDb, số lượng review tích cực (Positive) lớn hơn nhiều so với tiêu cực (Negative), tỷ lệ thực tế có thể lên tới 3:1 hoặc cao hơn → mô hình thiên lệch về lớp Positive.

**Giải pháp – 2 tầng bảo vệ:**

```
Tầng 1 – Undersample tại bước load dữ liệu:
  max_pos = int(n_neg × 1.5)
  Nếu n_pos > max_pos → lấy mẫu ngẫu nhiên df_pos xuống max_pos
  → Log: [Undersample POS] Giảm Positive xuống {max_pos} (tỷ lệ mục tiêu 1.5:1)

Tầng 2 – RandomOverSampler sau TF-IDF (nếu vẫn lệch):
  Sau train_test_split, kiểm tra tỷ lệ thực tế trong X_train
  Nếu pos:neg > 1.5 → dùng RandomOverSampler(sampling_strategy=1/1.5)
  → Duplicate ngẫu nhiên các mẫu Negative cho đến khi đạt tỷ lệ
  → Log: [Oversample NEG] Phân phối sau: {dict}
```

> **Lý do chọn RandomOverSampler thay SMOTE:**  
> SMOTE tạo mẫu tổng hợp bằng nội suy trong không gian đặc trưng. Với ma trận TF-IDF sparse (30,000 chiều), việc nội suy tạo ra các vector không có ý nghĩa ngữ nghĩa. RandomOverSampler sao chép mẫu gốc nên an toàn hơn với text data.

**Kỳ vọng cải thiện:**

| Chỉ số | Trước (3:1) | Kỳ vọng (1.5:1) |
|---|---|---|
| Recall Negative | ~0.80 | ≥ 0.85 |
| Macro F1 | 0.895 | ≥ 0.91 |
| Precision Negative | 0.887 | Ổn định hoặc tăng nhẹ |

---

## 6. Phạm Vi Đồ Án

Hệ thống chỉ sử dụng:
- ✅ **SVM (LinearSVC)** – Phân loại cảm xúc review
- ✅ **TF-IDF** – Vector hóa văn bản
- ✅ **Cosine Similarity** – Đo độ tương đồng nội dung
- ✅ **Truncated SVD** – Lọc cộng tác (Collaborative Filtering)
- ✅ **deep-translator** – Dịch truy vấn tiếng Việt → tiếng Anh

Không sử dụng: Neural Network, BERT, LLM, YouTube API, TMDb API

---

## 7. Hướng Dẫn Chạy

```bash
# Cài thư viện
pip install streamlit joblib scikit-learn deep-translator pandas numpy imbalanced-learn

# Huấn luyện mô hình (cần part-03.json và title.basics.tsv)
python train_SVM_cosine_SVD.py

# Chạy web app
streamlit run main_web.py
```

---

## 8. Liên Hệ Với Nội Dung Đồ Án

| Chương đồ án | Nội dung | File liên quan |
|---|---|---|
| Chương 2.3 | TF-IDF, Cosine Similarity | `train_SVM_cosine_SVD.py` → vectorizer |
| Chương 2.4 | SVM, Hyperplane, Margin | `train_SVM_cosine_SVD.py` → LinearSVC |
| Chương 2.5 | Collaborative Filtering, SVD | `train_SVM_cosine_SVD.py` → TruncatedSVD |
| Chương 3.2 | Lấy mẫu cân bằng | `run_training_pipeline()` |
| Chương 4.2 | Accuracy, Precision, Recall, F1 | `evaluate_and_save()` |
| Chương 4.3 | Kịch bản người dùng khác nhau | `main_web.py` → tab Gợi Ý + tab Trải Nghiệm |
| Chương 4.4 | Giao diện Streamlit | `main_web.py` (3 tabs) |
| Chương 5.2 | Cold-start, Sparse matrix | Sidebar expander + tab Trang Chủ |

---

*KPDL_DoAn 2026 – Cập nhật tự động*

---

## 9. Nhật Ký Sửa Đổi v3 (20/05/2026) – Sửa Lỗi Đọc File JSON

### Mô tả lỗi

Khi chạy `train_SVM_cosine_SVD.py`, pipeline báo:

```
Gộp thành công! Tổng số lượng review thô ban đầu: 3 dòng.
Lỗi hệ thống: ['rating', 'review_detail']
KeyError: ['rating', 'review_detail']
```

Thay vì hàng triệu dòng (3 file ~1GB/file), chỉ đọc được 3 dòng (1 dòng/file).

### Nguyên Nhân Gốc Rễ

| Vấn đề | Chi tiết |
|---|---|
| **Định dạng file** | Các file `part-*.json` lưu dạng **Standard JSON Array** (`[{...}, {...}, ...]`), không phải JSON Lines |
| **`pd.read_json(lines=True)` thất bại** | Đúng — file không phải JSON Lines, nên `ValueError` được ném ra |
| **Fallback `pd.read_json(file)` bị sai** | `pd.read_json` mặc định dùng `orient='columns'` thay vì `orient='records'` → đọc transpose, mỗi file chỉ trả về 1 dòng |
| **Hậu quả** | `len(df) = 3` (3 file → 3 dòng), không có cột `rating` / `review_detail` → `KeyError` |

### Giải Pháp (v3)

**File sửa đổi:** `train_SVM_cosine_SVD.py`

#### Thay đổi 1 – Thêm `import json`
```python
# Dòng 8
import json
```

#### Thay đổi 2 – Thêm hàm helper `_load_json_file()` (trước `run_training_pipeline`)
```python
def _load_json_file(filepath):
    """
    Đọc file JSON lớn an toàn. Hỗ trợ 2 định dạng:
      1. JSON Lines: {"k": v}\n{"k": v}\n...
      2. Standard JSON Array: [{...}, {...}, ...]
    """
    # Bước 1: Thử JSON Lines
    try:
        df = pd.read_json(filepath, lines=True, encoding='utf-8')
        if df.shape[0] > 1:
            return df
    except Exception:
        pass

    # Bước 2: Standard JSON Array dùng json.load → pd.DataFrame
    # (tránh bug orientation của pd.read_json mặc định)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame.from_dict(data, orient='index')
    except Exception as e:
        raise ValueError(f"Không thể đọc file '{filepath}': {e}")
```

#### Thay đổi 3 – Thay vòng lặp đọc file trong `run_training_pipeline`
```python
# TRƯỚC (sai – fallback pd.read_json không có orient='records')
try:
    temp_df = pd.read_json(file, lines=True)
except ValueError:
    temp_df = pd.read_json(file)   # ← BUG: đọc sai orientation

# SAU (đúng – dùng _load_json_file)
temp_df = _load_json_file(file)
print(f"     -> Đọc được {len(temp_df):,} dòng, cột: {list(temp_df.columns)}")
```

### Tại Sao Không Dùng `orient='records'` Trực Tiếp?

`pd.read_json(file, orient='records')` với file 1GB sẽ vẫn load toàn bộ file vào RAM cùng lúc. Cách dùng `json.load()` và tạo DataFrame cũng làm vậy nhưng **chính xác hơn** vì tránh hoàn toàn logic orientation phức tạp của pandas. Với RAM đủ lớn (≥ 16GB) thì không có vấn đề; nếu cần tối ưu RAM có thể nâng cấp lên dùng `ijson` streaming.

### Phạm Vi Ảnh Hưởng

| Phần | Ảnh hưởng |
|---|---|
| `_load_json_file()` | Hàm mới, không ảnh hưởng code cũ |
| `run_training_pipeline()` | Chỉ thay đổi 5 dòng trong vòng lặp đọc file |
| Toàn bộ pipeline sau đó | **Không thay đổi** |
| `main_web.py` | **Không thay đổi** |
| `PhanTichCamXuc_IMDB.py` | **Không thay đổi** |

---

## 10. Nhật Ký Clean Code v4 (20/05/2026)

### Mục tiêu
Rà soát và làm sạch cả hai file theo nguyên tắc clean code:
- **DRY** (Don't Repeat Yourself) – không lặp logic
- **Single Responsibility** – mỗi hàm một nhiệm vụ rõ ràng
- **Naming** – tên biến/hàm rõ nghĩa, nhất quán
- **Consistency** – cùng 1 quy tắc dùng suốt toàn bộ codebase

---

### File: `train_SVM_cosine_SVD.py`

#### Vấn đề 1 – Mâu thuẫn `sample_size` (BUG LOGIC)
```python
# TRƯỚC – default ≠ lời gọi → gây nhầm lẫn: "script đang dùng giá trị nào?"
def run_training_pipeline(..., sample_size=500000, ...):
    ...
run_training_pipeline(..., sample_size=150000, ...)  # ← override ngầm

# SAU – 1 hằng số duy nhất tại đầu file, dùng làm default
SAMPLE_SIZE = 150_000                               # ← nguồn sự thật duy nhất
def run_training_pipeline(..., sample_size=SAMPLE_SIZE, ...):
    ...
run_training_pipeline(folder_path=FOLDER_PATH, meta_file_path=META_FILE_PATH)
# Không cần truyền lại tham số đã có default đúng
```
**Tất cả tham số cấu hình (`SAMPLE_SIZE`, `RATING_POS_THRESHOLD`, `FILTER_YEAR`...) đều tập trung tại phần "CẤU HÌNH CHUNG" đầu file.**

#### Vấn đề 2 – Import thừa
```python
# TRƯỚC
from sklearn.metrics import (..., average_precision_score)  # không dùng ở đâu

# SAU – bỏ import này
```

#### Vấn đề 3 – `dual=True` deprecated
```python
# TRƯỚC
LinearSVC(dual=True, ...)   # FutureWarning trong sklearn mới

# SAU
LinearSVC(dual="auto", ...)  # tự chọn tối ưu theo n_samples vs n_features
```

#### Vấn đề 4 – Hàm nội tuyến ẩn `_normalize_title`
```python
# TRƯỚC – hàm super_clean_title định nghĩa lồng trong filter_movies_by_year
def filter_movies_by_year(...):
    def super_clean_title(title): ...

# SAU – đổi tên rõ nghĩa, vẫn nội tuyến (scope hợp lý)
def _normalize_title(title: str) -> str: ...
```

#### Vấn đề 5 – Raise đúng exception type
```python
# TRƯỚC
raise ValueError("Không tim thấy file...")

# SAU – dùng FileNotFoundError đúng ngữ nghĩa
raise FileNotFoundError("Không tìm thấy file...")
```

#### Vấn đề 6 – Logic gán nhãn dạng lambda khó đọc
```python
# TRƯỚC
df['label'] = df['rating'].apply(
    lambda x: 1 if x >= pos else (0 if x <= neg else -1)
)

# SAU – hàm nội tuyến có tên
def _assign_label(r):
    if r >= rating_pos_threshold: return 1
    if r <= rating_neg_threshold: return 0
    return -1
df_balanced['label'] = df_balanced['rating'].apply(_assign_label)
```

#### Vấn đề 7 – Comment sai chính tả
```python
# TRƯỚC
print(f"\n[!]       file IMDb metadata từ {meta_file_path}...")  # thiếu "Đang tải"

# SAU
print(f"\n[!] Đang tải file IMDb metadata từ: {meta_file_path}")
```

---

### File: `main_web.py`

#### Vấn đề 1 – `import scipy.sparse` trong thân hàm
```python
# TRƯỚC – import nằm TRONG hàm recommend_by_movies
def recommend_by_movies(...):
    import scipy.sparse as sp   # ← vi phạm PEP8, khó phát hiện dependency
    ...

# SAU – xóa luôn (sp không được dùng sau khi refactor, đã dùng numpy thay thế)
```

#### Vấn đề 2 – Logic SVD lặp lại ở 2 hàm
```python
# TRƯỚC – copy-paste y hệt trong recommend_by_query VÀ recommend_by_movies
is_pers = user_key != "🎭 Khách" and user_key in svd_data['user_map']
if is_pers:
    u_vec = svd_data['user_factors'][svd_data['user_map'][user_key]]
    for i, m in enumerate(cands['movie']):
        if m in svd_data['movie_map']:
            svd_scores[i] = np.dot(u_vec, svd_data['item_factors'][...])

# SAU – tách thành 3 helper functions
def _is_personalized(user_key): ...
def _compute_svd_scores(candidate_movies, user_key): ...
def _blend_scores(cos_norm, svd_norm, is_pers): ...
```

#### Vấn đề 3 – Tên hàm quá ngắn, không rõ nghĩa
```python
# TRƯỚC
def yt_url(t): ...      # không rõ là gì
def imdb_url(t): ...    # mơ hồ
def stars(r): ...       # quá ngắn

# SAU
def build_youtube_url(title: str): ...
def build_imdb_url(title: str): ...
def rating_to_stars(rating): ...
```

#### Vấn đề 4 – Accuracy hardcode trong UI
```python
# TRƯỚC
"92.15%"  # hardcode – mỗi lần retrain phải sửa tay

# SAU – đọc động từ evaluation_metrics.txt
if os.path.exists(_metrics_path):
    # parse dòng có "Accuracy" và "%"
    _svm_accuracy = ...
```

#### Vấn đề 5 – Code render lặp ở 2 tab
```python
# TRƯỚC – vòng lặp render card + tabs lặp lại ở tab Search VÀ tab Experience
for rank, (_, row) in enumerate(results.iterrows(), 1):
    render_movie_card(...)
    r_tab, l_tab = st.tabs([...])
    with r_tab: render_reviews_tab(row['movie'])
    with l_tab: render_links_tab(row['movie'])
    st.markdown("---")

# SAU – gộp vào 1 hàm duy nhất
def render_result_list(results: pd.DataFrame): ...
```

#### Vấn đề 6 – Spacer HTML thô
```python
# TRƯỚC
st.markdown("<br>", unsafe_allow_html=True)   # không có nghĩa, magic number

# SAU
st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)  # rõ ràng hơn
```

#### Vấn đề 7 – Kiểm tra file trước khi tải model
```python
# TRƯỚC – try/except chung chung, thông báo lỗi không rõ file nào thiếu
try:
    return (joblib.load(...), ...)
except Exception as e:
    st.error(f"Không tải được mô hình: {e}")

# SAU – kiểm tra từng file trước, báo rõ file nào thiếu
missing = [f for f in required_files if not os.path.exists(...)]
if missing:
    st.error(f"Thiếu file: {missing}")
    st.stop()
```

---

### Tổng Kết Thay Đổi

| Hạng mục | `train_SVM_cosine_SVD.py` | `main_web.py` |
|---|---|---|
| Hàm/biến thêm mới | `SAMPLE_SIZE`, `RATING_*`, `_assign_label` | `_is_personalized`, `_compute_svd_scores`, `_blend_scores`, `render_result_list` |
| Hàm/biến đổi tên | `super_clean_title` → `_normalize_title` | `yt_url` → `build_youtube_url`, `stars` → `rating_to_stars`, v.v. |
| Hàm/biến xóa | `average_precision_score` (import thừa) | `import scipy.sparse` (trong hàm), logic SVD lặp |
| Chức năng thay đổi | **Không** | **Không** |
| Kết quả train | **Không thay đổi** | N/A |

---

## 11. Nhật Ký UI Improvements v5 (21/05/2026)

> **Phạm vi:** Chỉ `main_web.py` — không ảnh hưởng mô hình hay pipeline training

### Mục tiêu

Cải thiện trải nghiệm người dùng theo 4 nhóm: tương phản màu sắc, từ khoá kết quả, hiển thị review,
và cấu trúc layout poster (sẵn sàng chờ API key).

---

### Thay đổi 1 – Sửa Màu Chữ (WCAG AA Compliance)

**Vấn đề:** Nhiều yếu tố UI dùng màu `#8b949e` cho text nội dung trên nền `#0d1117`
— tương phản ~4.1:1, dưới ngưỡng WCAG AA (4.5:1).

| Phần tử | Trước | Sau | Lý do |
|---|---|---|---|
| `.rbox` text (nội dung review) | `#8b949e` | `#b8c0cc` | Đạt WCAG AA (~6:1) |
| `.rbox strong` (tên reviewer) | `#c9d1d9` | `#e6edf3` | Nổi bật hơn |
| `.rbox em` (rating) | mặc định | `#8b949e`, không in nghiêng | Nhất quán |
| `.hero p` (mô tả trang chủ) | `#8b949e` | `#c9d1d9` | Dễ đọc hơn |
| `.alert-i`, `.alert-g` text | `#58a6ff`/`#3fb950` | `#c9d1d9` | Body text, không nên dùng accent color |
| `.chip` mặc định | `#8b949e` | `#c9d1d9` | Chip label cần đọc được |
| Quick-search buttons | `color:#8b949e` / `bg:#21262d` | `color:#c9d1d9` / `bg:#1c2128` | Tương phản rõ hơn |
| Keyword tags (`.theme-tag`) | `#d2a520` | `#e6c94a` | Sáng hơn trên nền tối |

Thêm 2 class CSS mới: `.theme-tag-genre` (xanh dương) và `.theme-tag-tone` (xanh lá)
— sẵn sàng cho phân loại keyword theo nhóm (Nhưu tiên 2 trong UI Plan).

---

### Thay đổi 2 – Từ Khoá Phong Phú Hơn (`get_enriched_keywords`)

**Thay thế:** `get_top_keywords(n=10)` → `get_enriched_keywords(n=12)`

**Thuật toán mới:**
```
1. Tính vector TF-IDF trung bình của các phim đã chọn (giống cũ)
2. Phân tách feature names thành 2 nhóm:
   │
   ├─ N-gram (chứa khoảng trắng): bigram, trigram
   └─ Unigram (một từ) cưa lọc _GENERIC_TERMS
3. Xếp hạng từng nhóm theo điểm TF-IDF
4. Lấy top-60% từ n-gram + top-40% từ unigram
5. Sắp xếp lại theo điểm giảm dần
```

**`_GENERIC_TERMS` (ồ từ bị loại):**
```python
{"film", "movie", "story", "scene", "character", "plot", "time",
 "watch", "great", "good", "best", "little", "just", "like", "make",
 "really", "made", "one", "also", "even", "first", "well", "much",
 "think", "many", "people", "way", "year", "show", "series", "part",
 "get", "see", "know", "going", "come", "give", "take", "would",
 "could", "never", "still", "another", "pretty", "quite", "better"}
```

**Kết quả mong đợi:** Thay vì đưa ra `["captain", "america", "good", "great", "film"]`,
năng đưa ra `["captain america", "super hero", "avengers", "action packed", "marvel"]` —
ý nghĩa hơn nhiều.

---

### Thay đổi 3 – Snippet Review + Expander "Xem Đầy Đủ"

**Trước:** Hiển thị 320 ký tự — vẫn quá dài, làm tắt cả 3 review chồng lấp nhau.

**Sau:** Hằng số `SNIPPET_LEN = 150` ký tự, thêm nút expander nếu review dài hơn.

```
Hiển thị mặc định (150 ký tự):
  reviewer ★★★★★ (10/10)
  The film was absolutely incredible in every...

  [▼ Xem đầy đủ review]  ← expander chỉ xuất hiện nếu review > 150 ký tự
    reviewer ★★★★★ (10/10)
    The film was absolutely incredible in every aspect...
    [toàn bộ nội dung, border xanh phân biệt]
```

Spoiler review: vẫn ẩn toàn bộ trong expander cảnh báo, nay hiển thị dưới dạng `.rbox`
tương thích thay vì `st.write` thô.

---

### Thay đổi 4 – Layout Poster Phim (iTunes/Wiki Fallback)

**Cấu trúc mới của `render_result_list()`:**
```
[Col 1/5 – poster]  [Col 4/5 – nội dung                        ]
+------------------+  +------------------------------------------+
|  [Poster ảnh]   |  |  #1 🎬 Tên phìm                           |
|  hoặc           |  |  [Cosine] [SVD] [Tổng hợp]              |
|  [🎬 placeholder]|  |  [███████████████████████████████]       |
|                  |  |  [📝 Đánh giá] [▶️ Trailer]              |
+------------------+  +------------------------------------------+
```

- **Chiến lược API không cần đăng ký:** Sử dụng `iTunes Search API` làm ưu tiên 1 (ảnh sắc nét, ổn định), dự phòng `Wikipedia REST API` làm ưu tiên 2.
- **Cache tĩnh:** Ảnh được lưu cache trong phiên bản 24 giờ với `@st.cache_data`.
- **Không tìm thấy:** Hiển thị `<div class="poster-ph">🎬</div>` — dark card có icon,
  giao diện vẫn gọn gàng.
- **Lợi ích lớn nhất:** Người dùng (và ban giám khảo) không cần đăng ký hay khai báo số điện thoại để cấu hình API.

---

### Thay đổi 5 – Hằng Số và Cấu Hình Tập Trung

Thêm vào khối "HẰNG SỐ" đầu file:

```python
SNIPPET_LEN  = 150       # Số ký tự hiển thị trực tiếp trong review box
```

---

### Tổng Kết Thay Đổi v5

| Thay đổi | File | Dòng mã | Ảnh hưởng chức năng |
|---|---|---|---|
| Fix màu 9 phần tử CSS | `main_web.py` | CSS block | Không |
| `_GENERIC_TERMS` + `get_enriched_keywords` | `main_web.py` | ~30 dòng mới | Không (chỉ hiển thị) |
| `SNIPPET_LEN=150` + expander | `main_web.py` | `render_reviews_section` | Không |
| `fetch_poster_url` (iTunes/Wiki) | `main_web.py` | ~30 dòng mới | Không |
| Layout 2 cột poster | `main_web.py` | `render_result_list` | Không |
| Hằng số `SNIPPET_LEN` | `main_web.py` | HẰNG SỐ block | Không |
| Mô hình / Training pipeline | **Không đụng** | — | — |

*KPDL_DoAn 2026 – Cập nhật tự động*

---

## 12. Nhật Ký Nâng Cấp Toàn Diện v6 (21/05/2026)

> **Phạm vi:** Cả 	rain_SVM_cosine_SVD.py và main_web.py

### TRAIN – Weighted TF-IDF Movie Profile
Trọng số review = log(rating+1)/log(11). Rating 10 → weight 1.0, rating 7 → weight 0.90.  
Profile phim = tổng có trọng số TF-IDF thay vì concat text đơn giản.

### TRAIN – Quality Filtering
Lọc review < 10 từ (spam) và > 2000 từ (auto-generated). Không đụng SVM pipeline.

### TRAIN – Offline Evaluation (Precision@K, NDCG@K, ILD)
Đánh giá hệ thống gợi ý offline trên mẫu 200 user. Kết quả append vào evaluation_metrics.txt.

### WEB – MMR Reranking
MMR_score = λ × relevance − (1−λ) × max_sim_to_selected  λ=0.6.  
Tránh gợi ý liên tiếp các phần của cùng 1 series.

### WEB – Hybrid Weight Slider
Slider trên Sidebar: w_cosine ∈ [0.3, 1.0]. Score = w×Cosine + (1−w)×SVD.

### WEB – Poster Placeholder Thẩm Mỹ
Hash color từ tên phim (HSL) + chữ viết tắt (initials). Không cần API. Layout không vỡ.

### WEB – Explainability Keywords
Hiển thị keywords giao thoa giữa query vector và movie vector cho từng kết quả.

| Thay đổi | Ảnh hưởng lý thuyết |
|---|---|
| Weighted profile, quality filter, offline eval | Không |
| MMR, hybrid slider, poster placeholder, explainability | Không |
| Lý thuyết SVM/TF-IDF/Cosine/SVD | Giữ nguyên |
