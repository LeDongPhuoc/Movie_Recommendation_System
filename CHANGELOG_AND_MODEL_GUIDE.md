# 📄 Tài Liệu Cập Nhật – Hệ Thống Gợi Ý Phim Hybrid

> **Dự án:** KPDL_DoAn · SVM + TF-IDF + Cosine Similarity + SVD  
> **Cập nhật:** 14/05/2026 (v2 – Cân bằng dữ liệu 1.5:1)

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
