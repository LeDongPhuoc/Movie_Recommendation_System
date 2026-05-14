import pandas as pd
import numpy as np
import re
import time
import joblib
import os
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import RandomOverSampler

# --- THIẾT LẬP THƯ MỤC LƯU TRỮ ---
OUTPUT_DIR = "Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- NLTK SETUP ---
for resource in ['corpora/stopwords', 'tokenizers/punkt']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1])

stop_words = list(stopwords.words('english'))
# Giữ lại các từ phủ định vì chúng quan trọng với sentiment
NEGATIVE_WORDS = [
    'not', 'no', 'nor', "n't", "don't", "isn't", "aren't", "couldn't",
    "didn't", "doesn't", "hadn't", "hasn't", "haven't", "shouldn't",
    "won't", "wouldn't", "never", "hardly", "nothing", "nowhere",
    "neither", "nobody", "none"
]
stop_words = [w for w in stop_words if w not in NEGATIVE_WORDS]


# ============================================================
# BƯỚC 1: LỌC PHIM THEO NĂM & LOẠI
# ============================================================
def filter_movies_by_year(df, meta_file_path, start_year=1980, end_year=2024):
    """Lọc phim hợp lệ dựa trên file metadata IMDb."""
    print(f"\n[!] Đang đọc file IMDb metadata từ {meta_file_path}...")
    try:
        meta_df = pd.read_csv(
            meta_file_path, sep='\t', na_values='\\N', low_memory=False,
            usecols=['tconst', 'titleType', 'primaryTitle', 'originalTitle', 'startYear']
        )
        # Chỉ giữ phim & series
        meta_df = meta_df[meta_df['titleType'].isin(['movie', 'tvSeries', 'tvMovie', 'short'])]
        meta_df = meta_df.rename(columns={'primaryTitle': 'movie', 'startYear': 'year'})
        meta_df['year'] = pd.to_numeric(meta_df['year'], errors='coerce')
        valid_movies_df = meta_df[
            (meta_df['year'] >= start_year) & (meta_df['year'] <= end_year)
        ].copy()

        def super_clean_title(title):
            title = str(title).lower()
            title = re.sub(r'\(\d{4}\)', '', title)
            title = re.sub(r'\([IVX]+\)', '', title)
            title = re.sub(r'[^a-z0-9]', '', title)
            return title.strip()

        print("-> Đang làm sạch tên phim từ file Metadata...")
        valid_names = set(valid_movies_df['movie'].apply(super_clean_title))
        if 'originalTitle' in valid_movies_df.columns:
            valid_names |= set(valid_movies_df['originalTitle'].apply(super_clean_title))

        print("-> Đang khớp tên phim trong file JSON...")
        df = df.copy()
        df['_clean_name'] = df['movie'].apply(super_clean_title)
        original_len = len(df)
        df = df[df['_clean_name'].isin(valid_names)].drop(columns=['_clean_name'])
        print(f"-> Giữ lại {len(df):,}/{original_len:,} reviews ({len(df)/original_len*100:.1f}%).")
        return df
    except Exception as e:
        print(f"Cảnh báo: Lỗi khi xử lý file metadata ({e}). Bỏ qua bước lọc năm.")
        return df


# ============================================================
# BƯỚC 2: TIỀN XỬ LÝ VĂN BẢN NÂNG CAO
# ============================================================
# Từ điển mở rộng mang nghĩa tích cực / tiêu cực để tăng tín hiệu
POSITIVE_PHRASES = {
    "must see": "mustsee", "must watch": "mustwatch",
    "highly recommend": "highlyrecommend", "well done": "welldone",
    "beautifully shot": "beautifullyshot", "edge of my seat": "edgeofmyseat",
    "stands out": "standsout", "life changing": "lifechanging",
}
NEGATIVE_PHRASES = {
    "waste of time": "wasteoftimefilm", "waste of money": "wasteofmoneyfilm",
    "fell asleep": "fellasleepfilm", "walked out": "walkedoutfilm",
    "not worth": "notworthfilm", "poorly written": "poorlywritten",
    "too long": "toolongfilm", "bad acting": "badactingfilm",
}

def replace_phrases(text, phrase_dict):
    for phrase, replacement in phrase_dict.items():
        text = text.replace(phrase, replacement)
    return text

def clean_text(text):
    """Làm sạch và chuẩn hóa văn bản review."""
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)          # Loại HTML tags
    text = re.sub(r'https?://\S+', ' ', text)   # Loại URLs
    # Thay thế cụm từ đặc biệt trước khi xóa ký tự đặc biệt
    text = replace_phrases(text, POSITIVE_PHRASES)
    text = replace_phrases(text, NEGATIVE_PHRASES)
    text = re.sub(r"[^a-zA-Z\\s']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================
# BƯỚC 3: PIPELINE LOAD & XỬ LÝ DỮ LIỆU
# ============================================================
def run_training_pipeline(
    file_path,
    meta_file_path=None,
    sample_size=200000,
    filter_year=True,
    rating_pos_threshold=7,
    rating_neg_threshold=4
):
    """
    Load dữ liệu JSON, lọc & tiền xử lý, gán nhãn sentiment.
    
    Nhãn:
      1 (Positive) : rating >= rating_pos_threshold
      0 (Negative) : rating <= rating_neg_threshold
      -1 (Neutral) : bị loại bỏ
    """
    print(f"Bắt đầu đọc dữ liệu từ: {file_path}")
    try:
        df = pd.read_json(file_path)
    except ValueError:
        df = pd.read_json(file_path, lines=True)

    df.columns = df.columns.astype(str).str.lower()

    # --- Chuẩn hóa tên cột ---
    col_map = {}
    for col in df.columns:
        if col in ['review', 'text', 'content', 'review_text', 'review_detail', 'review_content']:
            col_map[col] = 'review_detail'
        if col in ['score', 'stars', 'rating', 'ratings', 'reviewer_rating']:
            col_map[col] = 'rating'
        if col in ['spoiler', 'spoiler_tag', 'is_spoiler', 'spoilers']:
            col_map[col] = 'spoiler_tag'
    df = df.rename(columns=col_map)

    if 'spoiler_tag' not in df.columns:
        df['spoiler_tag'] = 0

    required_cols = ['rating', 'review_detail']
    df = df.dropna(subset=required_cols)
    df['spoiler_tag'] = pd.to_numeric(df['spoiler_tag'], errors='coerce').fillna(0).astype(int)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])

    # --- Lọc theo năm (nếu có metadata) ---
    if filter_year and meta_file_path and os.path.exists(meta_file_path):
        df = filter_movies_by_year(df, meta_file_path, start_year=1980, end_year=2024)

    # --- Lấy mẫu cân bằng hơn (tỷ lệ pos:neg tối đa 1.5:1) ---
    df_pos = df[df['rating'] >= rating_pos_threshold]
    df_neg = df[df['rating'] <= rating_neg_threshold]

    n_neg = len(df_neg)
    n_pos = len(df_pos)
    print(f"  [Dữ liệu gốc] Positive: {n_pos:,} | Negative: {n_neg:,} | Tỷ lệ: {n_pos/max(n_neg,1):.2f}:1")

    # Giới hạn tỷ lệ pos:neg KHÔNG VƯỢT QUÁ 1.5:1
    MAX_RATIO = 1.5
    max_pos = int(n_neg * MAX_RATIO)  # pos tối đa = 1.5 × số neg
    if n_pos > max_pos:
        df_pos = df_pos.sample(max_pos, random_state=42)
        print(f"  [Undersample POS] Giảm Positive xuống {max_pos:,} (tỷ lệ mục tiêu {MAX_RATIO}:1)")

    actual_ratio = len(df_pos) / max(n_neg, 1)
    print(f"  [Sau cân bằng]   Positive: {len(df_pos):,} | Negative: {n_neg:,} | Tỷ lệ: {actual_ratio:.2f}:1")

    df_balanced = pd.concat([df_pos, df_neg]).sample(
        min(sample_size, len(df_pos) + n_neg), random_state=42
    ).reset_index(drop=True)

    df_balanced.to_csv(os.path.join(OUTPUT_DIR, 'data_raw.csv'), index=False, encoding='utf-8-sig')

    # --- Tiền xử lý ---
    df_balanced = df_balanced.rename(columns={'review_detail': 'review'})
    print("Đang tiền xử lý văn bản nâng cao...")
    df_balanced['clean_text'] = df_balanced['review'].apply(clean_text)

    # --- Gán nhãn ---
    df_balanced['label'] = df_balanced['rating'].apply(
        lambda x: 1 if x >= rating_pos_threshold else (0 if x <= rating_neg_threshold else -1)
    )
    # Chỉ giữ positive / negative (bỏ neutral) & lọc spoiler cho SVM
    df_svm = df_balanced[
        (df_balanced['label'] != -1) & (df_balanced['spoiler_tag'] == 0)
    ].copy()

    export_cols = ['review', 'clean_text', 'rating', 'label', 'spoiler_tag']
    if 'movie' in df_balanced.columns: export_cols.append('movie')
    if 'reviewer' in df_balanced.columns: export_cols.append('reviewer')
    df_svm[export_cols].to_csv(
        os.path.join(OUTPUT_DIR, 'data_cleaned.csv'), index=False, encoding='utf-8-sig'
    )

    print(f"Phân phối nhãn:\n{df_svm['label'].value_counts().to_string()}")
    return df_balanced, df_svm


# ============================================================
# BƯỚC 4: ĐÁNH GIÁ & LƯU KẾT QUẢ
# ============================================================
def evaluate_and_save(model, X_test, y_test, label_names=None, output_path=None):
    """Tính toán và lưu đầy đủ các chỉ số đánh giá mô hình."""
    y_pred = model.predict(X_test)

    acc     = accuracy_score(y_test, y_pred)
    prec_w  = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_w   = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_w    = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    prec_mac= precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_mac = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_mac  = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred, target_names=label_names, zero_division=0)

    # ROC-AUC (chỉ khi model có predict_proba)
    roc_auc = None
    if hasattr(model, 'predict_proba'):
        try:
            classes = sorted(np.unique(y_test))
            if len(classes) == 2:
                proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, proba)
            else:
                y_bin = label_binarize(y_test, classes=classes)
                proba = model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_bin, proba, average='macro', multi_class='ovr')
        except Exception:
            pass

    lines = [
        "=" * 60,
        "    KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH LinearSVC + Calibration",
        "=" * 60,
        "",
        "--- CHỈ SỐ TỔNG QUAN ---",
        f"  Accuracy              : {acc:.4f}  ({acc*100:.2f}%)",
        f"  Precision (Weighted)  : {prec_w:.4f}",
        f"  Recall    (Weighted)  : {rec_w:.4f}",
        f"  F1-Score  (Weighted)  : {f1_w:.4f}",
        f"  Precision (Macro)     : {prec_mac:.4f}",
        f"  Recall    (Macro)     : {rec_mac:.4f}",
        f"  F1-Score  (Macro)     : {f1_mac:.4f}",
    ]
    if roc_auc is not None:
        lines.append(f"  ROC-AUC               : {roc_auc:.4f}")
    lines += [
        "",
        "--- MA TRẬN NHẦM LẪN (Confusion Matrix) ---",
        str(cm),
        "",
        "--- BÁO CÁO PHÂN LỚP CHI TIẾT ---",
        report,
        "=" * 60,
    ]

    result_text = "\n".join(lines)
    print(result_text)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result_text)
        print(f"-> Đã lưu kết quả đánh giá vào: {output_path}")

    return {
        'accuracy': acc, 'precision_weighted': prec_w, 'recall_weighted': rec_w,
        'f1_weighted': f1_w, 'precision_macro': prec_mac, 'recall_macro': rec_mac,
        'f1_macro': f1_mac, 'roc_auc': roc_auc, 'confusion_matrix': cm,
        'y_pred': y_pred
    }


# ============================================================
# MAIN: CHẠY TOÀN BỘ PIPELINE
# ============================================================
if __name__ == "__main__":
    FILE_PATH      = 'part-03.json'
    META_FILE_PATH = 'title.basics.tsv'

    try:
        start_time = time.time()

        # --------------------------------------------------
        # GIAI ĐOẠN 1: TẢI & CHUẨN BỊ DỮ LIỆU
        # --------------------------------------------------
        df, df_svm = run_training_pipeline(
            FILE_PATH,
            meta_file_path=META_FILE_PATH,
            sample_size=150000,
            filter_year=True,
            rating_pos_threshold=7,
            rating_neg_threshold=4
        )

        X_raw = df_svm['clean_text']
        y     = df_svm['label']

        # --------------------------------------------------
        # GIAI ĐOẠN 2: VECTOR HÓA TF-IDF NÂNG CAO
        # --------------------------------------------------
        print("\n[1] Đang vector hóa TF-IDF nâng cao...")
        vectorizer = TfidfVectorizer(
            max_features=30000,       # Tăng từ điển để bắt nhiều đặc trưng hơn
            min_df=3,                 # Giảm ngưỡng để giữ từ hiếm có nghĩa
            max_df=0.90,              # Lọc các từ xuất hiện > 90% (quá phổ biến)
            stop_words=stop_words,
            ngram_range=(1, 3),       # Thêm 3-gram để bắt cụm từ dài hơn
            sublinear_tf=True,        # Log normalization giảm ảnh hưởng từ lặp nhiều
            analyzer='word',
            token_pattern=r"(?u)\b[a-zA-Z'][a-zA-Z']{1,}\b"  # Bỏ từ 1 ký tự
        )
        X_vectorized = vectorizer.fit_transform(X_raw)

        # --------------------------------------------------
        # GIAI ĐOẠN 3: CHIA DỮ LIỆU (STRATIFIED)
        # --------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y,
            test_size=0.2,
            random_state=42,
            stratify=y      # Đảm bảo tỷ lệ lớp đồng đều ở train/test
        )

        print(f"  Train set: {X_train.shape[0]:,} mẫu | Test set: {X_test.shape[0]:,} mẫu")
        train_dist = pd.Series(y_train).value_counts()
        print(f"  Phân phối nhãn TRAIN:\n{train_dist.to_string()}")

        # Kiểm tra tỷ lệ sau khi chia – nếu vẫn > 1.5:1 thì dùng RandomOverSampler
        n_pos_train = train_dist.get(1, 0)
        n_neg_train = train_dist.get(0, 0)
        train_ratio = n_pos_train / max(n_neg_train, 1)
        print(f"  Tỷ lệ pos:neg trong TRAIN = {train_ratio:.2f}:1")
        if train_ratio > 1.5:
            print("  [!] Tỷ lệ > 1.5:1 → Áp dụng RandomOverSampler cho lớp Negative...")
            ros = RandomOverSampler(sampling_strategy=1/1.5, random_state=42)
            X_train, y_train = ros.fit_resample(X_train, y_train)
            new_dist = pd.Series(y_train).value_counts()
            print(f"  Phân phối sau oversample: {new_dist.to_dict()}")
            new_ratio = new_dist.get(1, 0) / max(new_dist.get(0, 1), 1)
            print(f"  Tỷ lệ pos:neg sau oversample = {new_ratio:.2f}:1")
        else:
            print("  ✅ Tỷ lệ pos:neg đã đạt ≤ 1.5:1, không cần oversample.")

        # --------------------------------------------------
        # GIAI ĐOẠN 4: HUẤN LUYỆN LinearSVC + CalibratedClassifierCV
        # --------------------------------------------------
        print("\n[2] Đang huấn luyện mô hình LinearSVC + Calibration...")
        base_svc = LinearSVC(
            C=1.0,              # Tham số regularization (tối ưu qua thực nghiệm)
            max_iter=3000,      # Tăng vòng lặp để đảm bảo hội tụ
            dual=True,          # Hiệu quả khi n_samples > n_features
            class_weight='balanced',  # Tự động cân bằng lớp mất cân bằng
            random_state=42
        )

        # CalibratedClassifierCV giúp model xuất xác suất (predict_proba)
        # và cải thiện calibration của xác suất dự đoán
        model = CalibratedClassifierCV(base_svc, cv=3, method='sigmoid')
        model.fit(X_train, y_train)

        # --------------------------------------------------
        # GIAI ĐOẠN 5: ĐÁNH GIÁ MÔ HÌNH
        # --------------------------------------------------
        print("\n[3] Đánh giá mô hình trên tập TEST...")
        label_names = ['Negative (0)', 'Positive (1)']
        eval_results = evaluate_and_save(
            model, X_test, y_test,
            label_names=label_names,
            output_path=os.path.join(OUTPUT_DIR, 'evaluation_metrics.txt')
        )

        # Cross-validation để kiểm tra tính ổn định
        print("\n[4] Đang chạy Cross-Validation (3-fold) để kiểm tra độ ổn định...")
        cv_scores = cross_val_score(
            CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000, class_weight='balanced', random_state=42), cv=3),
            X_vectorized, y, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='f1_weighted', n_jobs=-1
        )
        print(f"  CV F1-Weighted: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # --------------------------------------------------
        # GIAI ĐOẠN 6: XÂY DỰNG HỒ SƠ PHIM (CONTENT-BASED)
        # --------------------------------------------------
        if 'movie' in df.columns:
            print("\n[5] Đang xây dựng Hồ sơ Phim (Content-Based Filtering)...")
            # Chỉ dùng review tích cực & không spoiler để xây profile
            safe_positive = df[(df['label'] == 1) & (df['spoiler_tag'] == 0)].copy()

            # Tạo profile phim: ghép các review thành 1 đoạn văn đại diện
            movie_profiles = (
                safe_positive
                .groupby('movie')['clean_text']
                .apply(lambda x: ' '.join(x))
                .reset_index()
            )

            # Chỉ giữ phim có ít nhất 3 review tích cực (đảm bảo chất lượng profile)
            review_count = safe_positive.groupby('movie').size().reset_index(name='count')
            movie_profiles = movie_profiles.merge(review_count, on='movie')
            movie_profiles = movie_profiles[movie_profiles['count'] >= 3].drop(columns=['count'])
            movie_profiles = movie_profiles.reset_index(drop=True)

            movie_tfidf_matrix = vectorizer.transform(movie_profiles['clean_text'])

            joblib.dump(movie_profiles,    os.path.join(OUTPUT_DIR, 'content_movie_profiles.joblib'))
            joblib.dump(movie_tfidf_matrix, os.path.join(OUTPUT_DIR, 'content_tfidf_matrix.joblib'))
            print(f"  -> Đã xây dựng profile cho {len(movie_profiles):,} phim.")

            # Lưu dữ liệu review để hiển thị trên web (bao gồm cả spoiler để người dùng có thể mở)
            web_reviews = df[df['label'] == 1][
                ['movie', 'reviewer', 'rating', 'review', 'spoiler_tag']
            ].copy()
            joblib.dump(web_reviews, os.path.join(OUTPUT_DIR, 'web_display_reviews.joblib'))

        # --------------------------------------------------
        # GIAI ĐOẠN 7: PHÂN RÃ SVD (COLLABORATIVE FILTERING)
        # --------------------------------------------------
        if 'reviewer' in df.columns and 'movie' in df.columns:
            print("\n[6] Đang phân rã ma trận User-Item bằng SVD...")
            user_mapping  = {u: i for i, u in enumerate(df['reviewer'].unique())}
            movie_mapping = {m: i for i, m in enumerate(df['movie'].unique())}

            u_idx = df['reviewer'].map(user_mapping)
            m_idx = df['movie'].map(movie_mapping)

            # Ma trận user-item dùng rating gốc (1-10) làm giá trị
            user_item_matrix = csr_matrix(
                (df['rating'], (u_idx, m_idx)),
                shape=(len(user_mapping), len(movie_mapping))
            )

            # n_components=100 bắt được nhiều latent factor hơn
            svd = TruncatedSVD(n_components=100, n_iter=10, random_state=42)
            user_factors = svd.fit_transform(user_item_matrix)
            item_factors = svd.components_.T

            explained_var = svd.explained_variance_ratio_.sum()
            print(f"  -> SVD giải thích {explained_var*100:.1f}% phương sai.")

            joblib.dump(
                {
                    'user_map':    user_mapping,
                    'movie_map':   movie_mapping,
                    'user_factors': user_factors,
                    'item_factors': item_factors,
                    'explained_variance': explained_var
                },
                os.path.join(OUTPUT_DIR, 'collab_svd_model.joblib')
            )

        # --------------------------------------------------
        # GIAI ĐOẠN 8: LƯU MÔ HÌNH CHÍNH
        # --------------------------------------------------
        print("\n[7] Đang lưu mô hình Sentiment...")
        joblib.dump(model,      os.path.join(OUTPUT_DIR, 'model_sentiment_linearsvc.joblib'))
        joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, 'vectorizer_tfidf.joblib'))

        elapsed = time.time() - start_time
        print(f"\n✅ Hoàn tất toàn bộ pipeline trong {elapsed:.1f}s ({elapsed/60:.1f} phút).")

    except Exception as e:
        import traceback
        print(f"Lỗi hệ thống: {e}")
        traceback.print_exc()