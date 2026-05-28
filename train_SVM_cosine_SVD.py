import glob
import json
import math
import os
import re
import time
import warnings
import joblib
import nltk
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from scipy.sparse import csr_matrix, vstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=FutureWarning)

# ===========================================================================
# CẤU HÌNH TỔNG THỂ – Chỉnh tại đây, áp dụng toàn bộ script
# ===========================================================================
OUTPUT_DIR = "Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FOLDER_PATH          = "TrainData"
META_FILE_PATH       = "title.basics.tsv"

# --- Tham số dữ liệu ---
SAMPLE_SIZE          = 500_000   
RATING_POS_THRESHOLD = 7         
RATING_NEG_THRESHOLD = 4         
FILTER_YEAR          = True      

# --- Tham số chất lượng review ---
MIN_REVIEW_WORDS     = 10       
MAX_REVIEW_WORDS     = 2000     
MIN_REVIEWS_PER_FILM = 3         
# --- Tham số TF-IDF ---
TFIDF_MAX_FEATURES   = 50_000
TFIDF_MIN_DF         = 3         
TFIDF_MAX_DF         = 0.90      
TFIDF_NGRAM_RANGE    = (1, 3)   

# --- Tham số SVD ---
SVD_N_COMPONENTS     = 150       # Số chiều latent factors
SVD_N_ITER           = 10        # Số vòng lặp SVD (tăng → chính xác hơn nhưng chậm hơn)

# --- Tham số Offline Evaluation ---
EVAL_K_VALUES        = [5, 10]   # K dùng cho Precision@K và NDCG@K
EVAL_MIN_RATINGS     = 3         # User cần ít nhất n rating trong test set để đánh giá

# ===========================================================================
# NLTK SETUP
# ===========================================================================
for _res in ["corpora/stopwords", "tokenizers/punkt"]:
    try:
        nltk.data.find(_res)
    except LookupError:
        nltk.download(_res.split("/")[-1], quiet=True)

from nltk.corpus import stopwords  


_NEGATION_WORDS = {
    "not", "no", "nor", "n't", "don't", "isn't", "aren't", "couldn't",
    "didn't", "doesn't", "hadn't", "hasn't", "haven't", "shouldn't",
    "won't", "wouldn't", "never", "hardly", "nothing", "nowhere",
    "neither", "nobody", "none",
}
_STOP_WORDS = [w for w in stopwords.words("english") if w not in _NEGATION_WORDS]


# ===========================================================================
# BƯỚC 1: LỌC PHIM THEO NĂM (IMDb Metadata)
# ===========================================================================
def filter_movies_by_year(
    df: pd.DataFrame,
    meta_file_path: str,
    start_year: int = 1980,
    end_year: int = 2024,
) -> pd.DataFrame:
    """
    Lọc reviews chỉ giữ phim hợp lệ (movie/series) trong khoảng năm chỉ định.
    Trả về df gốc nếu gặp lỗi.
    """
    print(f"\n[!] Đang tải IMDb metadata: {meta_file_path}")
    try:
        meta_df = pd.read_csv(
            meta_file_path, sep="\t", na_values="\\N", low_memory=False,
            usecols=["tconst", "titleType", "primaryTitle", "originalTitle", "startYear"],
        )
        valid_types = {"movie", "tvSeries", "tvMovie", "short"}
        meta_df = meta_df[meta_df["titleType"].isin(valid_types)].rename(
            columns={"primaryTitle": "movie", "startYear": "year"}
        )
        meta_df["year"] = pd.to_numeric(meta_df["year"], errors="coerce")
        valid_df = meta_df[
            meta_df["year"].between(start_year, end_year)
        ]

        def _normalize(title: str) -> str:
            """Chuẩn hoá tên phim: lowercase → bỏ năm/số La Mã → bỏ ký tự đặc biệt."""
            title = str(title).lower()
            title = re.sub(r"\(\d{4}\)", "", title)
            title = re.sub(r"\([IVX]+\)", "", title)
            return re.sub(r"[^a-z0-9]", "", title).strip()

        print("-> Chuẩn hoá tên phim từ Metadata...")
        valid_names = set(valid_df["movie"].apply(_normalize))
        if "originalTitle" in valid_df.columns:
            valid_names |= set(valid_df["originalTitle"].apply(_normalize))

        print("-> Khớp tên phim trong dữ liệu review...")
        df = df.copy()
        df["_clean_name"] = df["movie"].apply(_normalize)
        n_before = len(df)
        df = df[df["_clean_name"].isin(valid_names)].drop(columns=["_clean_name"])
        print(f"-> Giữ lại {len(df):,}/{n_before:,} reviews ({len(df)/n_before*100:.1f}%).")
        return df

    except Exception as exc:
        print(f"Cảnh báo: Lỗi metadata ({exc}). Bỏ qua bước lọc năm.")
        return df


# ===========================================================================
# BƯỚC 2: TIỀN XỬ LÝ VĂN BẢN
# ===========================================================================
# Cụm từ đặc biệt → token đơn để TF-IDF bắt được tín hiệu mạnh hơn
_POSITIVE_PHRASES: dict[str, str] = {
    "must see": "mustsee", "must watch": "mustwatch",
    "highly recommend": "highlyrecommend", "well done": "welldone",
    "beautifully shot": "beautifullyshot", "edge of my seat": "edgeofmyseat",
    "stands out": "standsout", "life changing": "lifechanging",
}
_NEGATIVE_PHRASES: dict[str, str] = {
    "waste of time": "wasteoftimefilm", "waste of money": "wasteofmoneyfilm",
    "fell asleep": "fellasleepfilm", "walked out": "walkedoutfilm",
    "not worth": "notworthfilm", "poorly written": "poorlywritten",
    "too long": "toolongfilm", "bad acting": "badactingfilm",
}


def _replace_phrases(text: str, phrase_dict: dict) -> str:
    for phrase, replacement in phrase_dict.items():
        text = text.replace(phrase, replacement)
    return text


def clean_text(text: str) -> str:
    """Làm sạch văn bản: lowercase → bỏ HTML/URL → cụm từ → ký tự đặc biệt."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)           
    text = re.sub(r"https?://\S+", " ", text)    
    text = _replace_phrases(text, _POSITIVE_PHRASES)
    text = _replace_phrases(text, _NEGATIVE_PHRASES)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)   
    return re.sub(r"\s+", " ", text).strip()


def count_words(text: str) -> int:
    """Đếm số từ trong chuỗi đã làm sạch."""
    return len(str(text).split())


# ===========================================================================
# HELPER: ĐỌC FILE JSON LỚN AN TOÀN (JSON Lines & Standard Array)
# ===========================================================================
def _load_json_file(filepath: str) -> pd.DataFrame:
    """
    Đọc file JSON hỗ trợ 2 định dạng:
      1. JSON Lines – mỗi dòng là một object: {"k": v}
      2. Standard JSON Array – [{...}, {...}, ...]

    Lý do không dùng pd.read_json trực tiếp:
      - pd.read_json mặc định orient='columns' → đọc sai với Standard Array
        → mỗi file chỉ trả về 1 dòng thay vì hàng triệu dòng
    """
    try:
        df = pd.read_json(filepath, lines=True, encoding="utf-8")
        if df.shape[0] > 1:
            return df
    except Exception:
        pass

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.DataFrame.from_dict(data, orient="index")
    except Exception as exc:
        raise ValueError(f"Không thể đọc '{filepath}': {exc}") from exc

    raise ValueError(f"Định dạng không hỗ trợ: '{filepath}'")


# ===========================================================================
# BƯỚC 3: LOAD & TIỀN XỬ LÝ DỮ LIỆU
# ===========================================================================
def run_training_pipeline(
    folder_path: str,
    meta_file_path: str | None = None,
    sample_size: int = SAMPLE_SIZE,
    filter_year: bool = FILTER_YEAR,
    rating_pos_threshold: int = RATING_POS_THRESHOLD,
    rating_neg_threshold: int = RATING_NEG_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Quét tất cả part-*.json, gộp, chuẩn hoá, cân bằng và tiền xử lý.

    Nhãn sentiment:
      1  (Positive) : rating >= rating_pos_threshold
      0  (Negative) : rating <= rating_neg_threshold
      -1 (Neutral)  : bị loại khỏi tập huấn luyện SVM

    
      - Lọc review quá ngắn (< MIN_REVIEW_WORDS từ) hoặc quá dài (spam)
      - Giữ nguyên review gốc cho web_display nhưng chỉ dùng review
        chất lượng để build movie profile

    Returns:
      df_balanced : DataFrame sau lấy mẫu, có cột clean_text & label
      df_svm      : df_balanced lọc neutral & spoiler – dùng để train SVM
    """
    print(f"[!] Bắt đầu quét dữ liệu từ: {folder_path}")

    file_list = sorted(glob.glob(os.path.join(folder_path, "part-*.json")))
    if not file_list:
        raise FileNotFoundError(
            f"Không tìm thấy 'part-*.json' trong '{folder_path}'."
        )

    print(f"-> Tìm thấy {len(file_list)} file.")
    df_list = []
    for filepath in file_list:
        print(f"   + Đang nạp: {filepath}")
        part = _load_json_file(filepath)
        print(f"     -> {len(part):,} dòng | cột: {list(part.columns)}")
        df_list.append(part)

    df = pd.concat(df_list, ignore_index=True)
    print(f"Gộp thành công! Tổng review thô: {len(df):,} dòng.")

    # --- Chuẩn hoá tên cột ---
    df.columns = df.columns.astype(str).str.lower()
    _COL_MAP = {
        "review": "review_detail", "text": "review_detail",
        "content": "review_detail", "review_text": "review_detail",
        "review_content": "review_detail",
        "score": "rating", "stars": "rating", "ratings": "rating",
        "reviewer_rating": "rating",
        "spoiler": "spoiler_tag", "is_spoiler": "spoiler_tag",
        "spoilers": "spoiler_tag",
    }
    rename_map = {c: _COL_MAP[c] for c in df.columns if c in _COL_MAP}
    df = df.rename(columns=rename_map)

    if "spoiler_tag" not in df.columns:
        df["spoiler_tag"] = 0

    required_cols = ["rating", "review_detail"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Thiếu cột bắt buộc: {missing}\n"
            f"Các cột hiện có: {list(df.columns)}"
        )

    df = df.dropna(subset=required_cols)
    df["spoiler_tag"] = pd.to_numeric(df["spoiler_tag"], errors="coerce").fillna(0).astype(int)
    df["rating"]      = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])

    # --- Lọc năm ---
    if filter_year and meta_file_path and os.path.exists(meta_file_path):
        df = filter_movies_by_year(df, meta_file_path)

    # --- Cân bằng tỷ lệ pos:neg ≤ 1.5:1 ---
    df_pos = df[df["rating"] >= rating_pos_threshold]
    df_neg = df[df["rating"] <= rating_neg_threshold]
    n_pos, n_neg = len(df_pos), len(df_neg)
    print(f"  [Phân phối gốc] Positive: {n_pos:,} | Negative: {n_neg:,} | Tỷ lệ: {n_pos/max(n_neg,1):.2f}:1")

    _MAX_RATIO = 1.5
    max_pos_allowed = int(n_neg * _MAX_RATIO)
    if n_pos > max_pos_allowed:
        df_pos = df_pos.sample(max_pos_allowed, random_state=42)
        print(f"  [Undersample POS] Giảm Positive → {max_pos_allowed:,}")

    total_available = len(df_pos) + n_neg
    df_balanced = (
        pd.concat([df_pos, df_neg])
        .sample(min(sample_size, total_available), random_state=42)
        .reset_index(drop=True)
    )
    df_balanced.to_csv(os.path.join(OUTPUT_DIR, "data_raw.csv"), index=False, encoding="utf-8-sig")

    # --- Tiền xử lý văn bản ---
    df_balanced = df_balanced.rename(columns={"review_detail": "review"})
    print("Đang tiền xử lý văn bản...")
    df_balanced["clean_text"] = df_balanced["review"].apply(clean_text)

    # Lọc review quá ngắn hoặc quá dài (quality filtering)
    df_balanced["_word_count"] = df_balanced["clean_text"].apply(count_words)
    n_before = len(df_balanced)
    df_balanced = df_balanced[
        df_balanced["_word_count"].between(MIN_REVIEW_WORDS, MAX_REVIEW_WORDS)
    ].drop(columns=["_word_count"]).reset_index(drop=True)
    print(
        f"  [Quality Filter] Giữ lại {len(df_balanced):,}/{n_before:,} reviews "
        f"({len(df_balanced)/n_before*100:.1f}%) "
        f"[{MIN_REVIEW_WORDS}–{MAX_REVIEW_WORDS} từ]"
    )

    # --- Gán nhãn ---
    def _assign_label(r: float) -> int:
        if r >= rating_pos_threshold:  return 1
        if r <= rating_neg_threshold:  return 0
        return -1  

    df_balanced["label"] = df_balanced["rating"].apply(_assign_label)

    
    df_svm = df_balanced[
        (df_balanced["label"] != -1) & (df_balanced["spoiler_tag"] == 0)
    ].copy() 

    export_cols = ["review", "clean_text", "rating", "label", "spoiler_tag"]
    if "movie"    in df_balanced.columns: export_cols.append("movie")
    if "reviewer" in df_balanced.columns: export_cols.append("reviewer")
    df_svm[export_cols].to_csv(
        os.path.join(OUTPUT_DIR, "data_cleaned.csv"), index=False, encoding="utf-8-sig"
    )
    print(f"Phân phối nhãn sau lọc:\n{df_svm['label'].value_counts().to_string()}")
    return df_balanced, df_svm


# ===========================================================================
# BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH SENTIMENT (SVM)
# ===========================================================================
def evaluate_and_save(
    model,
    X_test,
    y_test,
    label_names: list[str] | None = None,
    output_path: str | None = None,
) -> dict:
    """
    Tính đầy đủ chỉ số: Accuracy, Precision, Recall, F1, ROC-AUC,
    Confusion Matrix. Lưu ra file text nếu output_path được cung cấp.
    """
    y_pred   = model.predict(X_test)
    acc      = accuracy_score(y_test, y_pred)
    prec_w   = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_w    = recall_score(y_test,   y_pred, average="weighted", zero_division=0)
    f1_w     = f1_score(y_test,       y_pred, average="weighted", zero_division=0)
    prec_mac = precision_score(y_test, y_pred, average="macro",   zero_division=0)
    rec_mac  = recall_score(y_test,   y_pred, average="macro",   zero_division=0)
    f1_mac   = f1_score(y_test,       y_pred, average="macro",   zero_division=0)
    cm       = confusion_matrix(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=label_names, zero_division=0)

    # ROC-AUC – chỉ tính khi model có predict_proba
    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            classes = sorted(np.unique(y_test))
            if len(classes) == 2:
                proba   = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, proba)
            else:
                y_bin   = label_binarize(y_test, classes=classes)
                proba   = model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
        except Exception:
            pass

    lines = [
        "=" * 60,
        "    KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH LinearSVC + Calibration",
        "=" * 60, "",
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
        "", "--- MA TRẬN NHẦM LẪN (Confusion Matrix) ---", str(cm), "",
        "--- BÁO CÁO PHÂN LỚP CHI TIẾT ---", report,
        "=" * 60,
    ]

    result_text = "\n".join(lines)
    print(result_text)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result_text)
        print(f"-> Đã lưu kết quả đánh giá: {output_path}")

    return {
        "accuracy": acc, "precision_weighted": prec_w, "recall_weighted": rec_w,
        "f1_weighted": f1_w, "precision_macro": prec_mac, "recall_macro": rec_mac,
        "f1_macro": f1_mac, "roc_auc": roc_auc, "confusion_matrix": cm, "y_pred": y_pred,
    }


# ===========================================================================
# BƯỚC 5: XÂY DỰNG MOVIE PROFILE VỚI WEIGHTED TF-IDF
# ===========================================================================
def build_weighted_movie_profiles(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    rating_pos_threshold: int = RATING_POS_THRESHOLD,
    min_reviews: int = MIN_REVIEWS_PER_FILM,
) -> tuple[pd.DataFrame, "csr_matrix"]:
    """
    Xây dựng hồ sơ phim (movie profile) bằng cách kết hợp có trọng số:
      - Trọng số mỗi review = log(rating + 1) / log(11)  [normalize về [0,1]]
      - Nhân trọng số vào vector TF-IDF tương ứng
      - Profile phim = tổng có trọng số của tất cả review tích cực

    Tại sao dùng weighted thay vì concat text?
      - Concat text: review 10/10 và review 7/10 có trọng số bằng nhau
      - Weighted:    review 10/10 đóng góp cao hơn → profile chính xác hơn

    Returns:
      movie_profiles    : DataFrame [movie, clean_text (concat gốc)]
      movie_tfidf_matrix: csr_matrix (N_films × N_features) đã có trọng số
    """
    print("\n[5] Đang xây dựng hồ sơ phim (Weighted TF-IDF)...")

    safe_positive = df[
        (df["label"] == 1) & (df["spoiler_tag"] == 0)
    ].copy()

    # Đếm review hợp lệ theo phim
    review_counts = safe_positive.groupby("movie").size().reset_index(name="count")
    valid_movies  = set(review_counts[review_counts["count"] >= min_reviews]["movie"])
    safe_positive = safe_positive[safe_positive["movie"].isin(valid_movies)]

    # Tính trọng số rating: log(rating+1) / log(11), normalize về [0,1]
    # Rating 10 → weight 1.0 | Rating 7 → weight ~0.90 | Rating 1 → weight ~0.30
    max_log = math.log(10 + 1)
    safe_positive = safe_positive.copy()
    safe_positive["weight"] = safe_positive["rating"].apply(
        lambda r: math.log(float(r) + 1) / max_log
    )

    # Xây dựng profile từng phim
    movies = safe_positive["movie"].unique()
    profile_rows  = []
    tfidf_rows    = []

    for movie in movies:
        film_reviews = safe_positive[safe_positive["movie"] == movie]
        texts   = film_reviews["clean_text"].tolist()
        weights = film_reviews["weight"].values

        # Vector TF-IDF từng review
        tfidf_mat = vectorizer.transform(texts)  # shape: (n_reviews, n_features)

        # Nhân trọng số vào từng dòng
        # weights.reshape(-1,1) broadcast thành (n_reviews, 1) → multiply col-wise
        weighted   = tfidf_mat.multiply(weights.reshape(-1, 1))
        profile_vec = weighted.sum(axis=0)  # (1, n_features) – tổng có trọng số

        tfidf_rows.append(csr_matrix(profile_vec))
        # Lưu concat text gốc để hiển thị (không dùng để tính cosine)
        profile_rows.append({
            "movie":      movie,
            "clean_text": " ".join(texts),
        })

    movie_profiles    = pd.DataFrame(profile_rows).reset_index(drop=True)
    movie_tfidf_matrix = vstack(tfidf_rows)  # (N_films × N_features)

    print(f"  -> Đã xây dựng profile có trọng số cho {len(movie_profiles):,} phim.")
    return movie_profiles, movie_tfidf_matrix


# ===========================================================================
# BƯỚC 6: OFFLINE EVALUATION CHO RECOMMENDATION
# ===========================================================================
def compute_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Precision@K = |recommended[:K] ∩ relevant| / K
    Đo lường: Trong K phim được gợi ý, bao nhiêu phim thực sự phù hợp?
    """
    top_k = recommended[:k]
    hits  = sum(1 for item in top_k if item in relevant)
    return hits / k if k > 0 else 0.0


def compute_ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    NDCG@K (Normalized Discounted Cumulative Gain) – đánh giá thứ hạng.
    Phim liên quan ở vị trí cao hơn → NDCG cao hơn.

    DCG@K  = Σ rel_i / log2(i+2)  với rel_i ∈ {0,1}
    IDCG@K = DCG của danh sách lý tưởng (tất cả relevant ở đầu)
    NDCG@K = DCG@K / IDCG@K
    """
    top_k = recommended[:k]
    dcg   = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(top_k)
        if item in relevant
    )
    # Số lượng phim liên quan có thể xếp hạng được
    n_ideal = min(len(relevant), k)
    idcg    = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    return dcg / idcg if idcg > 0 else 0.0


def compute_intra_list_diversity(
    recommended_titles: list,
    movie_tfidf_matrix: "csr_matrix",
    movie_index: dict,
) -> float:
    """
    Intra-List Diversity (ILD) = Đa dạng hoá danh sách gợi ý.
    ILD = 1 − avg(cosine_similarity của tất cả cặp phim trong danh sách)

    ILD gần 1 → Danh sách đa dạng (các phim rất khác nhau)
    ILD gần 0 → Danh sách đơn điệu (các phim quá giống nhau)
    """
    valid_indices = [
        movie_index[t] for t in recommended_titles if t in movie_index
    ]
    if len(valid_indices) < 2:
        return 0.0

    vecs = movie_tfidf_matrix[valid_indices]
    sim_matrix = cosine_similarity(vecs)

    # Lấy trung bình tam giác trên (không tính đường chéo)
    n = len(valid_indices)
    total_sim = sum(
        sim_matrix[i, j]
        for i in range(n) for j in range(i + 1, n)
    )
    n_pairs = n * (n - 1) / 2
    avg_sim = total_sim / n_pairs if n_pairs > 0 else 0.0
    return 1.0 - avg_sim


def evaluate_recommendation_offline(
    df: pd.DataFrame,
    movie_tfidf_matrix: "csr_matrix",
    movie_profiles: pd.DataFrame,
    k_values: list[int] = None,
    min_ratings: int = EVAL_MIN_RATINGS,
) -> dict:
    """
    Đánh giá hệ thống gợi ý offline bằng cách mô phỏng:
      - Mỗi user có ít nhất `min_ratings` lần rating trong test set
      - Dùng rating cao (>= RATING_POS_THRESHOLD) làm ground truth (relevant)
      - Tính Precision@K, NDCG@K trên mẫu người dùng
      - Tính Intra-List Diversity trung bình

    Lý do không dùng online evaluation:
      - Không có API A/B test
      - Offline evaluation là chuẩn mực học thuật cho recommender systems

    Returns:
      dict chứa metrics trung bình theo từng K
    """
    if k_values is None:
        k_values = EVAL_K_VALUES

    print("\n[Offline Eval] Đang tính Precision@K, NDCG@K, ILD...")

    if "reviewer" not in df.columns or "movie" not in df.columns:
        print("  -> Bỏ qua: thiếu cột 'reviewer' hoặc 'movie'.")
        return {}

    # Chỉ giữ phim có trong movie_profiles
    known_movies = set(movie_profiles["movie"].tolist())
    df_eval      = df[df["movie"].isin(known_movies)].copy()

    # Lập index: tên phim → vị trí dòng trong matrix
    movie_index = {row["movie"]: idx for idx, row in movie_profiles.iterrows()}

    # Tìm user có đủ dữ liệu để đánh giá
    user_counts = df_eval.groupby("reviewer").size()
    eval_users  = user_counts[user_counts >= min_ratings].index.tolist()

    if not eval_users:
        print(f"  -> Không có user nào có >= {min_ratings} ratings. Bỏ qua.")
        return {}

    # Lấy mẫu tối đa 200 user để chạy nhanh
    rng         = np.random.default_rng(42)
    sample_size = min(200, len(eval_users))
    sampled     = rng.choice(eval_users, size=sample_size, replace=False)

    results: dict[int, list] = {k: {"prec": [], "ndcg": []} for k in k_values}
    ild_scores = []
    max_k      = max(k_values)

    for user in sampled:
        user_df  = df_eval[df_eval["reviewer"] == user]
        relevant = set(user_df[user_df["rating"] >= RATING_POS_THRESHOLD]["movie"])
        if not relevant:
            continue

        # Dùng profile trung bình của user để tính cosine similarity
        user_idxs = [movie_index[m] for m in user_df["movie"] if m in movie_index]
        if not user_idxs:
            continue

        user_vec = np.asarray(
            movie_tfidf_matrix[user_idxs].mean(axis=0)
        ).reshape(1, -1)

        sims           = cosine_similarity(user_vec, movie_tfidf_matrix).flatten()
        sorted_indices = sims.argsort()[::-1]
        recommended    = [
            movie_profiles.iloc[i]["movie"]
            for i in sorted_indices[:max_k]
        ]

        for k in k_values:
            results[k]["prec"].append(compute_precision_at_k(recommended, relevant, k))
            results[k]["ndcg"].append(compute_ndcg_at_k(recommended, relevant, k))

        ild = compute_intra_list_diversity(recommended[:max_k], movie_tfidf_matrix, movie_index)
        ild_scores.append(ild)

    # Tính trung bình
    summary = {}
    for k in k_values:
        prec_list = results[k]["prec"]
        ndcg_list = results[k]["ndcg"]
        summary[k] = {
            "precision": np.mean(prec_list) if prec_list else 0.0,
            "ndcg":      np.mean(ndcg_list) if ndcg_list else 0.0,
        }
        print(f"  Precision@{k} = {summary[k]['precision']:.4f} | NDCG@{k} = {summary[k]['ndcg']:.4f}")

    avg_ild = np.mean(ild_scores) if ild_scores else 0.0
    print(f"  Intra-List Diversity (avg) = {avg_ild:.4f}")
    summary["ild"] = avg_ild
    summary["n_users_evaluated"] = len(sampled)
    return summary


def append_recommendation_metrics(
    rec_metrics: dict,
    output_path: str,
    k_values: list[int] = None,
) -> None:
    """Ghi thêm (append) kết quả offline evaluation vào file metrics."""
    if not rec_metrics or not output_path:
        return
    if k_values is None:
        k_values = EVAL_K_VALUES

    lines = [
        "",
        "=" * 60,
        "    ĐÁNH GIÁ HỆ THỐNG GỢI Ý (Offline Evaluation)",
        "=" * 60,
        f"  Số user đánh giá : {rec_metrics.get('n_users_evaluated', 0):,}",
        f"  K values         : {k_values}",
        "",
        "--- PRECISION@K & NDCG@K ---",
    ]
    for k in k_values:
        if k in rec_metrics:
            lines.append(
                f"  Precision@{k:<3}     : {rec_metrics[k]['precision']:.4f}"
            )
            lines.append(
                f"  NDCG@{k:<3}          : {rec_metrics[k]['ndcg']:.4f}"
            )
    ild = rec_metrics.get("ild", 0.0)
    lines += [
        "",
        "--- ĐA DẠNG HOÁ GỢI Ý ---",
        f"  Intra-List Div. : {ild:.4f}  (0=đơn điệu, 1=đa dạng)",
        "=" * 60,
    ]

    with open(output_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"-> Đã thêm Offline Eval vào: {output_path}")


# ===========================================================================
# MAIN – CHẠY TOÀN BỘ PIPELINE
# ===========================================================================
if __name__ == "__main__":
    try:
        start_time = time.time()

        # ------------------------------------------------------------------
        # GIAI ĐOẠN 1: TẢI & CHUẨN BỊ DỮ LIỆU
        # ------------------------------------------------------------------
        df, df_svm = run_training_pipeline(
            folder_path=FOLDER_PATH,
            meta_file_path=META_FILE_PATH,
        )

        X_raw = df_svm["clean_text"]
        y     = df_svm["label"]

        # ------------------------------------------------------------------
        # GIAI ĐOẠN 2: VECTOR HOÁ TF-IDF
        # Dùng corpus review (df_svm) để fit vectorizer.
        # Cùng vectorizer sẽ được dùng để transform movie profiles.
        # ------------------------------------------------------------------
        print("\n[1] Đang vector hoá TF-IDF...")
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            stop_words=_STOP_WORDS,
            ngram_range=TFIDF_NGRAM_RANGE,
            sublinear_tf=True,         # log-normalization giảm ảnh hưởng từ lặp nhiều
            analyzer="word",
            token_pattern=r"(?u)\b[a-zA-Z'][a-zA-Z']{1,}\b",  # bỏ token 1 ký tự
        )
        X_vectorized = vectorizer.fit_transform(X_raw)

        # ------------------------------------------------------------------
        # GIAI ĐOẠN 3: CHIA DỮ LIỆU (STRATIFIED 80/20)
        # ------------------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y,
            test_size=0.2, random_state=42, stratify=y,
        )
        print(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

        # Oversample nếu vẫn lệch sau chia
        train_dist  = pd.Series(y_train).value_counts()
        n_pos_train = train_dist.get(1, 0)
        n_neg_train = train_dist.get(0, 0)
        train_ratio = n_pos_train / max(n_neg_train, 1)
        if train_ratio > 1.5:
            print("  [!] Tỷ lệ > 1.5:1 → Áp dụng RandomOverSampler...")
            ros = RandomOverSampler(sampling_strategy=1 / 1.5, random_state=42)
            X_train, y_train = ros.fit_resample(X_train, y_train)
        else:
            print(f"  ✅ Tỷ lệ pos:neg = {train_ratio:.2f}:1, không cần oversample.")

        # ------------------------------------------------------------------
        # GIAI ĐOẠN 4: HUẤN LUYỆN LinearSVC + CalibratedClassifierCV
        # ------------------------------------------------------------------
        print("\n[2] Đang huấn luyện LinearSVC + Calibration...")
        base_svc = LinearSVC(
            C=1.0, max_iter=3000, dual="auto",
            class_weight="balanced", random_state=42,
        )
        # CalibratedClassifierCV → predict_proba → ROC-AUC
        model = CalibratedClassifierCV(base_svc, cv=3, method="sigmoid")
        model.fit(X_train, y_train)

        # ------------------------------------------------------------------
        # GIAI ĐOẠN 5: ĐÁNH GIÁ MÔ HÌNH SENTIMENT
        # Lưu vào evaluation_metrics.txt (tên chuẩn – main_web đọc được)
        # ------------------------------------------------------------------
        print("\n[3] Đánh giá mô hình trên tập TEST...")
        metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.txt")
        evaluate_and_save(
            model, X_test, y_test,
            label_names=["Negative (0)", "Positive (1)"],
            output_path=metrics_path,
        )

        print("\n[4] Đang chạy Cross-Validation (3-fold)...")
        _cv_est = CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=42), cv=3
        )
        cv_scores = cross_val_score(
            _cv_est, X_vectorized, y,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring="f1_weighted", n_jobs=-1,
        )
        print(f"  CV F1-Weighted: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # ------------------------------------------------------------------
        # GIAI ĐOẠN 6: XÂY DỰNG HỒ SƠ PHIM (WEIGHTED TF-IDF)
        # ------------------------------------------------------------------
        if "movie" in df.columns:
            movie_profiles, movie_tfidf_matrix = build_weighted_movie_profiles(
                df, vectorizer,
                rating_pos_threshold=RATING_POS_THRESHOLD,
                min_reviews=MIN_REVIEWS_PER_FILM,
            )

            joblib.dump(movie_profiles,     os.path.join(OUTPUT_DIR, "content_movie_profiles.joblib"))
            joblib.dump(movie_tfidf_matrix, os.path.join(OUTPUT_DIR, "content_tfidf_matrix.joblib"))
            print(f"  -> Lưu profiles cho {len(movie_profiles):,} phim.")

            # Review cho web display (bao gồm cả spoiler – người dùng tự mở)
            web_reviews = df[df["label"] == 1][
                ["movie", "reviewer", "rating", "review", "spoiler_tag"]
            ].copy()
            joblib.dump(web_reviews, os.path.join(OUTPUT_DIR, "web_display_reviews.joblib"))

            # ------------------------------------------------------------------
            # GIAI ĐOẠN 6b: OFFLINE EVALUATION CHO RECOMMENDATION
            # ------------------------------------------------------------------
            rec_metrics = evaluate_recommendation_offline(
                df, movie_tfidf_matrix, movie_profiles,
                k_values=EVAL_K_VALUES,
            )
            append_recommendation_metrics(rec_metrics, metrics_path, EVAL_K_VALUES)

        # ------------------------------------------------------------------
        # GIAI ĐOẠN 7: SVD COLLABORATIVE FILTERING
        # ------------------------------------------------------------------
        if "reviewer" in df.columns and "movie" in df.columns:
            print("\n[6] Phân rã ma trận User-Item bằng TruncatedSVD...")
            user_mapping  = {u: i for i, u in enumerate(df["reviewer"].unique())}
            movie_mapping = {m: i for i, m in enumerate(df["movie"].unique())}

            u_idx = df["reviewer"].map(user_mapping)
            m_idx = df["movie"].map(movie_mapping)

            user_item_matrix = csr_matrix(
                (df["rating"].values, (u_idx.values, m_idx.values)),
                shape=(len(user_mapping), len(movie_mapping)),
            )

            svd = TruncatedSVD(
                n_components=SVD_N_COMPONENTS,
                n_iter=SVD_N_ITER,
                random_state=42,
            )
            user_factors = svd.fit_transform(user_item_matrix)
            item_factors = svd.components_.T
            explained_var = svd.explained_variance_ratio_.sum()

            print(f"  -> SVD: {SVD_N_COMPONENTS} components, {explained_var*100:.1f}% explained variance.")

            # Ghi thêm kết quả SVD vào file metrics
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(
                    f"\n--- SVD Collaborative Filtering ---\n"
                    f"  n_components      : {SVD_N_COMPONENTS}\n"
                    f"  Explained Variance: {explained_var:.4f} ({explained_var*100:.1f}%)\n"
                    f"  User count        : {len(user_mapping):,}\n"
                    f"  Movie count       : {len(movie_mapping):,}\n"
                )

            joblib.dump(
                {
                    "user_map":           user_mapping,
                    "movie_map":          movie_mapping,
                    "user_factors":       user_factors,
                    "item_factors":       item_factors,
                    "explained_variance": explained_var,
                },
                os.path.join(OUTPUT_DIR, "collab_svd_model.joblib"),
            )

        # ------------------------------------------------------------------
        # GIAI ĐOẠN 8: LƯU MÔ HÌNH CHÍNH
        # ------------------------------------------------------------------
        print("\n[7] Lưu mô hình Sentiment...")
        joblib.dump(model,      os.path.join(OUTPUT_DIR, "model_sentiment_linearsvc.joblib"))
        joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, "vectorizer_tfidf.joblib"))

        elapsed = time.time() - start_time
        print(f"\n✅ Hoàn tất pipeline trong {elapsed:.1f}s ({elapsed/60:.1f} phút).")

    except Exception as exc:
        import traceback
        print(f"Lỗi hệ thống: {exc}")
        traceback.print_exc()