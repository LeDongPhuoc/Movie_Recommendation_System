

# =====================================================================
# IMPORT – giữ nguyên hoàn toàn từ train_SVM_cosine_SVD.py
# =====================================================================
import glob, json, math, os, re, time, warnings
import joblib, nltk, numpy as np, pandas as pd
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================================
# CẤU HÌNH
# =====================================================================
OUTPUT_DIR  = r"Experiments\v8_svd_500" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

FOLDER_PATH      = "TrainData"
META_FILE_PATH   = "title.basics.tsv"


SAMPLE_SIZE          = 750_000    # Số review mẫu
RATING_POS_THRESHOLD = 7          # Xếp loại rating >= 7 là positive
RATING_NEG_THRESHOLD = 4          # Xếp loại rating <= 4 là negative
FILTER_YEAR          = True       # Lọc theo năm

MIN_REVIEW_WORDS     = 10         # Số từ tối thiểu trong review
MAX_REVIEW_WORDS     = 2000       # Số từ tối đa trong review
MIN_REVIEWS_PER_FILM = 2         

TFIDF_MAX_FEATURES   = 60_000     # Số lượng từ tối đa trong TF-IDF vectorizer
TFIDF_MIN_DF         = 3          # Loại bỏ từ xuất hiện ít hơn 3 lần
TFIDF_MAX_DF         = 0.90       # Loại bỏ từ xuất hiện trong >90% review (có thể là stopword hoặc từ chung chung)
TFIDF_NGRAM_RANGE    = (1, 3)     # Sử dụng unigrams, bigrams và trigrams

SVD_N_COMPONENTS     = 500        # Số chiều sau khi giảm với SVD 
SVD_N_ITER           = 15         # Số lần lặp tối đa cho thuật toán SVD (tăng lên để hội tụ tốt hơn với nhiều thành phần)

EVAL_K_VALUES        = [5, 10]   # Các giá trị K để đánh giá Precision@K và NDCG@K trong offline evaluation
EVAL_MIN_RATINGS     = 3         # Số đánh giá tối thiểu của user để được chọn vào đánh giá offline

MMR_LAMBDA        = 0.5       # Cân bằng giữa relevance và diversity trong MMR (0.0 = chỉ relevance, 1.0 = chỉ diversity)

# =====================================================================
# GHI CONFIG VÀO FILE (để tracking)
# =====================================================================
import json as _json
_config = {
    "version": "v8",
    "label": "V8 – 750K, SVD 500, Min Review 2, SVD Filter User >=3",
    "date": time.strftime("%Y-%m-%d"),
    "config": {
        "sample_size": SAMPLE_SIZE,
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "tfidf_min_df": TFIDF_MIN_DF,
        "tfidf_ngram_range": list(TFIDF_NGRAM_RANGE),
        "svd_n_components": SVD_N_COMPONENTS,
        "svd_n_iter": SVD_N_ITER,
        "svc_C": 1.0,
        "mmr_lambda": MMR_LAMBDA,
        "min_reviews_per_film": MIN_REVIEWS_PER_FILM,
        "rating_pos_threshold": RATING_POS_THRESHOLD,
        "rating_neg_threshold": RATING_NEG_THRESHOLD,
    }
}
with open(os.path.join(OUTPUT_DIR, "config.json"), "w", encoding="utf-8") as _f:
    _json.dump(_config, _f, ensure_ascii=False, indent=2)
print(f" Config đã lưu vào {OUTPUT_DIR}/config.json")

# =====================================================================
# COPY TOÀN BỘ HÀM TỪ train_SVM_cosine_SVD.py
# (Giữ nguyên – không thay đổi logic)
# =====================================================================

# --- NLTK ---
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

_POSITIVE_PHRASES = {
    "must see": "mustsee", "must watch": "mustwatch",
    "highly recommend": "highlyrecommend", "well done": "welldone",
    "beautifully shot": "beautifullyshot", "edge of my seat": "edgeofmyseat",
    "stands out": "standsout", "life changing": "lifechanging",
}
_NEGATIVE_PHRASES = {
    "waste of time": "wasteoftimefilm", "waste of money": "wasteofmoneyfilm",
    "fell asleep": "fellasleepfilm", "walked out": "walkedoutfilm",
    "not worth": "notworthfilm", "poorly written": "poorlywritten",
    "too long": "toolongfilm", "bad acting": "badactingfilm",
}


def _replace_phrases(text, phrase_dict):
    for phrase, replacement in phrase_dict.items():
        text = text.replace(phrase, replacement)
    return text


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = _replace_phrases(text, _POSITIVE_PHRASES)
    text = _replace_phrases(text, _NEGATIVE_PHRASES)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def count_words(text):
    return len(str(text).split())


def _load_json_file(filepath):
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


def filter_movies_by_year(df, meta_file_path, start_year=1980, end_year=2021):
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
        valid_df = meta_df[meta_df["year"].between(start_year, end_year)]

        def _normalize(title):
            title = str(title).lower()
            title = re.sub(r"\(\d{4}\)", "", title)
            title = re.sub(r"\([IVX]+\)", "", title)
            return re.sub(r"[^a-z0-9]", "", title).strip()

        valid_names = set(valid_df["movie"].apply(_normalize))
        if "originalTitle" in valid_df.columns:
            valid_names |= set(valid_df["originalTitle"].apply(_normalize))

        df = df.copy()
        df["_clean_name"] = df["movie"].apply(_normalize)
        n_before = len(df)
        df = df[df["_clean_name"].isin(valid_names)].drop(columns=["_clean_name"])
        print(f"-> Giữ lại {len(df):,}/{n_before:,} reviews ({len(df)/n_before*100:.1f}%).")
        return df
    except Exception as exc:
        print(f"Cảnh báo: Lỗi metadata ({exc}). Bỏ qua.")
        return df


def run_training_pipeline(folder_path, meta_file_path=None):
    print(f"[!] Bắt đầu quét dữ liệu từ: {folder_path}")
    file_list = sorted(glob.glob(os.path.join(folder_path, "part-*.json")))
    if not file_list:
        raise FileNotFoundError(f"Không tìm thấy part-*.json trong '{folder_path}'.")

    print(f"-> Tìm thấy {len(file_list)} file.")
    df_list = []
    for filepath in file_list:
        print(f"   + Nạp: {filepath}")
        part = _load_json_file(filepath)
        df_list.append(part)

    df = pd.concat(df_list, ignore_index=True)
    print(f"Tổng review thô: {len(df):,}")

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
    df = df.rename(columns={c: _COL_MAP[c] for c in df.columns if c in _COL_MAP})

    if "spoiler_tag" not in df.columns:
        df["spoiler_tag"] = 0
    df = df.dropna(subset=["rating", "review_detail"])
    df["spoiler_tag"] = pd.to_numeric(df["spoiler_tag"], errors="coerce").fillna(0).astype(int)
    df["rating"]      = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])

    if FILTER_YEAR and meta_file_path and os.path.exists(meta_file_path):
        df = filter_movies_by_year(df, meta_file_path)

    df_pos = df[df["rating"] >= RATING_POS_THRESHOLD]
    df_neg = df[df["rating"] <= RATING_NEG_THRESHOLD]
    n_pos, n_neg = len(df_pos), len(df_neg)
    print(f"  [Phân phối gốc] Positive: {n_pos:,} | Negative: {n_neg:,} | Tỷ lệ: {n_pos/max(n_neg,1):.2f}:1")

    max_pos_allowed = int(n_neg * 1.5)
    if n_pos > max_pos_allowed:
        df_pos = df_pos.sample(max_pos_allowed, random_state=42)

    total_available = len(df_pos) + n_neg
    df_balanced = (
        pd.concat([df_pos, df_neg])
        .sample(min(SAMPLE_SIZE, total_available), random_state=42)
        .reset_index(drop=True)
    )
    df_balanced.to_csv(os.path.join(OUTPUT_DIR, "data_raw.csv"), index=False, encoding="utf-8-sig")

    df_balanced = df_balanced.rename(columns={"review_detail": "review"})
    df_balanced["clean_text"] = df_balanced["review"].apply(clean_text)
    df_balanced["_word_count"] = df_balanced["clean_text"].apply(count_words)
    df_balanced = df_balanced[
        df_balanced["_word_count"].between(MIN_REVIEW_WORDS, MAX_REVIEW_WORDS)
    ].drop(columns=["_word_count"]).reset_index(drop=True)

    def _assign_label(r):
        if r >= RATING_POS_THRESHOLD: return 1
        if r <= RATING_NEG_THRESHOLD: return 0
        return -1

    df_balanced["label"] = df_balanced["rating"].apply(_assign_label)
    df_svm = df_balanced[
        (df_balanced["label"] != -1) & (df_balanced["spoiler_tag"] == 0)
    ].copy()
    print(f"Phân phối nhãn:\n{df_svm['label'].value_counts().to_string()}")
    return df_balanced, df_svm


def evaluate_and_save(model, X_test, y_test, label_names=None, output_path=None):
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
    lines += ["", "--- MA TRẬN NHẦM LẪN ---", str(cm), "",
              "--- BÁO CÁO CHI TIẾT ---", report, "=" * 60]

    result_text = "\n".join(lines)
    print(result_text)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result_text)

    return {
        "accuracy": acc, "f1_weighted": f1_w, "roc_auc": roc_auc,
        "confusion_matrix": cm, "y_pred": y_pred,
    }


def build_weighted_movie_profiles(df, vectorizer, min_reviews=MIN_REVIEWS_PER_FILM):
    print("\n[5] Xây dựng Movie Profile (Weighted TF-IDF)...")
    safe_positive = df[(df["label"] == 1) & (df["spoiler_tag"] == 0)].copy()
    review_counts = safe_positive.groupby("movie").size().reset_index(name="count")
    valid_movies  = set(review_counts[review_counts["count"] >= min_reviews]["movie"])
    safe_positive = safe_positive[safe_positive["movie"].isin(valid_movies)]

    max_log = math.log(10 + 1)
    safe_positive = safe_positive.copy()
    safe_positive["weight"] = safe_positive["rating"].apply(
        lambda r: math.log(float(r) + 1) / max_log
    )

    movies       = safe_positive["movie"].unique()
    profile_rows = []
    tfidf_rows   = []

    for movie in movies:
        film_reviews = safe_positive[safe_positive["movie"] == movie]
        texts   = film_reviews["clean_text"].tolist()
        weights = film_reviews["weight"].values
        tfidf_mat   = vectorizer.transform(texts)
        weighted    = tfidf_mat.multiply(weights.reshape(-1, 1))
        profile_vec = weighted.sum(axis=0)
        tfidf_rows.append(csr_matrix(profile_vec))
        profile_rows.append({"movie": movie, "clean_text": " ".join(texts)})

    movie_profiles    = pd.DataFrame(profile_rows).reset_index(drop=True)
    movie_tfidf_matrix = vstack(tfidf_rows)
    print(f"  -> Profile cho {len(movie_profiles):,} phim.")
    return movie_profiles, movie_tfidf_matrix


def compute_precision_at_k(recommended, relevant, k):
    return sum(1 for item in recommended[:k] if item in relevant) / k if k > 0 else 0.0


def compute_ndcg_at_k(recommended, relevant, k):
    top_k = recommended[:k]
    dcg   = sum(1.0 / math.log2(i + 2) for i, item in enumerate(top_k) if item in relevant)
    n_ideal = min(len(relevant), k)
    idcg    = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    return dcg / idcg if idcg > 0 else 0.0


def compute_intra_list_diversity(recommended_titles, movie_tfidf_matrix, movie_index):
    valid_indices = [movie_index[t] for t in recommended_titles if t in movie_index]
    if len(valid_indices) < 2:
        return 0.0
    vecs = movie_tfidf_matrix[valid_indices]
    sim_matrix = cosine_similarity(vecs)
    n = len(valid_indices)
    total_sim = sum(sim_matrix[i, j] for i in range(n) for j in range(i + 1, n))
    n_pairs = n * (n - 1) / 2
    return 1.0 - (total_sim / n_pairs if n_pairs > 0 else 0.0)


def evaluate_recommendation_offline(df, movie_tfidf_matrix, movie_profiles, k_values=None):
    if k_values is None:
        k_values = EVAL_K_VALUES
    print("\n[Offline Eval] Tính Precision@K, NDCG@K, ILD...")

    if "reviewer" not in df.columns or "movie" not in df.columns:
        return {}

    known_movies = set(movie_profiles["movie"].tolist())
    df_eval      = df[df["movie"].isin(known_movies)].copy()
    movie_index  = {row["movie"]: idx for idx, row in movie_profiles.iterrows()}

    user_counts = df_eval.groupby("reviewer").size()
    eval_users  = user_counts[user_counts >= EVAL_MIN_RATINGS].index.tolist()

    if not eval_users:
        return {}

    rng         = np.random.default_rng(42)
    sample_size = min(200, len(eval_users))
    sampled     = rng.choice(eval_users, size=sample_size, replace=False)

    results = {k: {"prec": [], "ndcg": []} for k in k_values}
    ild_scores = []
    max_k = max(k_values)

    for user in sampled:
        user_df  = df_eval[df_eval["reviewer"] == user]
        relevant = set(user_df[user_df["rating"] >= RATING_POS_THRESHOLD]["movie"])
        if not relevant:
            continue
        user_idxs = [movie_index[m] for m in user_df["movie"] if m in movie_index]
        if not user_idxs:
            continue
        user_vec = np.asarray(movie_tfidf_matrix[user_idxs].mean(axis=0)).reshape(1, -1)
        sims     = cosine_similarity(user_vec, movie_tfidf_matrix).flatten()
        sorted_indices = sims.argsort()[::-1]
        recommended = [movie_profiles.iloc[i]["movie"] for i in sorted_indices[:max_k]]

        for k in k_values:
            results[k]["prec"].append(compute_precision_at_k(recommended, relevant, k))
            results[k]["ndcg"].append(compute_ndcg_at_k(recommended, relevant, k))
        ild_scores.append(compute_intra_list_diversity(recommended[:max_k], movie_tfidf_matrix, movie_index))

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
    print(f"  Intra-List Diversity = {avg_ild:.4f}")
    summary["ild"] = avg_ild
    summary["n_users_evaluated"] = len(sampled)
    return summary


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    try:
        start_time = time.time()
        print("=" * 60)
        print("  PIPELINE - FINAL CONFIG")
        print(f"  SAMPLE_SIZE={SAMPLE_SIZE:,} | SVD={SVD_N_COMPONENTS} | TF-IDF={TFIDF_MAX_FEATURES:,}")
        print("=" * 60)

        df, df_svm = run_training_pipeline(FOLDER_PATH, META_FILE_PATH)

        X_raw = df_svm["clean_text"]
        y     = df_svm["label"]

        print("\n[1] Vector hoá TF-IDF...")
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            stop_words=_STOP_WORDS,
            ngram_range=TFIDF_NGRAM_RANGE,
            sublinear_tf=True,
            analyzer="word",
            token_pattern=r"(?u)\b[a-zA-Z'][a-zA-Z']{1,}\b",
        )
        X_vectorized = vectorizer.fit_transform(X_raw)

        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42, stratify=y,
        )
        print(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

        train_dist  = pd.Series(y_train).value_counts()
        n_pos_train = train_dist.get(1, 0)
        n_neg_train = train_dist.get(0, 0)
        if n_pos_train / max(n_neg_train, 1) > 1.5:
            ros = RandomOverSampler(sampling_strategy=1 / 1.5, random_state=42)
            X_train, y_train = ros.fit_resample(X_train, y_train)

        print("\n[2] Huấn luyện LinearSVC + Calibration...")
        base_svc = LinearSVC(C=1.0, max_iter=3000, dual="auto",
                             class_weight="balanced", random_state=42)
        model = CalibratedClassifierCV(base_svc, cv=3, method="sigmoid")
        model.fit(X_train, y_train)

        print("\n[3] Đánh giá SVM...")
        metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.txt")
        eval_result  = evaluate_and_save(
            model, X_test, y_test,
            label_names=["Negative (0)", "Positive (1)"],
            output_path=metrics_path,
        )

        print("\n[4] Cross-Validation 3-fold...")
        _cv_est = CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=42), cv=3
        )
        cv_scores = cross_val_score(
            _cv_est, X_vectorized, y,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring="f1_weighted", n_jobs=-1,
        )
        print(f"  CV F1-Weighted: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(f"\n--- Cross-Validation ---\n")
            f.write(f"  CV F1-Weighted: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

        if "movie" in df.columns:
            movie_profiles, movie_tfidf_matrix = build_weighted_movie_profiles(
                df, vectorizer, min_reviews=MIN_REVIEWS_PER_FILM,
            )
            joblib.dump(movie_profiles,     os.path.join(OUTPUT_DIR, "content_movie_profiles.joblib"))
            joblib.dump(movie_tfidf_matrix, os.path.join(OUTPUT_DIR, "content_tfidf_matrix.joblib"))

            web_reviews = df[df["label"] == 1][
                ["movie", "reviewer", "rating", "review", "spoiler_tag"]
            ].copy()
            joblib.dump(web_reviews, os.path.join(OUTPUT_DIR, "web_display_reviews.joblib"))

            rec_metrics = evaluate_recommendation_offline(
                df, movie_tfidf_matrix, movie_profiles, k_values=EVAL_K_VALUES,
            )

            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 60 + "\n")
                f.write("    ĐÁNH GIÁ HỆ THỐNG GỢI Ý (Offline Evaluation)\n")
                f.write("=" * 60 + "\n")
                f.write(f"  Số user đánh giá : {rec_metrics.get('n_users_evaluated', 0):,}\n")
                for k in EVAL_K_VALUES:
                    if k in rec_metrics:
                        f.write(f"  Precision@{k:<3}     : {rec_metrics[k]['precision']:.4f}\n")
                        f.write(f"  NDCG@{k:<3}          : {rec_metrics[k]['ndcg']:.4f}\n")
                f.write(f"  Intra-List Div. : {rec_metrics.get('ild', 0):.4f}\n")

        if "reviewer" in df.columns and "movie" in df.columns:
            print("\n[6] SVD Collaborative Filtering...")
            print("  -> Lọc người dùng (>= 3 đánh giá) để tăng SVD Explained Variance...")
            user_counts_svd = df.groupby("reviewer").size()
            valid_svd_users = set(user_counts_svd[user_counts_svd >= 3].index)
            df_svd = df[df["reviewer"].isin(valid_svd_users)].copy()

            user_mapping  = {u: i for i, u in enumerate(df_svd["reviewer"].unique())}
            movie_mapping = {m: i for i, m in enumerate(df_svd["movie"].unique())}
            u_idx = df_svd["reviewer"].map(user_mapping)
            m_idx = df_svd["movie"].map(movie_mapping)
            user_item_matrix = csr_matrix(
                (df_svd["rating"].values, (u_idx.values, m_idx.values)),
                shape=(len(user_mapping), len(movie_mapping)),
            )
            svd = TruncatedSVD(n_components=SVD_N_COMPONENTS, n_iter=SVD_N_ITER, random_state=42)
            user_factors = svd.fit_transform(user_item_matrix)
            item_factors = svd.components_.T
            explained_var = svd.explained_variance_ratio_.sum()
            print(f"  -> SVD: {SVD_N_COMPONENTS} components, {explained_var*100:.1f}% explained variance.")

            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(f"\n--- SVD Collaborative Filtering ---\n")
                f.write(f"  n_components      : {SVD_N_COMPONENTS}\n")
                f.write(f"  Explained Variance: {explained_var:.4f} ({explained_var*100:.1f}%)\n")
                f.write(f"  User count        : {len(user_mapping):,}\n")
                f.write(f"  Movie count       : {len(movie_mapping):,}\n")

            joblib.dump({
                "user_map":           user_mapping,
                "movie_map":          movie_mapping,
                "user_factors":       user_factors,
                "item_factors":       item_factors,
                "explained_variance": explained_var,
            }, os.path.join(OUTPUT_DIR, "collab_svd_model.joblib"))

        print("\n[7] Lưu mô hình Sentiment...")
        joblib.dump(model,      os.path.join(OUTPUT_DIR, "model_sentiment_linearsvc.joblib"))
        joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, "vectorizer_tfidf.joblib"))

        # Cập nhật config.json với kết quả thực
        if "results" not in _config:
            _config["results"] = {}
        _config["results"].update({
            "svm_accuracy":      float(eval_result["accuracy"]),
            "f1_weighted":       float(eval_result["f1_weighted"]),
            "roc_auc":           float(eval_result["roc_auc"]) if eval_result["roc_auc"] else None,
            "cv_f1":             float(cv_scores.mean()),
            "cv_f1_std":         float(cv_scores.std()),
            "precision_at_5":    float(rec_metrics.get(5, {}).get('precision', 0)) if 'rec_metrics' in locals() else None,
            "ndcg_at_5":         float(rec_metrics.get(5, {}).get('ndcg', 0)) if 'rec_metrics' in locals() else None,
            "precision_at_10":   float(rec_metrics.get(10, {}).get('precision', 0)) if 'rec_metrics' in locals() else None,
            "ndcg_at_10":        float(rec_metrics.get(10, {}).get('ndcg', 0)) if 'rec_metrics' in locals() else None,
            "ild":               float(rec_metrics.get('ild', 0)) if 'rec_metrics' in locals() else None,
            "svd_explained_variance": float(explained_var) if 'explained_var' in locals() else None,
            "n_movies_profiled": len(movie_profiles) if 'movie_profiles' in locals() else None,
            "train_time_minutes": round((time.time() - start_time) / 60, 1),
        })
        with open(os.path.join(OUTPUT_DIR, "config.json"), "w", encoding="utf-8") as _f:
            _json.dump(_config, _f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start_time
        print(f"\n✅ V7 Hoàn tất trong {elapsed:.1f}s ({elapsed/60:.1f} phút).")
        print(f"   Output → {OUTPUT_DIR}/")

    except Exception as exc:
        import traceback
        print(f"Lỗi: {exc}")
        traceback.print_exc()
