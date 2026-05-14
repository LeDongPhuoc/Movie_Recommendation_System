import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import re
import nltk
from nltk.corpus import stopwords
import time
import requests
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import plotly.express as px

import joblib  
import os
# --- CẤU HÌNH ---
st.set_page_config(page_title="Sent-AI Pro: Theory to Practice", page_icon="🧠", layout="wide")

# CSS làm đẹp
st.markdown("""
    <style>
    .stApp {background-color: #f4f6f9;}
    .metric-card {
        background-color: white; padding: 15px; border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- HÀM HỖ TRỢ  ---
@st.cache_resource 
def setup_resources():
    try: nltk.data.find('corpora/stopwords')
    except LookupError: nltk.download('stopwords')
    
    stop_words = stopwords.words('english')
    negative_words = ['not', 'no', 'nor', "n't", "don't", "isn't", "aren't", "couldn't",
                    "didn't", "doesn't", "hadn't", "hasn't", "haven't", "isn't",
                    "shouldn't", "won't", "wouldn't", "never", "hardly", "nothing", "nowhere"]
    
    for word in negative_words:
        if word in stop_words:
            stop_words.remove(word)
            
    return stop_words

stop_words = setup_resources()

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r"[^a-zA-Z\s']", '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_lottieurl(url):
    try: return requests.get(url).json()
    except: return None

#Cập nhập thêm để lưu dữ liệu vector phân tích
def save_model(model, vectorizer, vis_data):
    try:
        joblib.dump(model, 'svm_model.joblib')
        joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
        joblib.dump(vis_data, 'vis_data.joblib')
    except Exception as e:
        st.error(f"Lỗi khi lưu model: {e}")

def load_model_from_file():
    if os.path.exists('svm_model.joblib') and os.path.exists('tfidf_vectorizer.joblib') and os.path.exists('vis_data.joblib'):
        try:
            model = joblib.load('svm_model.joblib')
            vec = joblib.load('tfidf_vectorizer.joblib')
            vis_data = joblib.load('vis_data.joblib')
            return model, vec, vis_data
        except Exception as e:
            return None, None, None
    return None, None, None

# --- QUẢN LÝ STATE
# if 'model' not in st.session_state: st.session_state.model = None # Model SVM
# if 'vectorizer' not in st.session_state: st.session_state.vectorizer = None 
# if 'is_trained' not in st.session_state: st.session_state.is_trained = False 
# if 'train_data' not in st.session_state: st.session_state.train_data = None 
# if 'metrics' not in st.session_state: st.session_state.metrics = None # Kết quả đánh giá
if 'model' not in st.session_state: 
    # [CẬP NHẬT] Thử tải từ file trước khi gán None
    loaded_model, loaded_vec, loaded_vis = load_model_from_file()
    if loaded_model is not None:
        st.session_state.model = loaded_model
        st.session_state.vectorizer = loaded_vec
        st.session_state.train_data = loaded_vis
        st.session_state.is_trained = True
    else:
        st.session_state.model = None
        st.session_state.vectorizer = None
        st.session_state.train_data = None
        st.session_state.is_trained = False

if 'vectorizer' not in st.session_state: 
    # Nếu model đã load được thì vectorizer cũng đã có ở trên, ngược lại thì None
    if st.session_state.model is None:
        st.session_state.vectorizer = None 

if 'is_trained' not in st.session_state: 
    st.session_state.is_trained = (st.session_state.model is not None)

if 'train_data' not in st.session_state: st.session_state.train_data = None 
if 'metrics' not in st.session_state: st.session_state.metrics = None 
if 'history' not in st.session_state: st.session_state.history = [] # Lịch sử
if 'result_export' not in st.session_state: st.session_state.result_export = None # Kết quả xuất file

# --- MENU ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80) 
    st.title("Sent-AI Bridge")
    selected = option_menu(
        menu_title=None,
        options=["Trang Chủ", "Huấn Luyện Model", "Phân Tích Vector", "Bản Đồ Dữ Liệu Thực", "Phân Tích & Demo"],
        icons=["house", "gear", "code-slash", "map", "play-btn"],
        default_index=0
    )

# --- 1. TRANG CHỦ ---
if selected == "Trang Chủ":
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.title("Từ Lý Thuyết Đến Ứng Dụng")
        st.markdown("""
        Chào mừng bạn đến với hệ thống minh họa đồ án **SVM Sentiment Analysis**.
        
        Ứng dụng này giải quyết câu hỏi: **"Làm sao áp dụng công thức vào văn bản thực tế?"**
        
        ### Quy trình khép kín:
        1.  **Input:** Văn bản thô (IMDB Reviews).
        2.  **Vector hóa (TF-IDF):** Biến văn bản thành các vector trong không gian nhiều chiều (tương tự điểm $x$ trong lý thuyết).
        3.  **SVM Kernel:** Tìm siêu phẳng phân tách trong không gian cao chiều đó.
        4.  **Minh họa:** Dùng PCA để chiếu dữ liệu thực tế xuống 2D để mắt thường nhìn thấy được.
        """)
    with col2:
        lottie = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_0yfsb3a1.json")
        if lottie: st_lottie(lottie, height=300)

# --- 2. HUẤN LUYỆN  ---
elif selected == "Huấn Luyện Model":
    st.header("🛠️ Huấn Luyện & Đánh Giá Hiệu Năng")
    
    # Lịch sử huấn luyện
    if 'history' not in st.session_state: 
        st.session_state.history = []

    col_conf, col_report = st.columns([1, 2])
    
    with col_conf:
        uploaded_file = st.file_uploader("Upload file 'IMDB Dataset.csv':", type=['csv'])
        limit = st.number_input("Số lượng mẫu train:", 500, 50000, 2000)
        
        st.write("**Cấu hình N-gram (Cụm từ):**")
        c_ngram1, c_ngram2 = st.columns(2)
        with c_ngram1:
            min_n = st.number_input("Min N:", 1, 5, 1)
        with c_ngram2:
            max_n = st.number_input("Max N:", 1, 5, 2)
        
        ngram_range = (min_n, max_n)
        
        if st.button("🚀 Bắt đầu Huấn Luyện", type="primary"):
            if uploaded_file and max_n >= min_n:
                try:
                    # BẮT ĐẦU ĐẾM GIỜ
                    t_start = time.time()
                    # 1. Load & Preprocess
                    with st.spinner("Đang xử lý dữ liệu..."):
                        file_name = uploaded_file.name

                        df = pd.read_csv(uploaded_file).sample(limit, random_state=42).reset_index(drop=True)
                        df['clean'] = df['review'].apply(preprocess_text)
                        df['label'] = df['sentiment'].apply(lambda x: 1 if x=='positive' else 0)
                        
                        # 2. Vectorization
                        max_feat = 3000 + (max_n - 1) * 2000 
                        vec = TfidfVectorizer(max_features=max_feat, stop_words=stop_words, ngram_range=ngram_range)
                        X = vec.fit_transform(df['clean'])
                        y = df['label']
                        
                        # 3. Train SVM
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model = SVC(kernel='linear', probability=True) 
                        model.fit(X_train, y_train)

                        # KẾT THÚC ĐẾM GIỜ
                        t_end = time.time()
                        training_duration = t_end - t_start
                        
                        # 4. Đánh giá chi tiết
                        y_pred = model.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        prec = precision_score(y_test, y_pred)
                        rec = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        cm = confusion_matrix(y_test, y_pred)
                        
                        # --- LƯU LỊCH SỬ (NEW) ---
                        run_log = {
                            "Thời gian chạy": time.strftime("%H:%M:%S"),
                            "Dữ liệu (Dòng)": limit,
                            "N-gram": f"{min_n}-{max_n}",
                            "Thời gian Train (s)": round(training_duration, 4),
                            "Accuracy": f"{acc:.2%}",
                            "F1-Score": f"{f1:.2%}"
                        }
                        st.session_state.history.append(run_log)

                        
                        # Dự đoán lại toàn bộ tập dữ liệu để xuất báo cáo
                        all_preds = model.predict(X)
                        df['Prediction'] = ['Positive' if p==1 else 'Negative' for p in all_preds]
                        export_df = df[['review', 'sentiment', 'clean', 'Prediction']]
                        st.session_state.result_export = export_df
                        
                        # Lưu State
                        st.session_state.model = model
                        st.session_state.vectorizer = vec
                        st.session_state.is_trained = True
                        st.session_state.train_time = training_duration
                        st.session_state.metrics = {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'cm': cm}
                        
                        # 5. Chuẩn bị dữ liệu vẽ bản đồ
                        reducer = TruncatedSVD(n_components=3)
                        n_vis = min(500, len(y_test))
                        X_vis = reducer.fit_transform(X_test[:n_vis])
                        vis_df = pd.DataFrame(X_vis, columns=['x', 'y', 'z'])
                        vis_df['label_str'] = ['Positive' if l==1 else 'Negative' for l in y_test.iloc[:n_vis]]
                        vis_df['text'] = df.loc[y_test.index[:n_vis], 'review'].values
                        st.session_state.train_data = vis_df

                        save_model(model, vec, vis_df)
                        
                    st.success("Huấn luyện thành công!")
                    st.success(f"Huấn luyện xong trong {training_duration:.4f} giây!")
                except Exception as e:
                    st.error(f"Có lỗi xảy ra: {e}")
            elif not uploaded_file:
                st.error("Vui lòng upload file CSV.")

        # --- HIỂN THỊ BẢNG LỊCH SỬ (NEW) ---
        if len(st.session_state.history) > 0:
            st.markdown("---")
            st.subheader("📜 Nhật ký các lần chạy")
            # Chuyển list thành DataFrame để hiển thị đẹp
            df_hist = pd.DataFrame(st.session_state.history)
            # Đảo ngược để lần chạy mới nhất lên đầu
            st.dataframe(df_hist.iloc[::-1], use_container_width=True)
            
            if st.button("Xóa lịch sử"):
                st.session_state.history = []
                st.rerun()

    with col_report:
        if st.session_state.is_trained and st.session_state.metrics:
            metrics = st.session_state.metrics
            
            st.subheader("📊 Kết quả Đánh giá")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{metrics['acc']:.1%}") 
            c2.metric("Precision", f"{metrics['prec']:.1%}")
            c3.metric("Recall", f"{metrics['rec']:.1%}")
            c4.metric("F1-Score", f"{metrics['f1']:.1%}")
                        
            st.markdown("---")
            st.write("**Ma trận Nhầm lẫn:**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=ax)
            st.pyplot(fig)

            if st.session_state.result_export is not None:
                st.markdown("---")
                csv = st.session_state.result_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Tải xuống kết quả chi tiết (.csv)",
                    data=csv,
                    file_name='imdb_sentiment_results.csv',
                    mime='text/csv',
                    help="Tải file Excel chứa: Review gốc, Nhãn gốc, Clean Text, và Dự đoán của máy."
                )

        else:
            st.info("Kết quả đánh giá sẽ hiện ở đây sau khi huấn luyện.")

# --- 3. Phân tích VECTOR ---
elif selected == "Phân Tích Vector":
    st.header("🔍 Giải mã: Máy tính 'nhìn' văn bản như thế nào?")
    st.write("Mục này giúp bạn hiểu sự liên kết giữa bài toán lý thuyết ($x_1, x_2$) và bài toán thực tế.")
    
    if not st.session_state.is_trained:
        st.warning("Vui lòng huấn luyện mô hình trước.")
    else:
        col_text, col_vec = st.columns(2)
        
        with col_text:
            user_text = st.text_area("Nhập một câu review ngắn:", "This movie is absolutely terrible and boring")
            if st.button("Phân tích Vector"):
                clean_text = preprocess_text(user_text)
                
                # 1. Biến đổi (Tính x)
                vec_op = st.session_state.vectorizer.transform([clean_text])
                
                # 2. Lấy dữ liệu
                feature_names = st.session_state.vectorizer.get_feature_names_out()
                indices = vec_op.nonzero()[1]
                values = vec_op.data
                
                # 3. Tạo dữ liệu bảng (Thêm w và x*w)
                data_list = []
                sum_contribution = 0
                
                for i, idx in enumerate(indices):
                    row = {
                        "Từ vựng (Feature)": feature_names[idx],
                        "Giá trị TF-IDF (x)": values[i],
                        "Chỉ số (Index)": idx
                    }
                    
                    if st.session_state.model.kernel == 'linear':
                        w = st.session_state.model.coef_.toarray()[0][idx]
                        row["Trọng số (w)"] = w
                        row["Tích (x*w)"] = values[i] * w
                        sum_contribution += row["Tích (x*w)"]
                    else:
                        row["Trọng số (w)"] = 0
                        row["Tích (x*w)"] = 0
                        
                    data_list.append(row)
                
                st.session_state.vec_data = pd.DataFrame(data_list).sort_values(by="Giá trị TF-IDF (x)", ascending=False)
                st.session_state.current_sum = sum_contribution
                if hasattr(st.session_state.model, 'intercept_'):
                    st.session_state.current_bias = st.session_state.model.intercept_[0]
                else:
                    st.session_state.current_bias = 0

        with col_vec:
            if 'vec_data' in st.session_state and not st.session_state.vec_data.empty:
                st.write("### Biểu diễn Toán học:")
                
                format_dict = {
                    "Giá trị TF-IDF (x)": "{:.4f}",
                    "Trọng số (w)": "{:.4f}",
                    "Tích (x*w)": "{:.4f}"
                }
                
                st.dataframe(
                    st.session_state.vec_data.style.format(format_dict, na_rep=""), 
                    use_container_width=True
                )
                
                if st.session_state.model.kernel == 'linear':
                    bias = st.session_state.current_bias
                    total = st.session_state.current_sum
                    st.info(f"""
                    🧮 **Thử tính tay:**
                    1. Tổng cột "Tích (x*w)": **{total:.4f}**
                    2. Cộng Bias ($b$): **{bias:.4f}**
                    3. Kết quả = **{total + bias:.4f}**
                    """)
            elif 'vec_data' in st.session_state:
                st.info("Không tìm thấy từ nào trong từ điển (Có thể do Stopwords hoặc từ hiếm).")

# --- 4. BẢN ĐỒ DỮ LIỆU THỰC ---
elif selected == "Bản Đồ Dữ Liệu Thực":
    st.header("🗺️ Hình dung Dữ liệu IMDB trong không gian")
    
    if st.session_state.train_data is None:
        st.warning("Vui lòng huấn luyện mô hình trước.")
    else:
        df_vis = st.session_state.train_data
        
        tab2d, tab3d = st.tabs(["Không gian 2D", "Không gian 3D"])
        
        with tab2d:
            if len(df_vis) > 0:
                fig = px.scatter(
                    df_vis, x='x', y='y', color='label_str',
                    hover_data=['text'],
                    color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'},
                    title="Dữ liệu Review sau khi giảm chiều (2D Projection)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Dữ liệu trực quan hóa bị rỗng.")

        with tab3d:
            if len(df_vis) > 0:
                fig3 = px.scatter_3d(
                    df_vis, x='x', y='y', z='z', color='label_str',
                    color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'},
                    opacity=0.7
                )
                st.plotly_chart(fig3, use_container_width=True)

# --- 5. PHÂN TÍCH & DEMO ---
elif selected == "Phân Tích & Demo":
    st.header("🔮 Demo Ứng dụng")
    
    if not st.session_state.is_trained:
        st.warning("Chưa có mô hình.")
    
        
    else:
        tab_single, tab_batch = st.tabs(["✍️ Dự đoán Đơn lẻ (Gốc)", "📂 Dự đoán từ File (Mới)"])

        with tab_single:
            txt = st.text_input("Review phim (Tiếng Anh):", "I really loved the acting, it was amazing!")
            
            if txt:
                # 1. Tiền xử lý & Vector hóa
                vec = st.session_state.vectorizer.transform([preprocess_text(txt)])
                
                # 2. Dự đoán
                pred = st.session_state.model.predict(vec)[0]
                decision_val = st.session_state.model.decision_function(vec)[0] 
                proba = st.session_state.model.predict_proba(vec)[0]
                
                # 3. Hiển thị kết quả
                c1, c2 = st.columns(2)
                with c1:
                    if pred == 1:
                        st.success(f"### 😊 POSITIVE ({proba[1]:.1%})")
                    else:
                        st.error(f"### 😡 NEGATIVE ({proba[0]:.1%})")
                    
                    st.metric("Kết quả Máy tính (f(x))", f"{decision_val:.4f}", help="So sánh số này với số bạn tính tay ở tab Cầu Nối")
                
                with c2:
                    if st.session_state.model.kernel == 'linear':
                        st.write("#### Tại sao AI chọn vậy?")
                        feature_names = st.session_state.vectorizer.get_feature_names_out()
                        coefs = st.session_state.model.coef_.toarray()[0]
                        indices = vec.nonzero()[1]
                        word_contributions = []
                        for i in indices:
                            word = feature_names[i]
                            weight = coefs[i]
                            word_contributions.append((word, weight))
                        word_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

                        for word, weight in word_contributions:
                            # Logic hiển thị màu sắc dựa trên trọng số
                            if weight > 0.05:
                                st.markdown(f"- **{word}**: :green[+{weight:.4f}] (Kéo về Positive)")
                            elif weight < -0.05:
                                st.markdown(f"- **{word}**: :red[{weight:.4f}] (Kéo về Negative)")
                            else:
                                st.markdown(f"- {word}: :grey[{weight:.4f}] (Trung tính)")
        with tab_batch:
            st.subheader("Tải lên file CSV chứa nhiều bình luận")
            st.markdown("File cần có cột tên là **`review`**.")
            
            uploaded_batch = st.file_uploader("Chọn file CSV:", type=['csv'], key="batch_uploader")
            
            if uploaded_batch:
                try:
                    df_new = pd.read_csv(uploaded_batch)
                    if 'review' not in df_new.columns:
                        st.error("❌ File không có cột 'review'.")
                    else:
                        if st.button("Chạy Phân Tích Hàng Loạt"):
                            with st.spinner("Đang xử lý..."):
                                start_batch = time.time() # Đếm giờ cho batch luôn
                                
                                clean_texts = df_new['review'].apply(preprocess_text)
                                X_new = st.session_state.vectorizer.transform(clean_texts)
                                preds = st.session_state.model.predict(X_new)
                                probas = st.session_state.model.predict_proba(X_new)
                                
                                end_batch = time.time()
                                
                                df_new['AI_Label'] = ['Positive' if p==1 else 'Negative' for p in preds]
                                df_new['Confidence'] = [max(pr) for pr in probas]
                                
                                st.success(f"✅ Xử lý {len(df_new)} dòng trong {end_batch - start_batch:.2f}s")
                                
                                # Thống kê & Download
                                col_b1, col_b2 = st.columns(2)
                                with col_b1:
                                    st.bar_chart(df_new['AI_Label'].value_counts())
                                with col_b2:
                                    st.dataframe(df_new.head())
                                    
                                csv_batch = df_new.to_csv(index=False).encode('utf-8')
                                st.download_button("📥 Tải kết quả về máy", csv_batch, "batch_result.csv", "text/csv")
                                
                except Exception as e:
                    st.error(f"Lỗi: {e}")    

