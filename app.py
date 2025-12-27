import streamlit as st
from streamlit_option_menu import option_menu
import model 
import os

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="Anime Recommendation System", layout="wide", page_icon="📊")

# --- 2. TÙY CHỈNH GIAO DIỆN (GIỮ NGUYÊN MÀU EMERALD & SLATE) ---
st.markdown("""
    <style>
        /* Nền Navy sâu */
        .stApp { background-color: #0a192f; color: #ccd6f6; }
        
        /* Sidebar/Navbar Slate Dark */
        [data-testid="stSidebar"] { 
            background-color: #020c1b !important; 
            border-right: 1px solid #10b981; 
        }
        
        /* Tiêu đề Emerald sáng */
        h1, h2, h3 { color: #10b981 !important; font-family: 'Inter', sans-serif; }
        p, span, label { color: #8892b0 !important; }

        /* Metric Cards */
        div[data-testid="stMetric"] {
            background-color: #112240;
            border: 1px solid #233554;
            border-radius: 10px;
            padding: 15px;
        }
        
        /* Màu số liệu Metric */
        div[data-testid="stMetricValue"] > div { color: #10b981 !important; }

        /* Slider */
        .stSlider [data-baseweb="slider"] { color: #10b981; }
        
        /* Hộp thông báo Info/Success */
        .stAlert {
            background-color: #172a45;
            color: #10b981;
            border: 1px solid #10b981;
        }
    </style>
    """, unsafe_allow_html=True)

# --- 3. KHỞI TẠO DỮ LIỆU ---
@st.cache_resource
def init_engine():
    return model.load_all_data()

anime_df, rating_df, tfidf, content_matrix, knn_content, sparse_rating_matrix, knn_item = init_engine()

# --- 4. TỪ ĐIỂN MÔ TẢ (NGÔN NGỮ CHUYÊN NGÀNH/KỸ THUẬT) ---
insight_data = {
    "user_rating": "Biểu đồ phân phối điểm số (User Rating Distribution) cho thấy dữ liệu lệch phải, tập trung ở mức 7.0 - 8.0, phản ánh xu hướng đánh giá tích cực của người dùng.",
    "anime_avg_rating": "Điểm trung bình (Mean Rating) của các bộ Anime tuân theo phân phối chuẩn với trung vị khoảng 6.5. Tỷ lệ phim đạt điểm >8.5 là rất thấp (ngoại lệ tích cực).",
    "type": "Thống kê số lượng theo định dạng: TV Series chiếm tỷ trọng lớn nhất trong kho dữ liệu, theo sau là OVA và Movie.",
    "popular": "Top 10 Anime có số lượng thành viên (Members) cao nhất, đại diện cho mức độ phổ biến và nhận diện thương hiệu trong cộng đồng.",
    "rated": "Top 10 Anime có điểm số trung bình (Weighted Score) cao nhất với điều kiện số lượng thành viên > 10,000.",
    "boxplot": "Biểu đồ hộp (Boxplot) so sánh phân phối điểm số giữa các định dạng: Movie có dải điểm hẹp và trung vị cao hơn so với TV Series.",
    "heatmap": "Ma trận tương quan (Correlation Matrix): Tương quan dương mạnh giữa số lượng thành viên (Members) và điểm số (Rating).",
    "genres": "Tần suất xuất hiện của các thể loại: Comedy và Action là hai nhãn (labels) phổ biến nhất trong tập dữ liệu.",
    "scatter": "Biểu đồ phân tán (Scatter Plot) giữa Members và Rating giúp nhận diện các điểm dữ liệu ngoại lai (Outliers) tiềm năng.",
    "activity": "Phân phối tần suất hoạt động của người dùng (User Activity), cho thấy độ thưa (Sparsity) của ma trận tương tác người dùng - vật phẩm."
}

# --- 5. SIDEBAR NAVBAR ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>⛩️ ANIME MOVIE</h2>", unsafe_allow_html=True)
    st.markdown("---")
    selected = option_menu(
        menu_title=None, 
        options=["Tổng quan Hệ thống", "Trực quan hóa Dữ liệu", "Hệ thống Gợi ý", "Tìm kiếm Nội dung"],
        icons=["grid-fill", "pie-chart-fill", "cpu-fill", "search"], 
        default_index=0,
        styles={
            "container": {"background-color": "#020c1b", "padding": "5px"},
            "icon": {"color": "#10b981", "font-size": "20px"}, 
            "nav-link": {
                "font-size": "15px", 
                "color": "#8892b0", 
                "text-align": "left", 
                "margin": "8px 0px",
                "padding": "12px"
            },
            "nav-link-selected": {
                "background-color": "#10b981", 
                "color": "#ffffff",            
                "font-weight": "600"
            },
        }
    )

# --- 6. LOGIC TỪNG TRANG ---

if selected == "Tổng quan Hệ thống":
    st.markdown("## 📊 Dashboard Tổng quan")
    st.write("Thống kê mô tả bộ dữ liệu Anime")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng số Anime", f"{len(anime_df):,}")
    c2.metric("Tổng lượt đánh giá", f"{len(rating_df):,}")
    c3.metric("Số lượng Người dùng", f"{len(rating_df['user_id'].unique()):,}")
    c4.metric("Điểm trung bình", round(anime_df['rating'].mean(), 2))
    
    st.markdown("### 📋 Dữ liệu mẫu")
    st.dataframe(anime_df.head(100), use_container_width=True)

elif selected == "Trực quan hóa Dữ liệu":
    st.markdown("## 📈 Phân tích Trực quan hóa")
    vis_path = "visualization"
    
    if os.path.exists(vis_path):
        images = sorted([f for f in os.listdir(vis_path) if f.endswith(('.png', '.jpg'))])
        if images:
            # Format tên tab: Bỏ số thứ tự, viết hoa chữ cái đầu
            tab_titles = []
            for img in images:
                name_clean = img.split('.')[0]
                parts = name_clean.split('_')
                if parts[0].isdigit():
                    name_clean = "_".join(parts[1:])
                tab_titles.append(name_clean.replace('_', ' ').title())
            
            tabs = st.tabs(tab_titles)
            
            for i, img_name in enumerate(images):
                with tabs[i]:
                    img_key_base = img_name.split('.')[0].lower()
                    found_insight = None
                    
                    # Logic so khớp từ khóa
                    for key, text in insight_data.items():
                        if key in img_key_base:
                            found_insight = text
                            break
                    
                    if found_insight:
                        st.info(f"💡 **Phân tích:** {found_insight}")
                    else:
                        st.warning(f"Chưa có mô tả cho biểu đồ: {img_name}")
                    
                    st.markdown("---")
                    st.image(os.path.join(vis_path, img_name), use_container_width=True)
        else:
            st.warning("Thư mục visualization không chứa tệp hình ảnh.")
    else:
        st.error(f"Không tìm thấy đường dẫn thư mục: {vis_path}")

elif selected == "Hệ thống Gợi ý":
    st.markdown("## ⚙️ Hệ thống Gợi ý (Hybrid Filtering)")
    st.write("Kết hợp thuật toán Collaborative Filtering (KNN) và Content-based Filtering.")
    
    target = st.selectbox("Chọn phim đầu vào:", anime_df['name'].values)
    limit = st.slider("Số lượng kết quả trả về:", 5, 50, 10)
    
    if st.button("Tìm gợi ý"):
        res = model.get_recommendations(target, anime_df, knn_content, content_matrix, knn_item, sparse_rating_matrix, top_n=limit)
        st.success(f"Kết quả gợi ý dựa trên sự tương đồng với **{target}**:")
        st.dataframe(res, use_container_width=True)

elif selected == "Tìm kiếm Nội dung":
    st.markdown("## 🔍 Tìm kiếm theo Từ khóa")
    st.write("Truy vấn dựa trên vector đặc trưng văn bản (TF-IDF).")
    
    q = st.text_input("Nhập từ khóa mô tả (VD: Samurai, Cyberpunk...):")
    q_limit = st.slider("Giới hạn hiển thị:", 5, 50, 10)
    
    if q:
        res = model.get_recommendations_by_query(q, anime_df, tfidf, knn_content, top_n=q_limit)
        st.dataframe(res, use_container_width=True)