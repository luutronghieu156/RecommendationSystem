import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import os
from datetime import datetime
import sys

# --- CẤU HÌNH FULL POWER ---
# 1. Dữ liệu: LẤY HẾT (None)
MAX_ROWS = None         
MIN_VOTES = 20          # Vẫn cần lọc nhẹ để tránh User ảo/spam làm nhiễu Model
TEST_SIZE = 0.2         

# 2. Model
K_NEIGHBORS = 20        
HYBRID_WEIGHT = 0.5     
THRESHOLD = 7.0         

# 3. Đánh giá (Test trên mẫu đại diện để không treo máy)
# Model đã học hết data, nhưng lúc thi chỉ cần chấm 100 bài là biết giỏi hay dốt.
SAMPLE_TEST_RMSE = 1000 # Test sai số trên 1000 rating
SAMPLE_TEST_RANK = 100  # Test gợi ý trên 100 User (đủ để số liệu ổn định)
TOP_K = 10              

# 4. Lưu trữ
RESULT_FILE = 'evaluation_results_final.csv'


# --- 1. CHUẨN BỊ DỮ LIỆU (FULL) ---
def prepare_data():
    print(f"⏳ [1/4] Đang tải TOÀN BỘ dữ liệu...")
    try:
        anime_df = pd.read_csv('cleaned_data/anime_cleaned.csv')
        rating_df = pd.read_csv('cleaned_data/rating_scored.csv')
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file dữ liệu."); sys.exit(1)

    print(f"   - Dữ liệu gốc: {len(rating_df):,} ratings.")

    # Lọc User/Anime ít tương tác (Bắt buộc để Matrix không bị lỗi bộ nhớ)
    # Giữ lại những user/anime có ít nhất MIN_VOTES tương tác
    anime_counts = rating_df['anime_id'].value_counts()
    rating_df = rating_df[rating_df['anime_id'].isin(anime_counts[anime_counts >= MIN_VOTES].index)]
    
    user_counts = rating_df['user_id'].value_counts()
    rating_df = rating_df[rating_df['user_id'].isin(user_counts[user_counts >= MIN_VOTES].index)]
    
    # Metadata Content
    valid_ids = rating_df['anime_id'].unique()
    anime_df = anime_df[anime_df['anime_id'].isin(valid_ids)].copy()
    
    anime_df['soup'] = (anime_df['name'].astype(str) + " " + 
                        anime_df['genre'].fillna('').str.replace(',', ' ') + " " + 
                        anime_df['type'].fillna(''))
    
    tfidf = TfidfVectorizer(stop_words='english')
    content_matrix = tfidf.fit_transform(anime_df['soup'])

    # Chia Train/Test (80/20 trên toàn bộ dữ liệu)
    train, test = train_test_split(rating_df, test_size=TEST_SIZE, random_state=42)
    print(f"   -> Dữ liệu sạch đưa vào mô hình: {len(rating_df):,} ratings")
    print(f"      (Train: {len(train):,}, Test: {len(test):,})")
    
    return anime_df, rating_df, train, test, content_matrix


# --- 2. HUẤN LUYỆN (FULL TRAIN SET) ---
def train_models(train_df, full_rating_df, content_matrix, anime_df):
    print("🤖 [2/4] Đang huấn luyện mô hình (Có thể mất 1-2 phút)...")
    
    u_ids = full_rating_df['user_id'].unique()
    a_ids = anime_df['anime_id'].unique()
    u_map = {id: i for i, id in enumerate(u_ids)}
    a_map = {id: i for i, id in enumerate(a_ids)}

    # A. Content-based
    knn_content = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_content.fit(content_matrix)

    # B. Item-based (Cốt lõi)
    # Lấy toàn bộ dữ liệu trong tập Train
    train_valid = train_df[train_df['anime_id'].isin(a_ids) & train_df['user_id'].isin(u_ids)]
    row = train_valid['anime_id'].map(a_map).values
    col = train_valid['user_id'].map(u_map).values
    data = train_valid['rating'].values
    
    sparse_matrix = csr_matrix((data, (row, col)), shape=(len(a_ids), len(u_ids)))
    
    # Train KNN (Phần này tốn RAM nhất)
    try:
        knn_item = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=K_NEIGHBORS)
        knn_item.fit(sparse_matrix)
    except MemoryError:
        print("❌ LỖI TRÀN RAM! Dữ liệu quá lớn so với máy tính.")
        print("   Giải pháp: Tăng MIN_VOTES lên 50 hoặc 100 để giảm bớt dữ liệu.")
        sys.exit(1)
    
    return knn_content, knn_item, sparse_matrix, a_map, u_map


# --- 3. ĐÁNH GIÁ (TEST TRÊN MẪU) ---
def evaluate(test_df, knn_content, content_matrix, knn_item, sparse_matrix, a_map, u_map):
    print(f"📉 [3/4] Đang chấm điểm...")

    # --- A. RMSE & MAE ---
    y_true, y_pred = [], []
    # Test trên mẫu ngẫu nhiên (chứ không chạy hết cả triệu dòng test)
    rmse_sample = test_df.sample(min(SAMPLE_TEST_RMSE, len(test_df)), random_state=42)
    
    for _, row in rmse_sample.iterrows():
        try:
            if row['user_id'] not in u_map or row['anime_id'] not in a_map: continue
            u, a = u_map[row['user_id']], a_map[row['anime_id']]
            
            dists, idxs = knn_item.kneighbors(sparse_matrix[a], n_neighbors=K_NEIGHBORS+1)
            neighbor_ratings = sparse_matrix[:, u].toarray().flatten()[idxs.flatten()[1:]]
            valid = neighbor_ratings[neighbor_ratings > 0]
            
            if len(valid) > 0:
                y_true.append(row['rating'])
                y_pred.append(np.mean(valid))
        except: pass
        
    rmse = math.sqrt(mean_squared_error(y_true, y_pred)) if y_true else 0
    mae = mean_absolute_error(y_true, y_pred) if y_true else 0

    # --- B. RANKING ---
    metrics = {'Content-based': [], 'Hybrid': []}
    
    test_users = test_df['user_id'].unique()
    # Chỉ đánh giá trên 100 User ngẫu nhiên (để không phải đợi cả ngày)
    target_users = np.random.choice(test_users, size=min(SAMPLE_TEST_RANK, len(test_users)), replace=False)

    for i, u_id in enumerate(target_users):
        print(f"   -> Testing User {i+1}/{len(target_users)}...", end='\r')
        if u_id not in u_map: continue
        u_idx = u_map[u_id]
        
        # Ground Truth
        liked_real = set(test_df[(test_df['user_id']==u_id) & (test_df['rating']>=THRESHOLD)]['anime_id'])
        if not liked_real: continue
        
        # Train History
        history = sparse_matrix[:, u_idx].toarray().flatten()
        input_idxs = np.where(history >= THRESHOLD)[0]
        if len(input_idxs) == 0: continue
        
        # Helper: Get Scores
        def get_sim_scores(model, matrix):
            scores = {}
            # Lấy tối đa 5 phim thích nhất để tìm gợi ý (Tối ưu tốc độ)
            for idx in input_idxs[:5]:
                try:
                    d, n = model.kneighbors(matrix[idx], n_neighbors=10)
                    for dist, n_idx in zip(d[0], n[0]): 
                        scores[n_idx] = scores.get(n_idx, 0) + (1-dist)
                except: pass
            return scores

        # Tính điểm
        s_content = get_sim_scores(knn_content, content_matrix)
        s_item = get_sim_scores(knn_item, sparse_matrix)
        
        s_hybrid = {}
        all_candidates = set(s_content.keys()) | set(s_item.keys())
        for k in all_candidates:
            s_hybrid[k] = (s_content.get(k, 0) * (1-HYBRID_WEIGHT)) + (s_item.get(k, 0) * HYBRID_WEIGHT)
            
        # Map Index -> Anime ID (Cần thiết để so sánh chính xác)
        anime_ids_list = list(a_map.keys())
        
        for name, score_dict in zip(['Content-based', 'Hybrid'], [s_content, s_hybrid]):
            top_idxs = [k for k, v in sorted(score_dict.items(), key=lambda x: x[1], reverse=True) if history[k]==0][:TOP_K]
            rec_ids = [anime_ids_list[idx] for idx in top_idxs if idx < len(anime_ids_list)]
            
            hits = len(set(rec_ids) & liked_real)
            metrics[name].append((hits/TOP_K, hits/len(liked_real)))

    print("\n")
    final_metrics = {}
    for k, v in metrics.items():
        if v: final_metrics[k] = {'P': np.mean([x[0] for x in v]), 'R': np.mean([x[1] for x in v])}
        else: final_metrics[k] = {'P': 0, 'R': 0}

    return rmse, mae, final_metrics


# --- 4. LƯU KẾT QUẢ ---
def save_results(rmse, mae, metrics):
    print(f"💾 [4/4] Lưu kết quả vào '{RESULT_FILE}'...")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = []
    
    # 1. Hybrid
    data.append({'Timestamp': timestamp, 'Model': 'Hybrid', 'Metric': 'RMSE', 'Value': round(rmse, 4)})
    data.append({'Timestamp': timestamp, 'Model': 'Hybrid', 'Metric': 'MAE', 'Value': round(mae, 4)})
    data.append({'Timestamp': timestamp, 'Model': 'Hybrid', 'Metric': 'Precision@10', 'Value': round(metrics['Hybrid']['P'], 4)})
    data.append({'Timestamp': timestamp, 'Model': 'Hybrid', 'Metric': 'Recall@10', 'Value': round(metrics['Hybrid']['R'], 4)})

    # 2. Content-based
    data.append({'Timestamp': timestamp, 'Model': 'Content-based', 'Metric': 'Precision@10', 'Value': round(metrics['Content-based']['P'], 4)})
    data.append({'Timestamp': timestamp, 'Model': 'Content-based', 'Metric': 'Recall@10', 'Value': round(metrics['Content-based']['R'], 4)})
    
    df = pd.DataFrame(data)
    mode = 'a' if os.path.exists(RESULT_FILE) else 'w'
    df.to_csv(RESULT_FILE, mode=mode, header=(mode=='w'), index=False)
    
    print("✅ Hoàn tất!")
    print(df[['Model', 'Metric', 'Value']].to_string(index=False))

# --- MAIN ---
if __name__ == "__main__":
    anime_df, rating_df, train, test, content_mat = prepare_data()
    knn_content, knn_item, sparse_mat, a_map, u_map = train_models(train, rating_df, content_mat, anime_df)
    rmse, mae, metrics = evaluate(test, knn_content, content_mat, knn_item, sparse_mat, a_map, u_map)
    save_results(rmse, mae, metrics)