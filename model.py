import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# --- 1. LOAD DATA & MODEL (Đóng gói để App gọi) ---
def load_all_data():
    anime_df = pd.read_csv('cleaned_data/anime_cleaned.csv')
    rating_df = pd.read_csv('cleaned_data/rating_scored.csv')

    # Xử lý Content-based
    anime_df['genre_fixed'] = anime_df['genre'].fillna('').str.replace(',', ' ')
    anime_df['metadata_soup'] = (
        anime_df['name'].astype(str) + " " + 
        anime_df['genre_fixed'] + " " + 
        anime_df['type'].fillna('')
    )

    tfidf = TfidfVectorizer(stop_words='english')
    content_matrix = tfidf.fit_transform(anime_df['metadata_soup'])
    knn_content = NearestNeighbors(metric='cosine', algorithm='brute').fit(content_matrix)

    # Xử lý Item-based
    anime_id_to_idx = {id: i for i, id in enumerate(anime_df['anime_id'])}
    user_id_to_idx = {id: i for i, id in enumerate(rating_df['user_id'].unique())}
    row = rating_df['anime_id'].map(anime_id_to_idx).values
    col = rating_df['user_id'].map(user_id_to_idx).values
    data = rating_df['rating'].values

    sparse_rating_matrix = csr_matrix((data, (row, col)), shape=(len(anime_df), len(user_id_to_idx)))
    knn_item = NearestNeighbors(metric='cosine', algorithm='brute').fit(sparse_rating_matrix)

    return anime_df, rating_df, tfidf, content_matrix, knn_content, sparse_rating_matrix, knn_item

# --- 2. HÀM GỢI Ý (Sửa tham số để nhận biến từ App truyền vào) ---

def get_recommendations(target_anime_name, anime_df, knn_content, content_matrix, knn_item, sparse_rating_matrix, top_n=10, weight_content=0.5):
    try:
        idx = anime_df[anime_df['name'] == target_anime_name].index[0]
    except IndexError:
        return None

    dist_c, idx_c = knn_content.kneighbors(content_matrix[idx], n_neighbors=top_n+5)
    dist_i, idx_i = knn_item.kneighbors(sparse_rating_matrix[idx], n_neighbors=top_n+5)

    scores = {}
    for d, i in zip(dist_c.flatten(), idx_c.flatten()):
        if i == idx: continue
        scores[i] = scores.get(i, 0) + (1 - d) * weight_content

    for d, i in zip(dist_i.flatten(), idx_i.flatten()):
        if i == idx: continue
        scores[i] = scores.get(i, 0) + (1 - d) * (1 - weight_content)

    top_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return anime_df.iloc[[x[0] for x in top_indices]][['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members']]

def get_recommendations_by_query(query_text, anime_df, tfidf, knn_content, top_n=10):
    query_vector = tfidf.transform([query_text])
    distances, indices = knn_content.kneighbors(query_vector, n_neighbors=top_n)
    cols_to_show = ['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members']
    return anime_df.iloc[indices.flatten()][cols_to_show]