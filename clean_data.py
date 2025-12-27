import pandas as pd
import numpy as np
import os

# Tạo folder để lưu dữ liệu đã làm sạch
output_folder = 'cleaned_data'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Đã tạo folder: {output_folder}")

# ===================== ANIME.CSV =====================
print("\n" + "="*50)
print("ĐỌC VÀ LÀM SẠCH ANIME.CSV")
print("="*50)

anime_df = pd.read_csv('data/anime.csv')
print(f"\nSố dòng ban đầu: {len(anime_df)}")
print(f"\nThông tin dữ liệu:")
print(anime_df.info())
print(f"\nSố giá trị null mỗi cột:")
print(anime_df.isnull().sum())
print(f"\nSố giá trị trùng lặp: {anime_df.duplicated().sum()}")

# Làm sạch anime_df
# 1. Xóa dòng trùng lặp
anime_cleaned = anime_df.drop_duplicates()

# 2. Xử lý giá trị null/missing
# - genre: thay null bằng 'Unknown'
anime_cleaned['genre'] = anime_cleaned['genre'].fillna('Unknown')

# - type: thay null bằng 'Unknown'
anime_cleaned['type'] = anime_cleaned['type'].fillna('Unknown')

# - episodes: thay 'Unknown' bằng NaN, chuyển sang numeric
anime_cleaned['episodes'] = pd.to_numeric(anime_cleaned['episodes'], errors='coerce')

# - rating: xóa các dòng có rating null (không có giá trị đánh giá)
anime_cleaned = anime_cleaned.dropna(subset=['rating'])

# 3. Xử lý HTML entities (&#039; -> ')
anime_cleaned['name'] = anime_cleaned['name'].str.replace('&#039;', "'", regex=False)
anime_cleaned['name'] = anime_cleaned['name'].str.replace('&amp;', '&', regex=False)
anime_cleaned['name'] = anime_cleaned['name'].str.replace('&lt;', '<', regex=False)
anime_cleaned['name'] = anime_cleaned['name'].str.replace('&gt;', '>', regex=False)
anime_cleaned['name'] = anime_cleaned['name'].str.replace('&quot;', '"', regex=False)

# 4. Đảm bảo kiểu dữ liệu đúng
anime_cleaned['anime_id'] = anime_cleaned['anime_id'].astype(int)
anime_cleaned['rating'] = anime_cleaned['rating'].astype(float)
anime_cleaned['members'] = anime_cleaned['members'].astype(int)

print(f"\nSố dòng sau khi làm sạch: {len(anime_cleaned)}")
print(f"Số dòng đã loại bỏ: {len(anime_df) - len(anime_cleaned)}")

# ===================== RATING.CSV =====================
print("\n" + "="*50)
print("ĐỌC VÀ LÀM SẠCH RATING.CSV")
print("="*50)

rating_df = pd.read_csv('data/rating.csv')
print(f"\nSố dòng ban đầu: {len(rating_df)}")
print(f"\nThông tin dữ liệu:")
print(rating_df.info())
print(f"\nSố giá trị null mỗi cột:")
print(rating_df.isnull().sum())
print(f"\nSố giá trị trùng lặp: {rating_df.duplicated().sum()}")
print(f"\nPhân bố rating:")
print(rating_df['rating'].value_counts().sort_index())

# Làm sạch rating_df
# 1. Xóa dòng trùng lặp
rating_cleaned = rating_df.drop_duplicates()

# 2. Xóa các dòng có giá trị null
rating_cleaned = rating_cleaned.dropna()

# 3. Chuyển đổi rating = -1 thành NaN (user đã xem nhưng chưa đánh giá)
# Hoặc có thể giữ lại tùy mục đích sử dụng
# Option A: Giữ lại -1
# Option B: Loại bỏ rating = -1 (chỉ giữ rating thực sự)
rating_with_score = rating_cleaned[rating_cleaned['rating'] != -1].copy()

# 4. Chỉ giữ các rating liên kết với anime còn tồn tại
valid_anime_ids = set(anime_cleaned['anime_id'])
rating_cleaned = rating_cleaned[rating_cleaned['anime_id'].isin(valid_anime_ids)]
rating_with_score = rating_with_score[rating_with_score['anime_id'].isin(valid_anime_ids)]

# 5. Đảm bảo kiểu dữ liệu đúng
rating_cleaned['user_id'] = rating_cleaned['user_id'].astype(int)
rating_cleaned['anime_id'] = rating_cleaned['anime_id'].astype(int)
rating_cleaned['rating'] = rating_cleaned['rating'].astype(int)

rating_with_score['user_id'] = rating_with_score['user_id'].astype(int)
rating_with_score['anime_id'] = rating_with_score['anime_id'].astype(int)
rating_with_score['rating'] = rating_with_score['rating'].astype(int)

print(f"\nSố dòng sau khi làm sạch (bao gồm -1): {len(rating_cleaned)}")
print(f"Số dòng sau khi làm sạch (chỉ rating thực): {len(rating_with_score)}")
print(f"Số dòng đã loại bỏ: {len(rating_df) - len(rating_cleaned)}")

# ===================== LƯU DỮ LIỆU =====================
print("\n" + "="*50)
print("LƯU DỮ LIỆU ĐÃ LÀM SẠCH")
print("="*50)

# Lưu anime đã làm sạch
anime_output = os.path.join(output_folder, 'anime_cleaned.csv')
anime_cleaned.to_csv(anime_output, index=False, encoding='utf-8')
print(f"✓ Đã lưu: {anime_output}")

# Lưu rating đã làm sạch (bao gồm cả -1)
rating_output = os.path.join(output_folder, 'rating_cleaned.csv')
rating_cleaned.to_csv(rating_output, index=False, encoding='utf-8')
print(f"✓ Đã lưu: {rating_output}")

# Lưu rating chỉ có điểm thực (không có -1)
rating_scored_output = os.path.join(output_folder, 'rating_scored.csv')
rating_with_score.to_csv(rating_scored_output, index=False, encoding='utf-8')
print(f"✓ Đã lưu: {rating_scored_output}")

# ===================== THỐNG KÊ TỔNG KẾT =====================
print("\n" + "="*50)
print("THỐNG KÊ TỔNG KẾT")
print("="*50)

print(f"\n📊 ANIME:")
print(f"   - Tổng số anime: {len(anime_cleaned)}")
print(f"   - Số thể loại (type): {anime_cleaned['type'].nunique()}")
print(f"   - Rating trung bình: {anime_cleaned['rating'].mean():.2f}")
print(f"   - Rating cao nhất: {anime_cleaned['rating'].max()}")
print(f"   - Rating thấp nhất: {anime_cleaned['rating'].min():.2f}")

print(f"\n👥 RATING:")
print(f"   - Tổng số đánh giá (bao gồm -1): {len(rating_cleaned)}")
print(f"   - Tổng số đánh giá thực: {len(rating_with_score)}")
print(f"   - Số user: {rating_cleaned['user_id'].nunique()}")
print(f"   - Số anime được đánh giá: {rating_cleaned['anime_id'].nunique()}")

print(f"\n📁 Files đã tạo trong folder '{output_folder}':")
for f in os.listdir(output_folder):
    file_path = os.path.join(output_folder, f)
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"   - {f} ({size_mb:.2f} MB)")

print("\n✅ HOÀN THÀNH!")

