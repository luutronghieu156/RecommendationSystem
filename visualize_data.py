import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

# 1. Cấu hình giao diện biểu đồ
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 2. Tạo folder visualization nếu chưa tồn tại
output_folder = 'visualization'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Đã tạo thư mục: {output_folder}")

# 3. Đọc dữ liệu
# Giả sử file nằm cùng thư mục với script code
try:
    anime_df = pd.read_csv('cleaned_data/anime_cleaned.csv')
    rating_df = pd.read_csv('cleaned_data/rating_scored.csv')
    print("Đã load dữ liệu thành công!")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file csv. Hãy đảm bảo file nằm cùng thư mục.")
    # Dừng chương trình nếu không có file (để demo logic)
    anime_df = pd.DataFrame() 
    rating_df = pd.DataFrame()

if not anime_df.empty and not rating_df.empty:
    
    # --- CHART 1: Phân bố điểm đánh giá của người dùng (User Ratings) ---
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rating', data=rating_df, palette='viridis')
    plt.title('Phân bố điểm đánh giá từ người dùng (User Ratings Distribution)')
    plt.xlabel('Điểm (Rating)')
    plt.ylabel('Số lượng đánh giá')
    plt.savefig(os.path.join(output_folder, '1_user_rating_distribution.png'))
    plt.close()

    # --- CHART 2: Phân bố điểm đánh giá trung bình của Anime ---
    plt.figure(figsize=(10, 6))
    sns.histplot(anime_df['rating'], bins=30, kde=True, color='skyblue')
    plt.title('Phân bố điểm trung bình của các Anime')
    plt.xlabel('Điểm trung bình (Average Rating)')
    plt.ylabel('Số lượng Anime')
    plt.savefig(os.path.join(output_folder, '2_anime_avg_rating_distribution.png'))
    plt.close()

    # --- CHART 3: Số lượng Anime theo Loại (TV, Movie, OVA,...) ---
    plt.figure(figsize=(10, 6))
    type_counts = anime_df['type'].value_counts()
    sns.barplot(x=type_counts.index, y=type_counts.values, palette='magma')
    plt.title('Số lượng Anime theo từng loại hình (Type)')
    plt.xlabel('Loại (Type)')
    plt.ylabel('Số lượng')
    plt.savefig(os.path.join(output_folder, '3_anime_count_by_type.png'))
    plt.close()

    # --- CHART 4: Top 10 Anime có nhiều thành viên nhất (Popularity) ---
    top_members = anime_df.sort_values(by='members', ascending=False).head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(y='name', x='members', data=top_members, palette='coolwarm')
    plt.title('Top 10 Anime phổ biến nhất (theo số lượng thành viên)')
    plt.xlabel('Số lượng thành viên')
    plt.ylabel('Tên Anime')
    plt.tight_layout() # Giúp nhãn không bị cắt
    plt.savefig(os.path.join(output_folder, '4_top_10_popular_anime.png'))
    plt.close()

    # --- CHART 5: Top 10 Anime có điểm đánh giá cao nhất (Lọc > 10000 thành viên để tránh nhiễu) ---
    popular_anime = anime_df[anime_df['members'] > 10000]
    top_rated = popular_anime.sort_values(by='rating', ascending=False).head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(y='name', x='rating', data=top_rated, palette='Spectral')
    plt.title('Top 10 Anime điểm cao nhất (trên 10k thành viên)')
    plt.xlabel('Điểm đánh giá')
    plt.xlim(8, 10) # Zoom vào khoảng điểm cao
    plt.ylabel('Tên Anime')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '5_top_10_highest_rated_anime.png'))
    plt.close()

    # --- CHART 6: Boxplot so sánh điểm Rating giữa các loại Anime ---
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='type', y='rating', data=anime_df, palette='Set2')
    plt.title('So sánh phân bố điểm Rating theo loại Anime')
    plt.xlabel('Loại (Type)')
    plt.ylabel('Điểm Rating')
    plt.savefig(os.path.join(output_folder, '6_rating_boxplot_by_type.png'))
    plt.close()

    # --- CHART 7: Heatmap tương quan giữa các biến số (Rating, Members, Episodes) ---
    plt.figure(figsize=(8, 6))
    corr_matrix = anime_df[['rating', 'members', 'episodes']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap tương quan giữa Rating, Members và Episodes')
    plt.savefig(os.path.join(output_folder, '7_correlation_heatmap.png'))
    plt.close()

    # --- CHART 8: Top 10 Thể loại (Genre) phổ biến nhất ---
    # Xử lý chuỗi genre (với các genre ngăn cách bởi dấu phẩy)
    all_genres = []
    for genres in anime_df['genre'].dropna():
        all_genres.extend([g.strip() for g in genres.split(',')])
    
    genre_counts = Counter(all_genres).most_common(10)
    genres, counts = zip(*genre_counts)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(counts), y=list(genres), palette='Blues_r')
    plt.title('Top 10 Thể loại (Genre) xuất hiện nhiều nhất')
    plt.xlabel('Số lượng Anime')
    plt.ylabel('Thể loại')
    plt.savefig(os.path.join(output_folder, '8_top_10_genres.png'))
    plt.close()

    # --- CHART 9: Scatter plot quan hệ giữa Độ phổ biến (Members) và Điểm số (Rating) ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='members', y='rating', data=anime_df, alpha=0.5, color='purple')
    plt.title('Quan hệ giữa Số lượng thành viên và Điểm đánh giá')
    plt.xlabel('Số lượng thành viên')
    plt.ylabel('Điểm đánh giá')
    plt.xscale('log') # Dùng log scale vì số lượng member chênh lệch lớn
    plt.savefig(os.path.join(output_folder, '9_scatter_members_vs_rating.png'))
    plt.close()

    # --- CHART 10: Histogram hoạt động của User (Mỗi user đánh giá bao nhiêu phim) ---
    user_activity = rating_df['user_id'].value_counts()
    plt.figure(figsize=(10, 6))
    # Chỉ lấy những user đánh giá dưới 200 phim để biểu đồ dễ nhìn hơn (loại bỏ outlier quá lớn)
    sns.histplot(user_activity[user_activity < 200], bins=50, color='orange', kde=True)
    plt.title('Phân bố số lượng đánh giá của mỗi User (User Activity)')
    plt.xlabel('Số lượng phim đã đánh giá')
    plt.ylabel('Số lượng User')
    plt.savefig(os.path.join(output_folder, '10_user_activity_histogram.png'))
    plt.close()

    print(f"Hoàn tất! 10 biểu đồ đã được lưu trong thư mục '{output_folder}'.")

else:
    print("Không thể vẽ biểu đồ do thiếu dữ liệu.")