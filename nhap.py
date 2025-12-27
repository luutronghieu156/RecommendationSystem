import matplotlib.pyplot as plt
import pandas as pd

# Đọc dữ liệu từ file CSV
data = pd.read_csv('data/anime.csv')  
# Hiển thị thông tin tổng quan về dữ liệu
print(data.info())
# Hiển thị 5 dòng đầu tiên của dữ liệu

print(data.head())
