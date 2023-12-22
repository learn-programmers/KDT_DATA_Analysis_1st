import pandas as pd
import matplotlib.pyplot as plt

# 데이터를 불러옵니다.
data = pd.read_csv('C:/Users/82103/Desktop/과제/KDT_DA/billboard/popular_genre.csv')
years = data['Year']
genres = data['spotify_genre']

unique_genres = {year: genre for year, genre in zip(years,genres)}

# 장르별 출현 빈도 계산
genre_counts = {}
for genre in unique_genres.values():
    if genre in genre_counts:
        genre_counts[genre] += 1
    else:
        genre_counts[genre] = 1

# 막대 그래프 그리기
plt.figure(figsize=(12, 6))
plt.bar(genre_counts.keys(), genre_counts.values(), color='skyblue')
plt.title('Frequency of Most Popular Genres on Billboard Hot 100')
plt.xlabel('Genre')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()