import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 문제 해결을 위한 설정 (필요한 경우)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
file_path = 'C:\\Users\\tndud\\바탕 화면\\4th.csv'
data = pd.read_csv(file_path)

# 남성용 아이템 추천 수준 분류 함수
def classify_male_items(row):
    diff = row['differ_male_female_per']
    if diff >= 0.9:
        return '강력추천'
    elif 0.2 <= diff < 0.9:
        return '추천'
    elif 0.0 <= diff < 0.2:
        return '추천_약한'
    else:
        return '추천하지않음'

# 여성용 아이템 추천 수준 분류 함수
def classify_female_items(row):
    diff = row['differ_female_male_per']
    if diff >= 0.9:
        return '강력추천'
    elif 0.2 <= diff < 0.9:
        return '추천'
    elif 0.0 <= diff < 0.2:
        return '추천_약한'
    else:
        return '추천하지않음'

# 남성과 여성 데이터에 대해 추천 수준을 분류
data['Male Recommendation Level'] = data.apply(classify_male_items, axis=1)
data['Female Recommendation Level'] = data.apply(classify_female_items, axis=1)

# 남성과 여성 추천 아이템 데이터 분리
male_data = data[data['Male_Cnt'] > 1]
female_data = data[data['Female_Cnt'] > 1]

# 추천 수준별 아이템 시각화 준비
male_recommendation_data = male_data.groupby('Male Recommendation Level')['Item Purchased'].value_counts().unstack().fillna(0)
female_recommendation_data = female_data.groupby('Female Recommendation Level')['Item Purchased'].value_counts().unstack().fillna(0)


# 남성 추천 아이템 시각화
plt.figure(figsize=(12, 8))
sns.heatmap(male_recommendation_data, annot=True, cmap="Blues")
plt.title("Recommendation Levels for Male Items")
plt.xlabel("Items")
plt.ylabel("Recommendation Level")
plt.show()

# 여성 추천 아이템 시각화
plt.figure(figsize=(12, 8))
sns.heatmap(female_recommendation_data, annot=True, cmap="Reds")
plt.title("Recommendation Levels for Female Items")
plt.xlabel("Items")
plt.ylabel("Recommendation Level")
plt.show()
