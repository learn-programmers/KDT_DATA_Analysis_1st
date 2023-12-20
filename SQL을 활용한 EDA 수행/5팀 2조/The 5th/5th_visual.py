import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
file_path_1 = 'C:\\Users\\tndud\\바탕 화면\\5th.csv'
file_path_2 = 'C:\\Users\\tndud\\바탕 화면\\5th_2.csv'
data_1 = pd.read_csv(file_path_1)
data_2 = pd.read_csv(file_path_2)

# 두 데이터셋 합치기
merged_data = pd.concat([data_1, data_2])

# 계절별로 데이터를 분리
seasons = merged_data['Season'].unique()

# 시각화
plt.figure(figsize=(20, 10))
for i, season in enumerate(seasons):
    plt.subplot(2, 2, i + 1)
    season_data = merged_data[merged_data['Season'] == season]
    season_gender_item = season_data.groupby(['Gender', 'Item_Purchased']).agg({'Total_Purchase_Amount':'sum'}).reset_index()
    sns.barplot(x='Item_Purchased', y='Total_Purchase_Amount', hue='Gender', data=season_gender_item)
    plt.title(f'{season} Season Item Purchases')
    plt.xticks(rotation=45)
    plt.xlabel("Item")
    plt.ylabel("Total Purchase Amount")
    plt.tight_layout()

plt.show()
