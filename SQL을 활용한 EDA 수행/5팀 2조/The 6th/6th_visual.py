import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
file_path_sub_yes = 'C:\\Users\\tndud\\바탕 화면\\6th_Sub_yes.csv'
file_path_sub_no = 'C:\\Users\\tndud\\바탕 화면\\6th_Sub_No.csv'
data_sub_yes = pd.read_csv(file_path_sub_yes)
data_sub_no = pd.read_csv(file_path_sub_no)

# 평점에 따른 카테고리별 제품 개수 시각화 및 개발할 제품 카운팅을 위한 함수 정의
def visualize_and_count_products(data, subscription_status):
    # 평점 구분
    rating_categories = ['High', 'Medium', 'Low']
    if 'Bad' in data.columns:
        rating_categories.append('Bad')

    # 개발할 제품 카운팅 (Low 이하 평점 제품)
    low_rating_count = data['Low'].sum()
    if 'Bad' in data.columns:
        low_rating_count += data['Bad'].sum()

    # 시각화
    plt.figure(figsize=(10, 6))
    data.set_index('Category')[rating_categories].plot(kind='bar', stacked=True)
    plt.title(f'Product Ratings for {subscription_status} Subscription')
    plt.xlabel('Category')
    plt.ylabel('Number of Products')
    plt.legend(title='Rating')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return low_rating_count

# 구독 'Yes' 데이터 시각화 및 개발할 제품 카운팅
low_rating_count_yes = visualize_and_count_products(data_sub_yes, 'Yes')

# 구독 'No' 데이터 시각화 및 개발할 제품 카운팅
low_rating_count_no = visualize_and_count_products(data_sub_no, 'No')

# 개발할 제품 수 출력
print(f"Total products to be developed for 'Yes' subscription: {low_rating_count_yes}")
print(f"Total products to be developed for 'No' subscription: {low_rating_count_no}")
