import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import mysql.connector
import scipy.stats as stats


# # MySQL 연결 설정
# mysql_host = '****'
# mysql_user = '****'
# mysql_password = '****'
# mysql_database = '****'


# MySQL 연결 설정
connection = mysql.connector.connect(
    host=mysql_host,
    user=mysql_user,
    password=mysql_password,
    database=mysql_database
)

# 커서 생성
cursor = connection.cursor()


# 카테고리와 평점 추출 쿼리 실행
analysis_query = """with order_product_count as (
select a.order_id, product_id, sum(order_item_id) as order_item
from olist_orders_dataset a join olist_order_items_dataset b
on a.order_id = b.order_id
where order_status != 'canceled' and order_status != 'unavailable' and product_id is not null
group by 1, 2
order by order_item desc),
order_product_count_review as
(select a.order_id, product_id, sum(order_item) as order_item, avg(review_score) as avg_review_score
from order_product_count a join olist_order_reviews_dataset b
on a.order_id = b.order_id
where review_score is not null
group by 1, 2
order by order_item desc),
portuguese_category_item as (
select product_category_name, sum(order_item) as order_item, avg(avg_review_score) as avg_review_score
from order_product_count_review a join olist_products_dataset b
on a.product_id = b.product_id
where product_category_name is not null
group by 1)
select a.product_category_name, product_category_name_english, order_item, round(avg_review_score, 2) as avg_review_score
from portuguese_category_item a join product_category_name_translation b
on a.product_category_name = b.product_category_name
where product_category_name_english is not null
order by 3 desc;"""

cursor.execute(analysis_query)


item_count = []
reivew_score = []


for line in cursor:
    item_count.append(float(line[2]))
    reivew_score.append(float(line[3]))


data = {
    'item_count': item_count,
    'reivew_score': reivew_score
}

# 데이터프레임 생성
df = pd.DataFrame(data)


# 상관 계수 분석
correlation, p_value = stats.pearsonr(df['item_count'], df['reivew_score'])

print(f"상관 계수: {correlation}")
print(f"p-value: {p_value}")

# 유의수준 0.05에서의 유의성 검정
alpha = 0.05
if p_value < alpha:
    print("귀무가설을 기각하며, 두 변수 간에 통계적으로 유의한 상관 관계가 있다.")
else:
    print("귀무가설을 기각하지 못하며, 두 변수 간에 통계적으로 유의한 상관 관계가 없다.")


# 산점도 그리기
plt.scatter(df['reivew_score'], df['item_count'])
plt.title('reivew_score vs item_count')
plt.xlabel('reivew_score')
plt.ylabel('item_count')
plt.show()


# 커넥션 및 커서 닫기
cursor.close()
connection.close()
