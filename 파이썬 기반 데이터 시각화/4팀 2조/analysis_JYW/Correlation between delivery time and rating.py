import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import mysql.connector
import scipy.stats as stats


# MySQL 연결 설정
mysql_host = '****'
mysql_user = '****'
mysql_password = '****'
mysql_database = '****'


# MySQL 연결 설정
connection = mysql.connector.connect(
    host=mysql_host,
    user=mysql_user,
    password=mysql_password,
    database=mysql_database
)


# 커서 생성
cursor = connection.cursor()


# 배송기간과 평점 추출 쿼리 실행
analysis_query = """select review_score, DATEDIFF(DATE(order_delivered_customer_date), DATE(order_approved_at)) AS date_sub
from olist_orders_dataset a join olist_order_reviews_dataset b
on a.order_id = b.order_id
where order_approved_at is not null
and order_delivered_customer_date is not null
and order_status != 'canceled' and order_status != 'unavailable'
and review_score is not null"""


cursor.execute(analysis_query)


review_score = []
date = []


for line in cursor:
    review_score.append(int(line[0]))
    date.append(int(line[1]))


data = {
    'review_score': review_score,
    'delivery_time': date
}


# 데이터프레임 생성
df = pd.DataFrame(data)


# 상관 관계 분석
correlation = df['review_score'].corr(df['delivery_time'])


# 상관 관계 분석
correlation, p_value = stats.pearsonr(df['review_score'], df['delivery_time'])

print(f"상관 계수: {correlation}")
print(f"p-value: {p_value}")


# 유의수준 0.05에서의 유의성 검정
alpha = 0.05
if p_value < alpha:
    print("귀무가설을 기각하며, 두 변수 간에 통계적으로 유의한 상관 관계가 있다.")
else:
    print("귀무가설을 기각하지 못하며, 두 변수 간에 통계적으로 유의한 상관 관계가 없다.")




# 산점도 그리기
plt.scatter(df['delivery_time'], df['review_score'])
plt.title('Delivery Time vs Review Score')
plt.xlabel('Delivery Time')
plt.ylabel('Review Score')
plt.show()


# 커넥션 및 커서 닫기
cursor.close()
connection.close()
