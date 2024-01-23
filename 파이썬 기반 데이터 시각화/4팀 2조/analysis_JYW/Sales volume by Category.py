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

# 카테고리별 판매량 추출 쿼리 실행
analysis_query = """with order_product_count as (
select a.order_id, product_id, sum(order_item_id) as order_item
from olist_orders_dataset a left join olist_order_items_dataset b
on a.order_id = b.order_id
where order_status != 'canceled' and order_status != 'unavailable' and product_id is not null
group by 1, 2
order by order_item desc),
portuguese_category_item as (
select product_category_name, sum(order_item) as order_item
from order_product_count a join olist_products_dataset b
on a.product_id = b.product_id
where product_category_name is not null
group by 1)
select a.product_category_name, product_category_name_english, order_item
from portuguese_category_item a left join product_category_name_translation b
on a.product_category_name = b.product_category_name
where product_category_name_english is not null
order by 3 desc"""

cursor.execute(analysis_query)



category_por = []
category_eng = []
total = []

for line in cursor:
    category_por.append(line[0])
    category_eng.append(line[1])
    total.append(int(line[2]))

    
# 막대 그래프 생성
fig, ax = plt.subplots(figsize=(15, 8))
bars = plt.barh(category_eng[:20], total[:20], color='skyblue')  # 상위 20개만 표시
ax.set_xlabel('Total Order Item Count')
ax.set_ylabel('Product Category (English)')
ax.set_title('Top 20 Product Categories by Order Item Count')
ax.invert_yaxis()  # 상위부터 표시하기 위해 y축을 역순으로 설정

# X 축 레이블 지수 형태 해제
formatter = ScalarFormatter(useOffset=False, useMathText=True)
formatter.set_scientific(False)
ax.xaxis.set_major_formatter(formatter)

# 각 막대에 값 표시
for bar, category, value in zip(bars, category_eng[:20], total[:20]):
    ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{value}', ha='left', va='center')

plt.show()
