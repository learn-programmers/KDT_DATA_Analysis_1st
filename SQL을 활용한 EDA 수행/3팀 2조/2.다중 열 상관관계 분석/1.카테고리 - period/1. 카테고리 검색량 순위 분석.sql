-- 밀키트 테마별 검색량 / 날짜별

WITH shopping_date as(--월별 카테고리 평균 ratio 값
    SELECT strftime('%Y-%m',period)as date
    ,category, avg(ratio) as avg_ratio
    from datatable
    WHERE 1=1
    group by 1,2
)

SELECT shopping_date.*, rank() over(partition By date order BY avg_ratio DESC) as rank
from shopping_date
order by date, avg_ratio DESC;