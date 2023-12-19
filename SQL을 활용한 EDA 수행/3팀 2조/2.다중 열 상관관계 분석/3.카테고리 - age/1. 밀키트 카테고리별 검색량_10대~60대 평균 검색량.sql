-- 밀키트 카테고리별 검색량 / 나이별

WITH shopping_age as(
    --나이별별 카테고리 평균 ratio 값
    SELECT age, category, round(avg(ratio),1) as avg_ratio
    from datatable
    WHERE 1=1
    AND age is not null
    group by 1,2
),

rnk_data as(
    --나이별 랭킹화
    SELECT shopping_age.*, rank() over(PARTITION by age order by avg_ratio DESC) as rnk
    FROM shopping_age
)

SELECT rnk_data.*
from rnk_data
WHERE rnk = 1
order by 1