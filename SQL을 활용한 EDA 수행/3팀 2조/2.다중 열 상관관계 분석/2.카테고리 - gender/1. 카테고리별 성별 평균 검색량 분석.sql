-- 밀키트 카테고리별 키워드 트랜드 / 나이별

WITH keyword_age as(
    --나이, 카테고리별 키워드 평균 ratio 값
    SELECT age, category, product_name, round(avg(ratio),1) as avg_ratio
    from datatable
    WHERE 1=1
    AND age is NOT NULL
    group by 1,2,3
)

SELECT keyword_age.*
from keyword_age
order by age, category ,avg_ratio DESC