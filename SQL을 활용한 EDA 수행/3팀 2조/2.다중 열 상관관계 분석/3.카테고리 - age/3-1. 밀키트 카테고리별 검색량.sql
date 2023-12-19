-- 10-60대 까지의 각 연령대 별 카테고리 순위
WITH shopping_age as(
    SELECT age, category, round(avg(ratio),1) as avg_ratio
    from datatable
    WHERE 1=1
    AND age is not null
    group by 1,2
),

rnk_data as(
    SELECT shopping_age.*, rank() over(PARTITION by age order by avg_ratio DESC) as rnk
    FROM shopping_age
)

SELECT rnk_data.*
from rnk_data
order by 1

