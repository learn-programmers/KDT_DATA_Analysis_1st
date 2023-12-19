-- 밀키트 카테고리별 검색량 / 성별

WITH shopping_gender as(
    --성별 카테고리 평균 ratio 값
    SELECT if(gender = 'f', '여성','남성') as gender, category, round(avg(ratio),1) as avg_ratio
    from datatable
    WHERE 1=1
    AND gender is not NULL
    group by 1,2
)

SELECT shopping_gender.*
from shopping_gender
WHERE shopping_gender.gender = '여성'
order by avg_ratio DESC