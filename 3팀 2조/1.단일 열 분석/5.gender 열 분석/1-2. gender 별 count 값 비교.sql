-- 성별에 따른 키워드 정렬 count 값
WITH category_avg AS (
    SELECT 
        category, 
        gender, 
        round(avg(ratio)) as avg_ratio,
        count(*) AS count
    FROM datatable 
    GROUP BY category, gender
)

SELECT 
    c1.category, c1.count as male_count, c2.count as female_count
FROM category_avg c1
JOIN category_avg c2 ON c1.category = c2.category 
WHERE c1.gender = 'm'
AND c2.gender = 'f'
ORDER BY c1.category;