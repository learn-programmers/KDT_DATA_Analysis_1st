-- 카테고리별 기기 사용값 비교
SELECT 
    category,
    SUM(CASE WHEN device = 'pc' THEN 1 ELSE 0 END) AS pc_count,
    SUM(CASE WHEN device = 'mo' THEN 1 ELSE 0 END) AS mo_count
FROM datatable
GROUP BY category;