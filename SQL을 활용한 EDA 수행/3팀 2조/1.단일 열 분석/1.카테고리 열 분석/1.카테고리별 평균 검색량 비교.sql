-- 카테고리별 avg_ratio
SELECT category, ROUND(AVG(ratio),1)
FROM datatable
GROUP BY 1
ORDER BY 2 DESC