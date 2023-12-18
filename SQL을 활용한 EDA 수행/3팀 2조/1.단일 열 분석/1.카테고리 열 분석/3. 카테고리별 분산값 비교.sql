-- 카테고리별 분산
WITH differences AS (
  SELECT
    category,
    (ratio - 36.9) AS difference,
    POWER((ratio - 36.9), 2) AS squared_difference
  FROM datatable
 
)

SELECT
  category,
  round(SUM(squared_difference) / COUNT(*), 1) AS variance
FROM differences
GROUP BY 1;