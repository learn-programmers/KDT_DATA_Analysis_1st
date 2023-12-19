-- gender 별 분산 비교 
WITH differences AS (
  SELECT
    gender,
    (ratio - 36.9) AS difference,
    POWER((ratio - 36.9), 2) AS squared_difference
  FROM datatable
  where gender is not null 
 
)

SELECT
  gender,
  round(SUM(squared_difference) / COUNT(*), 1) AS variance
FROM differences
GROUP BY 1;