-- age 별 분산 비교 
WITH differences AS (
  SELECT
    age,
    (ratio - 36.9) AS difference,
    POWER((ratio - 36.9), 2) AS squared_difference
  FROM datatable
  where age is not null 
 
)

SELECT
  age,
  round(SUM(squared_difference) / COUNT(*), 1) AS variance
FROM differences
GROUP BY 1;