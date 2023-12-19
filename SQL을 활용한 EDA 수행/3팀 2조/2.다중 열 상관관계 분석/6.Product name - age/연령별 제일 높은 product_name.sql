-- 연령별 제일 높은 product_name
WITH AvgRatioByAge AS (
  SELECT
    age, product_name, COUNT(ratio) AS count_ratio
  FROM
    datatable
  WHERE
    age IS NOT NULL
  GROUP BY
    age, product_name
)

SELECT
  age, product_name, count_ratio
FROM (
  SELECT
    age, product_name, count_ratio,
    ROW_NUMBER() OVER (PARTITION BY age ORDER BY count_ratio DESC) AS row_num
  FROM
    AvgRatioByAge
) AS ranked
WHERE
  row_num = 1
ORDER BY
  age;