-- 나이대별 평균 ratio, count 값 비교
SELECT age, ROUND(AVG(ratio)) AS AVG_RATIO, count(*) as count
FROM datatable
WHERE age IS NOT null
GROUP BY age
order by 2 desc, 3 desc