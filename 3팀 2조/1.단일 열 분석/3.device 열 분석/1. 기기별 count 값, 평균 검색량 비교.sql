-- 기기에 따른 평균 ratio, count 값 비교 
SELECT device, ROUND(AVG(ratio)) AS AVG_RATIO, count(*) as count
FROM datatable
WHERE device IS NOT null
GROUP BY device