SELECT gender, ROUND(AVG(ratio)) AS AVG_RATIO , count(*) as count
FROM datatable
WHERE gender IS NOT null
GROUP BY genderSELECT gender, ROUND(AVG(ratio)) AS AVG_RATIO 
FROM datatable
WHERE gender IS NOT null
GROUP BY gender