SELECT product_name
FROM datatable
WHERE gender = 'f'
GROUP BY product_name
ORDER BY AVG(ratio) DESC
LIMIT 10;