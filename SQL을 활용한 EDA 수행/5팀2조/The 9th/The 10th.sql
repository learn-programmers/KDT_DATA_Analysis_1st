SELECT
    Category,
    CASE
        WHEN Age BETWEEN 10 AND 19 THEN '10s'
        WHEN Age BETWEEN 20 AND 29 THEN '20s'
        WHEN Age BETWEEN 30 AND 39 THEN '30s'
        WHEN Age BETWEEN 40 AND 49 THEN '40s'
        WHEN Age BETWEEN 50 AND 59 THEN '50s'
        WHEN Age BETWEEN 60 AND 69 THEN '60s'
        WHEN Age >= 70 THEN '70s'
        ELSE 'Unknown'
    END AS `Age Group`,
    AVG(`Previous Purchases`) AS `Avg Previous Purchases`,
    COUNT(*) AS `Total Count`
FROM
    shopping_trends
WHERE
    `Subscription Status` = 'YES' -- 'NO'로도 바꿔서
GROUP BY
    Category, `Age Group`
ORDER BY
    Category, `Age Group`;
