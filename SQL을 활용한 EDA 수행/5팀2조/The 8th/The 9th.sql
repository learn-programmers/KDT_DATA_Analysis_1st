SELECT
    `Shipping Type`,
    `Frequency of Purchases`,
    COUNT(*) AS `Purchase Count`
FROM
    consumer.shopping_trends
    
-- 재구매 횟수가 평균보다 적은 소비자들 중 추출함    
WHERE
    `Previous Purchases` <= (SELECT AVG(`Previous Purchases`) FROM consumer.shopping_trends)
    
GROUP BY
    `Shipping Type`,
    `Frequency of Purchases`
ORDER BY
    `Shipping Type`,
    `Frequency of Purchases`;
