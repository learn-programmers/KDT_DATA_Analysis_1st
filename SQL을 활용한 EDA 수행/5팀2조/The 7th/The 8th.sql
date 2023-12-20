SELECT
    `Category`,
    `Payment Method`,
    `Preferred Payment Method`,
    COUNT(*) AS `Mismatched Payment Method`
FROM
    consumer.shopping_trends
WHERE
    `Payment Method` != `Preferred Payment Method`
GROUP BY
    `Category`,
    `Payment Method`,
    `Preferred Payment Method`;

