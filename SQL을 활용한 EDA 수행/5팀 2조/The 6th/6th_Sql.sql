-- SELECT Category
-- FROM Shopping_Behavior_updated
-- GROUP BY Category;

-- CREATE TABLE Product_Manage (
-- 	Category VARCHAR(255),
--     Subscription_Status VARCHAR(255),
--     High INT,
--     Medium INT,
--     Low INT,
--     Bad INT
-- );
-- INSERT INTO Product_Manage (Category, Subscription_Status, High, Medium, Low, Bad)
-- SELECT
--     Category,
--     Subscription_Status,
--     COUNT(CASE WHEN Review_Rating >= 4.5 THEN 1 END) AS High,
--     COUNT(CASE WHEN Review_Rating < 4.5 AND Review_Rating >= 3.5 THEN 1 END) AS Medium,
--     COUNT(CASE WHEN Review_Rating < 3.5 AND Review_Rating >= 2.5 THEN 1 END) AS Low,
--     COUNT(CASE WHEN Review_Rating < 2.5 THEN 1 END) AS Bad
-- FROM shopping_behavior_updated
-- GROUP BY Category, Subscription_Status;


-- SELECT * FROM Product_Manage;
WITH Product_Review_Summary AS (
    SELECT
        Category,
        Subscription_Status,
        COUNT(CASE WHEN Review_Rating >= 4 THEN 1 END) AS High,
        COUNT(CASE WHEN Review_Rating >= 3 AND Review_Rating < 4 THEN 1 END) AS Medium,
        COUNT(CASE WHEN Review_Rating < 3 THEN 1 END) AS Low,
        COUNT(CASE WHEN Review_Rating < 2 THEN 1 END) AS Bad
    FROM shopping_behavior_updated
    GROUP BY Category, Subscription_Status
),
High_Rating_Percentage AS (
    SELECT
        Category,
        Subscription_Status,
        High,
        Medium,
        Low,
        (High / (High + Medium + Low)) * 100 AS High_Rating_Percent
    FROM Product_Review_Summary
),
Max_High_Rating AS (
    SELECT
        Category,
        'Yes' AS Subscription_Status,
        MAX(High) AS Max_High
    FROM Product_Review_Summary
    WHERE Subscription_Status = 'Yes'
    GROUP BY Category
),
Max_Low_Rating AS (
    SELECT
        Category,
        'No' AS Subscription_Status,
        MAX(Low) AS Max_Low
    FROM Product_Review_Summary
    WHERE Subscription_Status = 'No'
    GROUP BY Category
),
Developed_Product AS (
    SELECT
        Category,
        COUNT(CASE WHEN Review_Rating < 3 THEN 1 END) AS Low_Rating_Count
    FROM shopping_behavior_updated
    GROUP BY Category
),
Low_Rating_Products AS (
    SELECT
        Category,
        Low_Rating_Count
    FROM Developed_Product
    WHERE Low_Rating_Count > 0
    ORDER BY Low_Rating_Count DESC
)
-- SELECT * FROM Low_Rating_Products;

SELECT * FROM High_Rating_Percentage
WHERE Subscription_Status = 'No'
ORDER BY High_Rating_Percent DESC;

