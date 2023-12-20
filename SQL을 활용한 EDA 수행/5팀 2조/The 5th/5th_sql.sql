-- SELECT *
-- FROM shopping_behavior_updated_Sql;

-- SELECT Gender, Purchase_Amount_USD, Previous_Purchases, Item_Purchased, Season
-- FROM shopping_behavior_updated;

WITH Item_Sum AS (
    SELECT 
        Item_Purchased,
        Season,
        Gender,
        SUM(Purchase_Amount_USD) AS Total_Purchase_Amount,
        SUM(Previous_Purchases) AS Total_Previous_Purchases,
        SUM(Purchase_Amount_USD) - SUM(Previous_Purchases) AS Current_Purchased
    FROM shopping_behavior_updated
    GROUP BY Item_Purchased, Season, Gender
),
Ranked_Items AS(
	SELECT
		*,
        ROW_NUMBER() OVER(PARTITION BY Season, Gender ORDER BY Current_Purchased DESC) AS Ranked
        FROM Item_Sum
),
Div_Season_Amount As(
	SELECT
		 Item_Purchased,
         SUM(CASE WHEN Season = 'Fall' AND Gender = 'Female' THEN Purchase_Amount_USD ELSE 0 END) AS Female_Fall_Sum,
         SUM(CASE WHEN Season = 'Summer' AND Gender = 'Female' THEN Purchase_Amount_USD ELSE 0 END) AS Female_Summer_Sum,
         SUM(CASE WHEN Season = 'Winter' AND Gender = 'Female' THEN Purchase_Amount_USD ELSE 0 END) AS Female_Winter_Sum,
         SUM(CASE WHEN Season = 'Spring' AND Gender = 'Female' THEN Purchase_Amount_USD ELSE 0 END) AS Female_Spring_Sum,
         SUM(CASE WHEN Season = 'Fall' AND Gender = 'Male' THEN Purchase_Amount_USD ELSE 0 END) AS Male_Fall_Sum,
         SUM(CASE WHEN Season = 'Summer' AND Gender = 'Male' THEN Purchase_Amount_USD ELSE 0 END) AS Male_Summer_Sum,
         SUM(CASE WHEN Season = 'Winter' AND Gender = 'Male' THEN Purchase_Amount_USD ELSE 0 END) AS Male_Winter_Sum,
         SUM(CASE WHEN Season = 'Spring' AND Gender = 'Male' THEN Purchase_Amount_USD ELSE 0 END) AS Male_Spring_Sum
    FROM shopping_behavior_updated
    GROUP BY Item_Purchased
)
SELECT 
    Item_Purchased,
    Season,
    Gender,
    Total_Purchase_Amount,
    Total_Previous_Purchases,
    Current_Purchased
FROM Ranked_Items
WHERE Ranked <= 5
ORDER BY Season, Gender, Ranked;