--CREATE TABLE AgeGroup_Cnt
--
--SELECT
--	'Item Purchased', FLOOR(Age / 5) * 5,
--	SUM(CASE WHEN Gender = 'Male' THEN 1 ELSE 0 END) AS Male_cnt,
--    SUM(CASE WHEN Gender = 'Female' THEN 1 ELSE 0 END) AS Female_Cnt,
--    COUNT(*) as AgeGroupPurchase_Cnt
--FROM shopping_behavior_updated
--GROUP BY 'Item Purchased', FLOOR(Age / 5) * 5
--ORDER BY FLOOR(Age / 5) * 5;

--WITH Percent AS (-- male, female 칼럼의
--	SELECT
--		*,
--		(Male_Cnt / SUM(male_Cnt) OVER()) * 100 AS Male_per,
--		(Female_Cnt / Sum(Female_Cnt) OVER ()) * 100 As Female_Per
--FROM gender_count_trend
--)
--SELECT * FROM Percent
--WHERE Female_per > 4 AND Female_per - Male_per >= 0.25 -- 강력추천은 >= 1 , 중간 >= 0.5and, 약간 >= 0.25
--ORDER BY Female_per DESC;
--
--CREATE TABLE Gender_cnt AS
--
--WITH GenderBasedPurchases AS (
--    SELECT
--		Gender, `Item Purchased`,
--        COUNT(*) as Purchase_Cnt
--    FROM shopping_trends
--    GROUP BY Gender, `Item Purchased`
--)
--SELECT
--    `Item Purchased`,
--    SUM(CASE WHEN Gender = 'Male' THEN Purchase_Cnt ELSE 0 END) as Male_Cnt,
--    SUM(CASE WHEN Gender = 'Female' THEN Purchase_Cnt ELSE 0 END) as Female_Cnt,
--    SUM(Purchase_Cnt) as Total_Cnt
--FROM GenderBasedPurchases
--GROUP BY `Item Purchased`
--ORDER BY Total_Cnt Desc;

WITH GenderBasedPurchases AS (
    SELECT 
        Gender, 
        `Item Purchased`, 
        COUNT(*) as Purchase_Cnt
    FROM shopping_trends
    GROUP BY Gender, `Item Purchased`
),
AgeBasedPurchases AS (
    SELECT 
        `Item Purchased`, 
        FLOOR(Age / 5) * 5 AS AgeGroup,
        COUNT(*) as AgeGroupPurchase_Cnt
    FROM shopping_trends
    GROUP BY `Item Purchased`, FLOOR(Age / 5) * 5
)
SELECT 
    gbp.`Item Purchased`, 
    SUM(CASE WHEN gbp.Gender = 'Male' THEN gbp.Purchase_Cnt ELSE 0 END) as Male_Cnt,
    SUM(CASE WHEN gbp.Gender = 'Female' THEN gbp.Purchase_Cnt ELSE 0 END) as Female_Cnt,
    SUM(gbp.Purchase_Cnt) as Total_Cnt,
    abp.AgeGroup,
    abp.AgeGroupPurchase_Cnt
FROM GenderBasedPurchases gbp
LEFT JOIN AgeBasedPurchases abp ON gbp.`Item Purchased` = abp.`Item Purchased`
GROUP BY gbp.`Item Purchased`, abp.AgeGroup
ORDER BY gbp.`Item Purchased`, abp.AgeGroup, Total_Cnt Desc;
