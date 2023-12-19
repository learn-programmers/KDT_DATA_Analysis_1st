SELECT FLOOR((2021 - Year_Birth)/10)*10 AS Age_Group, AVG((MntWines+MntFruits+MntMeatProducts+MntFishProducts+MntSweetProducts+MntGoldProds)/24) AS Avg_Monthly_Purchases
FROM customer
GROUP BY Age_Group;

SELECT 
    FLOOR((2021 - Year_Birth)/10)*10 AS Age_Group, 
    (COUNT(*) / (SELECT COUNT(*) FROM customer))*100 AS Age_Ratio, 
    AVG((MntWines+MntFruits+MntMeatProducts+MntFishProducts+MntSweetProducts+MntGoldProds)/24) AS Avg_Monthly_Purchases
FROM customer
GROUP BY Age_Group;

SELECT 
	Income, 
	MAX(MntWines) AS Wines, 
	MAX(MntFruits) AS Fruits, 
	MAX(MntMeatProducts) AS Meat, 
	MAX(MntFishProducts) AS Fish,
	MAX(MntSweetProducts) AS Sweets, 
	MAX(MntGoldProds) AS Gold
FROM customer
GROUP BY Income
ORDER BY Income DESC
LIMIT 20;

SELECT 
	NumWebVisitsMonth, 
	AVG(NumWebPurchases) AS Avg_Web_Purchases
FROM customer
GROUP BY NumWebVisitsMonth;

SELECT 
	AVG(AcceptedCmp1) AS Avg_Acceptance_Rate_Cmp1, 
	AVG(AcceptedCmp2) AS Avg_Acceptance_Rate_Cmp2, 
	AVG(AcceptedCmp3) AS Avg_Acceptance_Rate_Cmp3, 
	AVG(AcceptedCmp4) AS Avg_Acceptance_Rate_Cmp4, 
	AVG(AcceptedCmp5) AS Avg_Acceptance_Rate_Cmp5
FROM customer;


SELECT 
    'Wines' AS Product, AVG(Income) AS Avg_Income FROM customer WHERE MntWines > 0
UNION ALL

SELECT 
    'Fruits' AS Product, AVG(Income) AS Avg_Income FROM customer WHERE MntFruits > 0
UNION ALL

SELECT 
    'Meat' AS Product, AVG(Income) AS Avg_Income FROM customer WHERE MntMeatProducts > 0
UNION ALL

SELECT 
    'Fish' AS Product, AVG(Income) AS Avg_Income FROM customer WHERE MntFishProducts > 0
UNION ALL

SELECT 
    'Sweets' AS Product, AVG(Income) AS Avg_Income FROM customer WHERE MntSweetProducts > 0
UNION ALL

SELECT 
    'Gold' AS Product, AVG(Income) AS Avg_Income FROM customer WHERE MntGoldProds > 0;

SELECT 
	YEAR, 
	MONTH, 
	DAY, 
	AVG((MntWines+MntFruits+MntMeatProducts+MntFishProducts+MntSweetProducts+MntGoldProds)/24) AS Avg_Purchases
FROM customer
GROUP BY year, month, DAY;

SELECT
    Education,
    AVG(MntWines) AS Avg_Wines,
    AVG(MntFruits) AS Avg_Fruits,
    AVG(MntMeatProducts) AS Avg_Meat,
    AVG(MntFishProducts) AS Avg_Fish,
    AVG(MntSweetProducts) AS Avg_Sweets,
    AVG(MntGoldProds) AS Avg_Gold,
    AVG(MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts + MntGoldProds) AS Total_Avg
FROM customer
GROUP BY Education
ORDER BY Education;


SELECT 
	Education, 
	Marital_Status, 
	AVG((MntWines+MntFruits+MntMeatProducts+MntFishProducts+MntSweetProducts+MntGoldProds)/24) AS Avg_Purchases
FROM customer
GROUP BY Education, Marital_Status;


SELECT 
	Kidhome, 
	Teenhome, 
	AVG(MntWines) AS Avg_Wines, 
	AVG(MntFruits) AS Avg_Fruits, 
	AVG(MntMeatProducts) AS Avg_Meat, 
	AVG(MntFishProducts) AS Avg_Fish, 
	AVG(MntSweetProducts) AS Avg_Sweets, 
	AVG(MntGoldProds) AS Avg_Gold
FROM customer
WHERE Kidhome > 0 OR Teenhome > 0
GROUP BY Kidhome, Teenhome;


SELECT AcceptedCmp1, AcceptedCmp2, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, AVG(NumStorePurchases + NumWebPurchases + NumCatalogPurchases + NumDealsPurchases) AS Avg_Purchases
FROM customer
GROUP BY AcceptedCmp1, AcceptedCmp2, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5;

SELECT NumWebVisitsMonth, AVG(NumWebPurchases) AS Avg_Web_Purchases
FROM customer
GROUP BY NumWebVisitsMonth;

SELECT ID, Income
FROM customer
ORDER BY Income DESC
LIMIT 10;

SELECT *
FROM customer
ORDER BY (NumStorePurchases + NumWebPurchases + NumCatalogPurchases + NumDealsPurchases) DESC
LIMIT 10;

SELECT FLOOR((2021 - Year_Birth)/10)*10 AS Age_Group, COUNT(*) AS Num_Customers
FROM customer
WHERE Response = 1
GROUP BY Age_Group;

SELECT AVG(2021 - Year_Birth) AS Avg_Age, MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds
FROM customer
GROUP BY MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds;

SELECT ID, (Kidhome + Teenhome) AS Num_Children, MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds
FROM customer
ORDER BY Num_Children DESC
LIMIT 1;

SELECT ID, NumWebVisitsMonth
FROM customer
ORDER BY NumWebVisitsMonth DESC
LIMIT 10;

SELECT NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumDealsPurchases
FROM customer;

SELECT Recency, COUNT(*) as Count
FROM customer
GROUP BY Recency
ORDER BY Recency;


