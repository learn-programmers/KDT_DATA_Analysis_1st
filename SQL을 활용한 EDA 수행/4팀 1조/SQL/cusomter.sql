# EDA
SELECT FLOOR((2021 - Year_Birth)/10)*10 AS Age_Group, AVG((MntWines+MntFruits+MntMeatProducts+MntFishProducts+MntSweetProducts+MntGoldProds)/24) AS Avg_Monthly_Purchases
FROM customer
GROUP BY Age_Group

SELECT
    FLOOR((2021 - Year_Birth)/10)*10 AS Age_Group, 
    (COUNT(*) / (SELECT COUNT(*) FROM customer))*100 AS Age_Ratio, 
    AVG((MntWines+MntFruits+MntMeatProducts+MntFishProducts+MntSweetProducts+MntGoldProds)/24) AS Avg_Monthly_Purchases
FROM customer
GROUP BY Age_Group


SELECT 
	NumWebVisitsMonth, 
    AVG(NumWebPurchases) AS Avg_Web_Purchases
FROM customer
GROUP BY NumWebVisitsMonth

SELECT 
	AVG(AcceptedCmp1) AS Avg_Acceptance_Rate_Cmp1, 
    AVG(AcceptedCmp2) AS Avg_Acceptance_Rate_Cmp2, 
    AVG(AcceptedCmp3) AS Avg_Acceptance_Rate_Cmp3, 
    AVG(AcceptedCmp4) AS Avg_Acceptance_Rate_Cmp4, 
    AVG(AcceptedCmp5) AS Avg_Acceptance_Rate_Cmp5
FROM customer


SELECT 
    'Wines' AS Product, 
    AVG(Income) AS Avg_Income 
FROM customer 
WHERE MntWines > 0

UNION ALL

SELECT 
    'Fruits' AS Product, 
    AVG(Income) AS Avg_Income 
FROM customer 
WHERE MntFruits > 0

UNION ALL

SELECT 
    'Meat' AS Product, 
    AVG(Income) AS Avg_Income 
FROM customer 
WHERE MntMeatProducts > 0

UNION ALL

SELECT 
    'Fish' AS Product, 
    AVG(Income) AS Avg_Income 
FROM customer 
WHERE MntFishProducts > 0

UNION ALL

SELECT 
    'Sweets' AS Product, 
    AVG(Income) AS Avg_Income 
FROM customer 
WHERE MntSweetProducts > 0

UNION ALL

SELECT 
    'Gold' AS Product, 
    AVG(Income) AS Avg_Income 
FROM customer 
WHERE MntGoldProds > 0;


SELECT 
	year, 
    month, 
    day, 
    AVG((MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts
    + MntGoldProds) / 24) AS Avg_Purchases
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
ORDER BY 1


SELECT 
	Education,
    Marital_Status, 
    AVG((MntWines + MntFruits + MntMeatProducts +
    MntFishProducts + MntSweetProducts + MntGoldProds) / 24) AS Avg_Purchases
FROM customer
GROUP BY 1, 2


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
GROUP BY 1, 2


SELECT 
	AcceptedCmp1, 
    AcceptedCmp2, 
    AcceptedCmp3, 
    AcceptedCmp4, 
    AcceptedCmp5, 
    AVG(NumStorePurchases + NumWebPurchases + NumCatalogPurchases + NumDealsPurchases) AS Avg_Purchases
FROM customer
GROUP BY AcceptedCmp1, AcceptedCmp2, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5;

SELECT 
	NumWebVisitsMonth, 
    AVG(NumWebPurchases) AS Avg_Web_Purchases
FROM customer
GROUP BY 1

SELECT ID, Income
FROM customer
ORDER BY 2 DESC
LIMIT 10;

SELECT *
FROM customer
ORDER BY (NumStorePurchases + NumWebPurchases + NumCatalogPurchases + NumDealsPurchases) DESC
LIMIT 10;

SELECT 
	FLOOR((2021 - Year_Birth) / 10) * 10 AS Age_Group,
    COUNT(*) AS Num_Customers
FROM customer
WHERE Response = 1
GROUP BY 1

SELECT 
	AVG(2021 - Year_Birth) AS Avg_Age, 
    MntWines, 
    MntFruits, 
    MntMeatProducts, 
    MntFishProducts, 
    MntSweetProducts, 
    MntGoldProds
FROM customer
GROUP BY MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds;

SELECT 
	ID, 
    (Kidhome + Teenhome) AS Num_Children, 
    MntWines, 
    MntFruits,
    MntMeatProducts,
    MntFishProducts, 
    MntSweetProducts, 
    MntGoldProds
FROM customer
ORDER BY Num_Children DESC
LIMIT 1

SELECT 
	ID, 
    NumWebVisitsMonth
FROM customer
ORDER BY NumWebVisitsMonth DESC
LIMIT 10

SELECT 
	NumWebPurchases, 
    NumCatalogPurchases, 
    NumStorePurchases, 
    NumDealsPurchases
FROM customer

SELECT 
	Recency, 
    COUNT(*) as Count
FROM customer
GROUP BY 1
ORDER BY 1

# 이탈, 비이탈 고객 분류
SELECT
    'churned_customer' AS stats,
    COUNT(id) AS count,
    round(COUNT(id) /(SELECT COUNT(id) FROM customer) *100,2) AS percentage
FROM customer
WHERE recency >= 90

UNION

SELECT
    'non-churned customer' AS stats,
    COUNT(id) AS count,
    round(COUNT(id) /(SELECT COUNT(id) FROM customer) *100,2) AS percentage
FROM customer
WHERE recency < 90;

# 자녀 유무에 따른 이탈/비이탈 고객 비율
WITH Churned AS (
    SELECT *
    FROM customer2
    WHERE recency > 90
),

NonChurned AS (
 	SELECT *
   FROM customer2
   WHERE recency <= 90
)

SELECT
    'Churned' AS CustomerStatus,
    COUNT(*) AS TotalCustomers,
    SUM(CASE WHEN Kidhome = 1 AND Teenhome = 1 THEN 1 ELSE 0 END) AS BothKidsAndTeens,
    AVG(CAST(Kidhome = 1 AND Teenhome = 1 AS DECIMAL)) * 100 AS RatioPercentage
FROM Churned

UNION ALL

SELECT
    'Non-Churned' AS CustomerStatus,
    COUNT(*) AS TotalCustomers,
    SUM(CASE WHEN Kidhome = 1 AND Teenhome = 1 THEN 1 ELSE 0 END) AS BothKidsAndTeens,
    AVG(CAST(Kidhome = 1 AND Teenhome = 1 AS DECIMAL)) * 100 AS RatioPercentage
FROM NonChurned

# 이탈 고객의 연령대
SELECT 
	  FLOOR(age/10)*10 AS age_group,
    COUNT(*) as Count,
    COUNT(*)/(SELECT COUNT(*) FROM customer WHERE Churn = 1)*100 AS ratio
FROM customer
WHERE Churn = 1
GROUP BY 1

# 결혼 여부 기준으로 이탈 고객 파악
SELECT 
	marital_status, COUNT(*) as cnt,
	ROUND((COUNT(*) * 100.0 / (SELECT COUNT(*) FROM non_churned_customer)), 2) as percentage
FROM non_churned_customer
GROUP BY marital_status;

# 교육 수준 별 이탈 고객 파악
SELECT 
	Education, 
    COUNT(*) as Count
FROM customer
WHERE Churn = 1
GROUP BY 1

# 연령대 별 평균 수입
SELECT 
    FLOOR(age/10)*10 AS age_group,
    AVG(Income) AS avg_income,
    (SUM(Income)/(SELECT SUM(Income) FROM customer WHERE Churn = 1))*100 AS income_per
FROM customer
WHERE Churn = 1
GROUP BY 1

# 결혼 여부 별 평균 수입
SELECT
    CASE
        WHEN Marital_Status IN ('Single', 'Alone', 'YOLO', 'Absurd') THEN 'notmarry'
        WHEN Marital_Status IN ('Together', 'Married') THEN 'marry'
        ELSE Marital_Status
    END as Marital_Status_group,
    AVG(Income) AS avg_income,
    (SUM(Income)/(SELECT SUM(Income) FROM customer WHERE Churn = 1))*100 AS income_percent
FROMcustomer
WHEREChurn = 1
GROUP BY 1 

# 교육 수준 별 평균 수입
SELECT 
	Education, 
    AVG(Income) as avg_income,
    (SUM(Income)/(SELECT SUM(Income) FROM customer WHERE Churn = 1)) * 100 AS Income_Per
FROM customer
WHERE Churn = 1
GROUP BY 1

# 아이 유무에 따른 평균 수입
ALTER TABLE customer
ADD COLUMN children INT;

UPDATE customer
SET children = Kidhome + Teenhome;

SELECT
    CASE
        WHEN children = 0 THEN 'no_children'
        WHEN children >= 1 THEN 'children'
    END as children_group,
    AVG(Income) AS avg_income,
    (SUM(Income)/(SELECT SUM(Income) FROM customer WHERE Churn = 1))*100 AS income_per
FROM customer
WHERE Churn = 1
GROUP BY 1

# 이탈 고객 중 컴플레인 고객
SELECT *
FROM churned_customer
WHERE complain <> 0;

SELECT 
	AcceptedCmp1,
    AcceptedCmp2,
    AcceptedCmp3,
    AcceptedCmp4,
    AcceptedCmp5,
    Response            
FROM churned_customer
WHERE complain <> 0;

# 컴플레인 제기 이탈고객의 상품별 평균 지출액
SELECT
	ROUND(AVG(mntwines),0) AS wines,
	ROUND(AVG(mntfruits),0) AS fruits,
	ROUND(AVG(mntmeatproducts),0) AS meat,
	ROUND(AVG(mntfishproducts),0) AS fish,
	ROUND(AVG(mntsweetproducts),0) AS sweets,
	ROUND(AVG(mntgoldprods),0) AS gold
FROM churned_customer
WHERE complain <> 0;

# 컴플레인 제기 이탈 고객의 연령대 파악
SELECT age ,age_group
FROM churned_customer
WHERE complain <> 0;

# 결혼 여부
SELECT marital_status
FROM churned_customer
WHERE complain <> 0;

# 구매 경로 별 평균 구매 횟수
SELECT
    ROUND(AVG(numdealspurchases), 0) AS deals,
    ROUND(AVG(numwebpurchases), 0) AS web,
    ROUND(AVG(numcatalogpurchases), 0) AS catalog,
    ROUND(AVG(numstorepurchases), 0) AS store
FROM churned_customer
WHERE complain <> 0;

# 이탈/비이탈 고객의 캠페인 수락률
SELECT 
	churn,
	avg(AcceptedCmp1) as avg_cmp1,
	avg(AcceptedCmp2) as avg_cmp2,
	avg(AcceptedCmp3) as avg_cmp3,
	avg(AcceptedCmp4) as avg_cmp4,
	avg(AcceptedCmp5) as avg_cmp5,
	avg(response) as avg_response
FROM customer
GROUP BY churn

# 구매 경로
WITH Churned AS (
    SELECT *
    FROM customer2
    WHERE recency > 90
)

SELECT
    'Web Purchases' AS PurchaseType,
    AVG(NumWebPurchases) AS AvgPurchases
FROM Churned

UNION

SELECT
    'Catalog Purchases' AS PurchaseType,
    AVG(NumCatalogPurchases) AS AvgPurchases
FROM Churned

UNION

SELECT
    'Store Purchases' AS PurchaseType,
    AVG(NumStorePurchases) AS AvgPurchases
FROM Churned

# 이탈 고객과 비이탈 고객의 최근 한 달 간 홈페이지 방문 횟수 차이
SELECT 
	churn,
	avg(numwebvisitsmonth) as avg_webvisit 
FROM customer
GROUP BY churn