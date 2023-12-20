SELECT * FROM adidas.adidas;

-- 2020년, 2021년 평균 매출
select year(InvoiceDate) as 'YEAR', avg(TotalSales) as 'TotalSales'
from adidas.adidas
group by YEAR;

-- 2020년 평균 매출
SELECT InvoiceDate, avg(TotalSales) as' Mean Total Sales'
FROM adidas.adidas
WHERE year(InvoiceDate) = 2020
Group by InvoiceDate;

-- 2021년 평균 매출
SELECT InvoiceDate, avg(TotalSales) as' Mean Total Sales'
FROM adidas.adidas
WHERE year(InvoiceDate) = 2021
Group by InvoiceDate;

-- 2020년 월별 평균 매출
select month(InvoiceDate) as '2020' , avg(TotalSales) as 'Mean Total Sales'
from adidas.adidas
where year(InvoiceDate) = 2020
Group by month(InvoiceDate);

-- 2021년 월별 평균 매출
select month(InvoiceDate) as '2021' , avg(TotalSales) as 'Mean Total Sales'
from adidas.adidas
where year(InvoiceDate) = 2021
Group by month(InvoiceDate);

-- 성별에 따른 매출액 비교
select
	case 
		when substr(Product,1,1) = 'M' then 'M'
        when substr(Product,1,1) = 'W' then 'W'
        Else 'Other'
	end as Product_group,
    avg(TotalSales) as 'Total Sales'
from adidas.adidas
group by product_group;

-- 2020&2021년 11월, 12월, 그 외 달 평균 매출 비교
SELECT
    CASE
        WHEN YEAR(InvoiceDate) = 2020 AND MONTH(InvoiceDate) = 11 THEN '2020 November'
        WHEN YEAR(InvoiceDate) = 2020 AND MONTH(InvoiceDate) = 12 THEN '2020 December'
        WHEN YEAR(InvoiceDate) = 2021 AND MONTH(InvoiceDate) = 11 THEN '2021 November'
        WHEN YEAR(InvoiceDate) = 2021 AND MONTH(InvoiceDate) = 12 THEN '2021 December'
        WHEN YEAR(InvoiceDate) = 2020 THEN '2020 Other'
        WHEN YEAR(InvoiceDate) = 2021 THEN '2021 Other'
        ELSE 'Other'
    END AS month_group,
    AVG(TotalSales) AS 'Average Total Sales'
FROM adidas.adidas
WHERE (YEAR(InvoiceDate) = 2020 OR YEAR(InvoiceDate) = 2021)
GROUP BY month_group;

-- 11월에 많이 팔린 제품
select Product , year(InvoiceDate) as 'year_11', avg(TotalSales) as 'Total Sales'
from adidas.adidas
where month(InvoiceDate) = 11 and year(InvoiceDate) in (2020,2021)
group by Product, year_11;

-- 12월에 많이 팔린 제품
select Product , year(InvoiceDate) as 'year_12', avg(TotalSales) as 'Total Sales'
from adidas.adidas
where month(InvoiceDate) = 12 and year(InvoiceDate) in (2020,2021)
group by Product, year_12;

-- 그 외의 달에 많이 팔린 제품
select Product , year(InvoiceDate) as 'year_other', avg(TotalSales) as 'Total Sales'
from adidas.adidas
where month(InvoiceDate) not in (11,12) and year(InvoiceDate) in (2020,2021)
group by Product, year_other;

-- 달 구분 없이 많이 팔린 제품 비교
select Product , year(InvoiceDate) as 'sales_year', avg(TotalSales) as 'Total Sales'
from adidas.adidas
group by Product, sales_year;

-- 매출이 많은 도시(state같이 표시) 내림차순
select State, City, avg(TotalSales) as 'TotalSales'
from adidas.adidas
group by State, City
order by TotalSales desc;

-- 매출이 많은 주 내림차순 
select State, avg(TotalSales) as 'TotalSales'
from adidas.adidas
group by State
order by TotalSales;

-- 매출이 많은 주에서 상위 5개 주에 포함된 도시의 평균 매출액
select City, avg(TotalSales) as 'AverageSalesPerCity'
from adidas.adidas
where State in (
    select State
    from (
        select State, avg(TotalSales) as 'AvgSalesPerState'
        from adidas.adidas
        group by State
        order by AvgSalesPerState desc
        limit 5
    ) as TopStates
)
group by City
order by AverageSalesPerCity desc;

-- 매출이 적은 주에서 하위 5개 주에 포함된 도시의 평균 매출액
select City, avg(TotalSales) as 'AverageSalesPerCity'
from adidas.adidas
where State in (
    select State
    from (
        select State, avg(TotalSales) as 'AvgSalesPerState'
        from adidas.adidas
        group by State
        order by AvgSalesPerState
        limit 5
    ) as TopStates
)
group by City
order by AverageSalesPerCity desc;

-- OperatingMargin 다시 계산
select Retailer, RetailerID, InvoiceDate, Region, State, City, Product, PriceperUnit, UnitsSold, TotalSales, OperatingProfit, OperatingMargin, OperatingProfit/TotalSales as OperatingMargin2, SalesMethod
from adidas.adidas;
-- 다시 계산해서 결과 확인했지만 기존 것과 다를게 없음 (소수점 자리 모두)

-- 2020년, 2021년 평균 마진률
select year(InvoiceDate) as 'YEAR', avg(OperatingMargin) as 'OperatingMargin'
from adidas.adidas
group by YEAR;

-- 2020년, 2021년 월별 평균 마진률
select year(InvoiceDate) as 'YEAR', month(InvoiceDate) as 'MONTH', avg(OperatingMargin) as 'OperatingMargin'
from adidas.adidas
group by YEAR,MONTH
order by YEAR, MONTH;

-- 2020&2021년 11월, 12월, 그 외 달 평균 마진률 비교
SELECT
    CASE
        WHEN YEAR(InvoiceDate) = 2020 AND MONTH(InvoiceDate) = 11 THEN '2020 November'
        WHEN YEAR(InvoiceDate) = 2020 AND MONTH(InvoiceDate) = 12 THEN '2020 December'
        WHEN YEAR(InvoiceDate) = 2021 AND MONTH(InvoiceDate) = 11 THEN '2021 November'
        WHEN YEAR(InvoiceDate) = 2021 AND MONTH(InvoiceDate) = 12 THEN '2021 December'
        WHEN YEAR(InvoiceDate) = 2020 THEN '2020 Other'
        WHEN YEAR(InvoiceDate) = 2021 THEN '2021 Other'
        ELSE 'Other'
    END AS month_group,
    AVG(OperatingMargin) AS 'Average OperatingMargin'
FROM adidas.adidas
WHERE (YEAR(InvoiceDate) = 2020 OR YEAR(InvoiceDate) = 2021)
GROUP BY month_group;

-- 11월에 많이 팔린 제품 마진률
select Product , year(InvoiceDate) as 'year_11', avg(OperatingMargin) as 'OperatingMargin'
from adidas.adidas
where month(InvoiceDate) = 11 and year(InvoiceDate) in (2020,2021)
group by Product, year_11;

-- 12월에 많이 팔린 제품 마진률
select Product , year(InvoiceDate) as 'year_12', avg(OperatingMargin) as 'OperatingMargin'
from adidas.adidas
where month(InvoiceDate) = 12 and year(InvoiceDate) in (2020,2021)
group by Product, year_12;

-- 그 외의 달에 많이 팔린 제품 마진률
select Product , year(InvoiceDate) as 'year_other', avg(OperatingMargin) as 'OperatingMargin'
from adidas.adidas
where month(InvoiceDate) not in (11,12) and year(InvoiceDate) in (2020,2021)
group by Product, year_other;

-- 달 구분 없이 많이 팔린 제품 마진률 비교
select Product , year(InvoiceDate) as 'margin_year', avg(OperatingMargin) as 'OperatingMargin'
from adidas.adidas
group by Product, margin_year;

-- 마진률이 높은 도시(state같이 표시) 내림차순
select State, City, avg(OperatingMargin) as 'OperatingMargin'
from adidas.adidas
group by State, City
order by OperatingMargin desc;

-- 마진률이 높은 주 내림차순 
select State, avg(OperatingMargin) as 'OperatingMargin'
from adidas.adidas
group by State
order by OperatingMargin desc;

-- 마진률이 높은 주에서 상위 5개 주에 포함된 도시의 평균 매출액
select City, avg(OperatingMargin) as 'AverageOperatingMarginPerCity'
from adidas.adidas
where State in (
    select State
    from (
        select State, avg(OperatingMargin) as 'AvgOperatingMarginPerState'
        from adidas.adidas
        group by State
        order by AvgOperatingMarginPerState desc
        limit 5
    ) as TopStates
)
group by City
order by AverageOperatingMarginPerCity desc;

-- 마진률이 낮은 주에서 하위 5개 주에 포함된 도시의 평균 매출액
select City, avg(OperatingMargin) as 'AverageOperatingMarginPerCity'
from adidas.adidas
where State in (
    select State
    from (
        select State, avg(OperatingMargin) as 'AvgOperatingMarginPerState'
        from adidas.adidas
        group by State
        order by AvgOperatingMarginPerState
        limit 5
    ) as TopStates
)
group by City
order by AverageOperatingMarginPerCity desc;

-- Retail별 판매 시장 점유율
select YEAR(InvoiceDate) as year,
Retailer,
avg(TotalSales) as AvgTotalSales,
AVG(TotalSales) / SUM(AVG(TotalSales)) OVER (PARTITION BY YEAR(InvoiceDate))*100 AS Ratio
from adidas.adidas
group by 1,2
order by 1,Ratio desc;

-- Retail별 판매 방식 비율
select YEAR(InvoiceDate) as year,
Retailer,
SalesMethod,
count(1) as Count,
count(1) / SUM(count(1)) OVER (PARTITION BY concat(year,Retailer))*100 AS Ratio
from adidas.adidas
group by 1,2,3
order by 1,2,Ratio desc;


-- 연도별 판매방식 평균 매출
SELECT year(InvoiceDate) as year, SalesMethod, AVG(TotalSales) as AvgTotalSales
from adidas.adidas
group by 1,2
order by 1,2;