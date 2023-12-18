-- 각 컬럼별로 평균치 구하기
SELECT 
    Brand,
    ROUND(AVG(Battery)) AS Battery,
	ROUND(AVG(Efficiency)) AS Efficiency,
	ROUND(AVG(Fast_charge)) AS FastCharge,
    ROUND(AVG(Price_DE)) AS Price_DE,
    ROUND(AVG(Range_)) AS Range_,
    ROUND(AVG(Top_speed)) AS Top_speed,
    ROUND(AVG(acceleration)) AS acceleration,
    ROUND(AVG(Sales)) AS Sales
-- SUBSTRING_INDEX 함수 활용하여 차량 브랜드 추출하기
FROM
    (SELECT 
        SUBSTRING_INDEX(Car_name, ' ', 1) AS Brand,
            Battery,
            Efficiency,
            Fast_Charge,
            Price_DE,
            Range_,
            Top_speed,
            acceleration,
            Sales
            
    FROM
        EV) AS BrandData
GROUP BY Brand
-- 판매량 높은 순으로 정렬하기
ORDER BY Sales DESC
LIMIT 5;