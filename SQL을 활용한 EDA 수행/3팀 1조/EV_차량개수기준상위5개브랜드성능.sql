# 총 브랜드 갯수
SELECT 
    COUNT(DISTINCT SUBSTRING_INDEX(Car_name, ' ', 1)) AS Number_Brand
FROM
    ev1;
--  -----------------------------------------------------------------
# 51개 브랜드 모두 조회 
SELECT DISTINCT
    SUBSTRING_INDEX(Car_name, ' ', 1) AS Brand
FROM
    ev1;
--  -----------------------------------------------------------------
# 차량 갯수가 가장 많은 상위 브랜드 5개
SELECT 
    Brand, COUNT(*) AS NumberOfCars
FROM
    (SELECT 
        SUBSTRING_INDEX(Car_name, ' ', 1) AS Brand
    FROM
        ev1) AS Brands
GROUP BY Brand
ORDER BY NumberOfCars DESC
LIMIT 5;
--  -----------------------------------------------------------------
# 브랜드별 FastCharge 평균 구하기.
SELECT 
    Brand, ROUND(AVG(Fast_charge)) AS FastCharge
FROM
    (SELECT 
        SUBSTRING_INDEX(Car_name, ' ', 1) AS Brand, Fast_charge
    FROM
        ev1) AS BrandData
GROUP BY Brand
ORDER BY FastCharge DESC;
--  -----------------------------------------------------------------
# 차량 갯수가 많은 상위 5개 브랜드에서 AverageFastCharge 구하기
SELECT 
    TopBrands.Brand,
    ROUND(AVG(ev1.Fast_charge)) AS AverageFastCharge
FROM
    (SELECT 
        SUBSTRING_INDEX(Car_name, ' ', 1) AS Brand
    FROM
        ev1
    GROUP BY SUBSTRING_INDEX(Car_name, ' ', 1)
    ORDER BY COUNT(*) DESC
    LIMIT 5) AS TopBrands
        JOIN
    ev1 ON TopBrands.Brand = SUBSTRING_INDEX(ev1.Car_name, ' ', 1)
GROUP BY TopBrands.Brand
ORDER BY AverageFastCharge DESC;
--  -----------------------------------------------------------------
# 차량 갯수가 많은 상위 5개 브랜드에서 AverageBattery구하기
SELECT 
    TopBrands.Brand,
    ROUND(AVG(ev1.Battery)) AS AverageBattery
FROM
    (SELECT 
        SUBSTRING_INDEX(Car_name, ' ', 1) AS Brand
    FROM
        ev1
    GROUP BY SUBSTRING_INDEX(Car_name, ' ', 1)
    ORDER BY COUNT(*) DESC
    LIMIT 5) AS TopBrands
        JOIN
    ev1 ON TopBrands.Brand = SUBSTRING_INDEX(ev1.Car_name, ' ', 1)
GROUP BY TopBrands.Brand
ORDER BY AverageBattery DESC;
--  -----------------------------------------------------------------
# 차량 갯수가 많은 상위 5개 브랜드에서 AverageRange구하기
SELECT 
    TopBrands.Brand,
    ROUND(AVG(ev1.range)) AS AverageRange
FROM
    (SELECT 
        SUBSTRING_INDEX(Car_name, ' ', 1) AS Brand
    FROM
        ev1
    GROUP BY SUBSTRING_INDEX(Car_name, ' ', 1)
    ORDER BY COUNT(*) DESC
    LIMIT 5) AS TopBrands
        JOIN
    ev1 ON TopBrands.Brand = SUBSTRING_INDEX(ev1.Car_name, ' ', 1)
GROUP BY TopBrands.Brand
ORDER BY AverageRange DESC;
--  -----------------------------------------------------------------
# 차량 갯수가 많은 상위 5개 브랜드에서 AverageSales구하기
SELECT 
    TopBrands.Brand,
    ROUND(AVG(ev1.Sales)) AS AverageSales
FROM
    (SELECT 
        SUBSTRING_INDEX(Car_name, ' ', 1) AS Brand
    FROM
        ev1
    GROUP BY SUBSTRING_INDEX(Car_name, ' ', 1)
    ORDER BY COUNT(*) DESC
    LIMIT 5) AS TopBrands
        JOIN
    ev1 ON TopBrands.Brand = SUBSTRING_INDEX(ev1.Car_name, ' ', 1)
GROUP BY TopBrands.Brand
ORDER BY AverageSales DESC;
--  -----------------------------------------------------------------
# 차량 갯수가 많은 상위 5개 브랜드에서 AveragePrice구하기
SELECT 
    TopBrands.Brand,
    ROUND(AVG(ev1.Price_DE)) AS AveragePrice
FROM
    (SELECT 
        SUBSTRING_INDEX(Car_name, ' ', 1) AS Brand
    FROM
        ev1
    GROUP BY SUBSTRING_INDEX(Car_name, ' ', 1)
    ORDER BY COUNT(*) DESC
    LIMIT 5) AS TopBrands
        JOIN
    ev1 ON TopBrands.Brand = SUBSTRING_INDEX(ev1.Car_name, ' ', 1)
GROUP BY TopBrands.Brand
ORDER BY AveragePrice;
