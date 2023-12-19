-- 키워드별 평균 가격 데이터
 
with avg as( -- 키워드별 평균 가격
SELECT category, product_name, avg(price) as avg_price
FROM datatable
group by 1,2
)
SELECT avg.category,avg.product_name, round(avg_price,-1) as price -- 가격 10단위 반올림
FROM avg
WHERE 1=1
ORDER BY avg.category