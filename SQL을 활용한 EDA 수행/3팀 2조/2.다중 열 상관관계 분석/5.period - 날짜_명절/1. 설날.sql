#period 확인
select period
from datatable
where strftime('%Y-%m', period) = '2023-01'
group by period

#period 별 평균 ratio, count 값 비교
with dateavg as (
select strftime('%Y-%m-%d', period) as period, round(avg(ratio)) as avg_ratio, COUNT(*) AS COUNT
from datatable 
where strftime('%Y-%m-%d', period) in ('2023-01-02', '2023-01-09', '2023-01-16', '2023-01-23' )
group by period
)

SELECT *
FROM dateavg
ORDER BY avg_ratio DESC

# 평균 ratio가 가장 높은 2023-01-09와 가장 낮은 2023-01-16 의 category, product_name 항목 별 비교 
# -> 어떤 category, product_name이 period에 따라 우세한지 알 수 있음 
with dateavg as (
select category, product_name, period, round(avg(ratio)) as avg_ratio
from datatable 
where strftime('%Y-%m-%d', period) in ('2023-01-09', '2023-01-16')
group by category, product_name, period
)
, ComparisonResult as (
select da1.category, da1.product_name
    , da1.avg_ratio as avg_ratio_01_09, da2.avg_ratio as avg_ratio_01_16
    , da1.avg_ratio - da2.avg_ratio as ratio_diff
    , case when da1.avg_ratio - da2.avg_ratio > 0 then '2023-01-09'
           when da1.avg_ratio - da2.avg_ratio < 0 then '2023-01-16'
           else 'same'
    end as higher_period
from dateavg as da1
inner join dateavg as da2
on da1.category = da2.category
and da1.product_name = da2.product_name
and da1.period = '2023-01-09'
and da2.period = '2023-01-16'

)

select category, product_name, higher_period
from ComparisonResult
group by 1,2,3
order by 3 asc