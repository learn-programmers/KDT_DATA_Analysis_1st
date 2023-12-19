#period 확인
select strftime('%Y-%m-%d', period) as period
from datatable
where strftime('%Y-%m', period) = '2023-09'
group by period

#period 별 평균 ratio, count 값 비교 
with dateavg as (
select strftime('%Y-%m-%d', period) as period, round(avg(ratio)) as avg_ratio, COUNT(*) AS COUNT
from datatable 
where strftime('%Y-%m', period) = '2023-09'
group by period
)
SELECT *
FROM dateavg
order by avg_ratio desc