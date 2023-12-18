-- period 월별 count , 평균 ratio 비교
select strftime('%Y-%m', period) as period, round(avg(ratio)) as avg_ratio , count(*) as count
from datatable
where strftime('%Y-%m', period) != '2023-12'
group by 1
order by 1