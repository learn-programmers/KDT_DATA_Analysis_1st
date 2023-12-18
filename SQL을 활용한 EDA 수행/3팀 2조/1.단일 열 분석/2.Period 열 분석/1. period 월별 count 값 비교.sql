-- period 월별 count
select strftime('%Y-%m', period) as month, count(*) as count
from datatable
group by strftime('%Y-%m', period)
order by 1 asc;

with rankedcount as (


select strftime('%Y-%m', period) as month, count(*) as count
from datatable
group by strftime('%Y-%m', period)
order by 1 asc



)
SELECT ROUND(AVG(count_ratio)) AS avg_count
FROM rankedcount
WHERE month <> '2023-12';