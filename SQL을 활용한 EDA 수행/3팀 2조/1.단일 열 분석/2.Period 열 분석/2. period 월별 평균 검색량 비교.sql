-- period 평균 ratio 
select strftime('%Y-%m', period) as period
, case when strftime('%Y-%m', period) != '2023-12' then round(avg(ratio),1) 
            else null
            end as month_average_ratio 
from datatable 
group by 1 order by 2 desc