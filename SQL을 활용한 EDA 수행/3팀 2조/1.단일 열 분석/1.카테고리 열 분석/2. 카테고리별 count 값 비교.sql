-- 카테고리별 count
select category, count(period) as count
from datatable
group by 1 
order by 1