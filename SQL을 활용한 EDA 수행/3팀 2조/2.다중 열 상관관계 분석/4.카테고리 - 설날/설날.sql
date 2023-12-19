-- 설날 period 별 평균 ratio, count 값 비교
with dateavg as (
    select strftime('%Y-%m-%d', period) as period, round(avg(ratio)) as avg_ratio, COUNT(*) AS COUNT
    from datatable 
    where period in ('2023-01-02', '2023-01-09', '2023-01-16', '2023-01-23')
    group by period
)
SELECT *
FROM dateavg
order by avg_ratio desc;