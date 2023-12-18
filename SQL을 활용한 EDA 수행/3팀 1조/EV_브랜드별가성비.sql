
-- BMW 가성비 차량
select cal.car_name, SUM(scores) AS total_score
 from (
	SELECT  car_name, sales, ((battery*2)+(efficiency*2)-(fast_charge*2)+(range_*5)+top_speed+acceleration-(price_de*10)) AS scores
  FROM ev_normalized
 union all
 SELECT  car_name, sales,((battery*2)+(efficiency*2)-(fast_charge*5)+(range_*2)+top_speed+acceleration-(price_de*10)) AS scores
  FROM ev_normalized
 union all 
 SELECT  car_name, sales,((battery*2)+(efficiency*5)-(fast_charge*2)+(range_*2)+top_speed+acceleration-(price_de*10))  AS scores
  FROM ev_normalized
 ) as cal where car_name like 'BMW%'
 group by car_name
 order by total_score desc
 Limit 5;
 
 -- BYD 가성비 차량
select cal.car_name, SUM(scores) AS total_score
 from (
	SELECT  car_name, sales, ((battery*2)+(efficiency*2)-(fast_charge*2)+(range_*5)+top_speed+acceleration-(price_de*10)) AS scores
  FROM ev_normalized
 union all
 SELECT  car_name, sales,((battery*2)+(efficiency*2)-(fast_charge*5)+(range_*2)+top_speed+acceleration-(price_de*10)) AS scores
  FROM ev_normalized
 union all 
 SELECT  car_name, sales,((battery*2)+(efficiency*5)-(fast_charge*2)+(range_*2)+top_speed+acceleration-(price_de*10))  AS scores
  FROM ev_normalized
 ) as cal where car_name like 'BYD%'
 group by car_name
 order by total_score desc
 Limit 5;
 
 -- Hyundai 가성비 차량
select cal.car_name, SUM(scores) AS total_score
 from (
	SELECT  car_name, sales, ((battery*2)+(efficiency*2)-(fast_charge*2)+(range_*5)+top_speed+acceleration-(price_de*10)) AS scores
  FROM ev_normalized
 union all
 SELECT  car_name, sales,((battery*2)+(efficiency*2)-(fast_charge*5)+(range_*2)+top_speed+acceleration-(price_de*10)) AS scores
  FROM ev_normalized
 union all 
 SELECT  car_name, sales,((battery*2)+(efficiency*5)-(fast_charge*2)+(range_*2)+top_speed+acceleration-(price_de*10))  AS scores
  FROM ev_normalized
 ) as cal where car_name like 'Hyundai%'
 group by car_name
 order by total_score desc
 Limit 5;
 
  -- Tesla 가성비 차량
select cal.car_name, SUM(scores) AS total_score
 from (
	SELECT  car_name, sales, ((battery*2)+(efficiency*2)-(fast_charge*2)+(range_*5)+top_speed+acceleration-(price_de*10)) AS scores
  FROM ev_normalized
 union all
 SELECT  car_name, sales,((battery*2)+(efficiency*2)-(fast_charge*5)+(range_*2)+top_speed+acceleration-(price_de*10)) AS scores
  FROM ev_normalized
 union all 
 SELECT  car_name, sales,((battery*2)+(efficiency*5)-(fast_charge*2)+(range_*2)+top_speed+acceleration-(price_de*10))  AS scores
  FROM ev_normalized
 ) as cal where car_name like 'Tesla%'
 group by car_name
 order by total_score desc
 Limit 5;
 
   -- Volkswagen 가성비 차량
select cal.car_name, SUM(scores) AS total_score
 from (
	SELECT  car_name, sales, ((battery*2)+(efficiency*2)-(fast_charge*2)+(range_*5)+top_speed+acceleration-(price_de*10)) AS scores
  FROM ev_normalized
 union all
 SELECT  car_name, sales,((battery*2)+(efficiency*2)-(fast_charge*5)+(range_*2)+top_speed+acceleration-(price_de*10)) AS scores
  FROM ev_normalized
 union all 
 SELECT  car_name, sales,((battery*2)+(efficiency*5)-(fast_charge*2)+(range_*2)+top_speed+acceleration-(price_de*10))  AS scores
  FROM ev_normalized
 ) as cal where car_name like 'Volkswagen%'
 group by car_name
 order by total_score desc
 Limit 5
 