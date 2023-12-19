USE project_ev;
SELECT * FROM ev_normalized;
SELECT count(*) FROM ev_normalized;

SELECT battery, car_name, efficiency, fast_charge, price_de, range_, top_speed, acceleration, sales
  FROM ev_normalized;

-- 주행거리 가성비 top 10
SELECT car_name, sales, (battery*2)+(efficiency*2)-(fast_charge*2)+(range_*5)+top_speed+acceleration-(price_de*23) AS scores
  FROM ev_normalized
 ORDER BY 3 DESC
 LIMIT 10 ;
 
-- 충전시간 가성비 top 10
SELECT car_name, sales, (battery*2)+(efficiency*2)-(fast_charge*5)+(range_*2)+top_speed+acceleration-(price_de*23) AS scores
  FROM ev_normalized
 ORDER BY 3 DESC
 LIMIT 10;
 
 -- 에너지 효율 가성비 top 10
SELECT car_name, sales, (battery*2)+(efficiency*5)-(fast_charge*2)+(range_*2)+top_speed+acceleration-(price_de*23) AS scores
  FROM ev_normalized
 ORDER BY 3 DESC
 LIMIT 10;
 
 -- 주행거리 점수 + 충전시간 점수 + 에너지효율 점수 = 총 가성비 점수  top 10 
SELECT cal.car_name, SUM(scores) AS total_score
 FROM (SELECT car_name, sales, ((battery*2)+(efficiency*2)-(fast_charge*2)+(range_*5)+top_speed+acceleration-(price_de*23)) AS scores
         FROM ev_normalized
        UNION ALL
       SELECT car_name, sales, ((battery*2)+(efficiency*2)-(fast_charge*5)+(range_*2)+top_speed+acceleration-(price_de*23)) AS scores
         FROM ev_normalized
        UNION ALL 
       SELECT car_name, sales, ((battery*2)+(efficiency*5)-(fast_charge*2)+(range_*2)+top_speed+acceleration-(price_de*23)) AS scores
         FROM ev_normalized) AS cal
  GROUP BY car_name
  ORDER BY total_score DESC
  LIMIT 10;
  