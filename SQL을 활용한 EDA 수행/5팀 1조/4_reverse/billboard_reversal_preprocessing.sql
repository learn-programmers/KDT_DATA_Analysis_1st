WITH stuff_processed AS (
	SELECT right(weekID, 4) as weekID_year,
	min(peak_position) as peak,
	song, songID
	FROM devcourse_db.stuff s
	GROUP BY songID, weekID_year
), stuff_processed2 AS (
	SELECT weekID_year, peak, song, songID,
	count(*) over (partition by songID) as cnt,
	max(cast(weekID_year as unsigned)) over (partition by songID) as max_year,
	min(cast(weekID_year as unsigned)) over (partition by songID) as min_year
	FROM stuff_processed
)
SELECT weekID_year, peak, song, songID, cnt, max_year, min_year
FROM stuff_processed2
WHERE max_year - min_year + 1 > cnt
ORDER BY songID, weekID_year ASC;