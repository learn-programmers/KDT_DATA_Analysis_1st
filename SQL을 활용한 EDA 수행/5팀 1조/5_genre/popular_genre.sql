-- stuff, features 테이블 song, performer 기준으로 INNER JOIN

-- SELECT STR_TO_DATE(s.weekID, '%m/%d/%Y') AS Year, s.week_position, f.spotify_genre
-- FROM stuff s
-- INNER JOIN features f ON s.song = f.song AND s.performer = f.performer
-- WHERE YEAR(STR_TO_DATE(s.weekID, '%m/%d/%Y')) >= 2010
-- ORDER BY Year DESC, s.week_position DESC;

-- 연도별 장르 평균 순위, 중복횟수

-- WITH score AS (
-- SELECT YEAR(weekID) AS Year, spotify_genre, AVG(week_position) AS avg_position, COUNT(*) AS genre_count
-- FROM genre
-- WHERE spotify_genre IS NOT NULL AND spotify_genre != ''
-- GROUP BY Year, spotify_genre
-- ORDER BY Year, genre_count DESC)


-- 연도별 인기 장르 선정을 위한 점수화 
WITH score AS (
    SELECT 
        YEAR(weekID) AS Year, 
        spotify_genre, 
        AVG(week_position) AS avg_position, 
        COUNT(*) AS genre_count
    FROM genre
    WHERE spotify_genre IS NOT NULL AND spotify_genre != ''
    GROUP BY Year, spotify_genre
    ORDER BY Year, genre_count DESC
),
-- 순위가 높을수록, 빈도수가 높을 수록 인기 점수가 높도록 계산, 가중치는 각각 0.4, 0.6
RankedGenres AS (
    SELECT 
        Year,
        spotify_genre,
        avg_position,
        genre_count,
        ROUND(
            (RANK() OVER (PARTITION BY Year ORDER BY avg_position DESC) / CAST(COUNT(*) OVER (PARTITION BY Year) AS DECIMAL)) * 0.4 +
            (RANK() OVER (PARTITION BY Year ORDER BY genre_count ASC) / CAST(COUNT(*) OVER (PARTITION BY Year) AS DECIMAL)) * 0.6,
            2
        ) AS popularity_score
    FROM score
)
-- 연도, 장르, 평균 순위, 빈도수, 점수화해서 연도별로 점수가 가장 높은 장르 추출 
SELECT Year, spotify_genre, avg_position, genre_count, popularity_score
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY Year ORDER BY popularity_score DESC) AS genre_rank
    FROM RankedGenres
) AS RankedGenresWithRowNum
WHERE genre_rank = 1
ORDER BY Year;
