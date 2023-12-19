# S21 출시 전 조회수 평균 증가율
WITH s21 AS (
    SELECT
        publishedAt,
        video_id,
        title,
        (MAX(view_count) - MIN(view_count)) / MIN(view_count) * 100 AS growth_rate
    FROM youtube_trending
    WHERE trending_date <= '2021-01-29' 
    	AND publishedAt_year = 2021
    	AND (title LIKE '%갤럭시%' OR title LIKE '%galaxy%')
        AND title LIKE '%S21%'
    GROUP BY video_id, title, publishedAt
    ORDER BY publishedAt
)
SELECT AVG(growth_rate) AS average
FROM s21;

# S21 출시 후 조회수 평균 증가율
WITH s21 AS (
    SELECT
        publishedAt,
        video_id,
        title,
        (MAX(view_count) - MIN(view_count)) / MIN(view_count) * 100 AS growth_rate
    FROM youtube_trending
    WHERE trending_date > '2021-01-29' 
    	AND publishedAt_year = 2021
    	AND (title LIKE '%갤럭시%' OR title LIKE '%galaxy%')
        AND title LIKE '%S21%'
    GROUP BY video_id, title, publishedAt
    ORDER BY publishedAt
)
SELECT AVG(growth_rate) AS average
FROM s21;

# S22 출시 전 조회수 평균 증가율
WITH s22 AS (
    SELECT
        publishedAt,
        video_id,
        title,
        (MAX(view_count) - MIN(view_count)) / MIN(view_count) * 100 AS growth_rate
    FROM youtube_trending
    WHERE trending_date <= '2022-02-25' 
    	AND publishedAt_year = 2022
    	AND (title LIKE '%갤럭시%' OR title LIKE '%galaxy%')
        AND title LIKE '%S22%'
    GROUP BY video_id, title, publishedAt
    ORDER BY publishedAt
)
SELECT AVG(growth_rate) AS average
FROM s22;

# S22 출시 후 조회수 평균 증가율
WITH s22 AS (
    SELECT
        publishedAt,
        video_id,
        title,
        (MAX(view_count) - MIN(view_count)) / MIN(view_count) * 100 AS growth_rate
    FROM youtube_trending
    WHERE trending_date > '2022-02-25' 
    	AND publishedAt_year = 2022
    	AND (title LIKE '%갤럭시%' OR title LIKE '%galaxy%')
        AND title LIKE '%S22%'
    GROUP BY video_id, title, publishedAt
    ORDER BY publishedAt
)
SELECT AVG(growth_rate) AS average
FROM s22;

# S23 출시 전 조회수 평균 증가율
WITH s23 AS (
    SELECT
        publishedAt,
        video_id,
        title,
        (MAX(view_count) - MIN(view_count)) / MIN(view_count) * 100 AS growth_rate
    FROM youtube_trending
    WHERE trending_date <= '2023-02-17' 
    	AND publishedAt_year = 2023
    	AND (title LIKE '%갤럭시%' OR title LIKE '%galaxy%')
        AND title LIKE '%S23%'
    GROUP BY video_id, title, publishedAt
    ORDER BY publishedAt
)
SELECT AVG(growth_rate) AS average
FROM s23;

# S23 출시 후 조회수 평균 증가율
WITH s23 AS (
    SELECT
        publishedAt,
        video_id,
        title,
        (MAX(view_count) - MIN(view_count)) / MIN(view_count) * 100 AS growth_rate
    FROM youtube_trending
    WHERE trending_date > '2023-02-17' 
    	AND publishedAt_year = 2023
    	AND (title LIKE '%갤럭시%' OR title LIKE '%galaxy%')
        AND title LIKE '%S23%'
    GROUP BY video_id, title, publishedAt
    ORDER BY publishedAt
)
SELECT AVG(growth_rate) AS average
FROM s23;

# ZF3 출시 전 조회수 평균 증가율
WITH zf3 AS (
    SELECT
        publishedAt,
        video_id,
        title,
        (MAX(view_count) - MIN(view_count)) / MIN(view_count) * 100 AS growth_rate
    FROM youtube_trending
    WHERE trending_date <= '2021-08-27' 
        AND publishedAt_year = 2021
        AND title LIKE '%Z%'
        AND title LIKE '%3%'
        AND (title LIKE '%플립%' OR title LIKE '%flip%' OR title LIKE '%폴드%' OR title LIKE '%fold%')
    GROUP BY video_id, title, publishedAt
    ORDER BY publishedAt
)
SELECT AVG(growth_rate) AS average
FROM zf3;

# ZF3 출시 후 조회수 평균 증가율
WITH zf3 AS (
    SELECT
        publishedAt,
        video_id,
        title,
        (MAX(view_count) - MIN(view_count)) / MIN(view_count) * 100 AS growth_rate
    FROM youtube_trending
    WHERE trending_date > '2021-08-27' 
        AND publishedAt_year = 2021
        AND title LIKE '%Z%'
        AND title LIKE '%3%'
        AND (title LIKE '%플립%' OR title LIKE '%flip%' OR title LIKE '%폴드%' OR title LIKE '%fold%')
    GROUP BY video_id, title, publishedAt
    ORDER BY publishedAt
)
SELECT AVG(growth_rate) AS average
FROM zf3;


# ZF4 출시 전 조회수 평균 증가율
WITH zf4 AS (
    SELECT
        publishedAt,
        video_id,
        title,
        (MAX(view_count) - MIN(view_count)) / MIN(view_count) * 100 AS growth_rate
    FROM youtube_trending
    WHERE trending_date <= '2022-08-26' 
        AND publishedAt_year = 2022
        AND title LIKE '%Z%'
        AND title LIKE '%4%'
        AND (title LIKE '%플립%' OR title LIKE '%flip%' OR title LIKE '%폴드%' OR title LIKE '%fold%')
    GROUP BY video_id, title, publishedAt
    ORDER BY publishedAt
)
SELECT AVG(growth_rate) AS average
FROM zf4;

# ZF4 출시 후 조회수 평균 증가율
WITH zf4 AS (
    SELECT
        publishedAt,
        video_id,
        title,
        (MAX(view_count) - MIN(view_count)) / MIN(view_count) * 100 AS growth_rate
    FROM youtube_trending
    WHERE trending_date > '2022-08-26' 
        AND publishedAt_year = 2022
        AND title LIKE '%Z%'
        AND title LIKE '%4%'
        AND (title LIKE '%플립%' OR title LIKE '%flip%' OR title LIKE '%폴드%' OR title LIKE '%fold%')
    GROUP BY video_id, title, publishedAt
    ORDER BY publishedAt
)
SELECT AVG(growth_rate) AS average
FROM zf4;

# ZF5 출시 전 조회수 평균 증가율
WITH zf5 AS (
    SELECT
        publishedAt,
        video_id,
        title,
        (MAX(view_count) - MIN(view_count)) / MIN(view_count) * 100 AS growth_rate
    FROM youtube_trending
    WHERE trending_date <= '2023-08-11' 
        AND publishedAt_year = 2023
        AND title LIKE '%Z%'
        AND title LIKE '%5%'
        AND (title LIKE '%플립%' OR title LIKE '%flip%' OR title LIKE '%폴드%' OR title LIKE '%fold%')
    GROUP BY video_id, title, publishedAt
    ORDER BY publishedAt
)
SELECT AVG(growth_rate) AS average
FROM zf5;

# ZF5 출시 후 조회수 평균 증가율
WITH zf5 AS (
    SELECT
        publishedAt,
        video_id,
        title,
        (MAX(view_count) - MIN(view_count)) / MIN(view_count) * 100 AS growth_rate
    FROM youtube_trending
    WHERE trending_date > '2023-08-11' 
        AND publishedAt_year = 2023
        AND title LIKE '%Z%'
        AND title LIKE '%5%'
        AND (title LIKE '%플립%' OR title LIKE '%flip%' OR title LIKE '%폴드%' OR title LIKE '%fold%')
    GROUP BY video_id, title, publishedAt
    ORDER BY publishedAt
)
SELECT AVG(growth_rate) AS average
FROM zf5;

##################################################### 아이폰
# iphone13 출시 전 조회수 평균 증가율
WITH iphone13 AS (
    SELECT
        publishedAt,
        video_id,
        title,
        (MAX(view_count) - MIN(view_count)) / MIN(view_count) * 100 AS growth_rate
    FROM youtube_trending
    WHERE trending_date <= '2021-09-24' 
    	AND publishedAt_year = 2021
    	AND (title LIKE '%아이폰%' OR title LIKE '%iphone%')
        AND title LIKE '%13%'
        AND title NOT LIKE '%플립%'
    GROUP BY video_id, title, publishedAt
    ORDER BY publishedAt
)
SELECT AVG(growth_rate) AS average
FROM iphone13;

# iphone13 출시 후 조회수 평균 증가율
WITH iphone13 AS (
    SELECT
        publishedAt,
        video_id,
        title,
        (MAX(view_count) - MIN(view_count)) / MIN(view_count) * 100 AS growth_rate
    FROM youtube_trending
    WHERE trending_date > '2021-09-24' 
    	AND publishedAt_year = 2021
    	AND (title LIKE '%아이폰%' OR title LIKE '%iphone%')
        AND title LIKE '%13%'
        AND title NOT LIKE '%플립%'
    GROUP BY video_id, title, publishedAt
    ORDER BY publishedAt
)
SELECT AVG(growth_rate) AS average
FROM iphone13;

# iphone14 출시 전 조회수 평균 증가율
WITH iphone14 AS (
    SELECT
        publishedAt,
        video_id,
        title,
        (MAX(view_count) - MIN(view_count)) / MIN(view_count) * 100 AS growth_rate
    FROM youtube_trending
    WHERE trending_date <= '2022-09-16' 
    	AND publishedAt_year = 2022
    	AND (title LIKE '%아이폰%' OR title LIKE '%iphone%')
        AND title LIKE '%14%'
    GROUP BY video_id, title, publishedAt
    ORDER BY publishedAt
)
SELECT AVG(growth_rate) AS average
FROM iphone14;

# iphone14 출시 후 조회수 평균 증가율
WITH iphone14 AS (
    SELECT
        publishedAt,
        video_id,
        title,
        (MAX(view_count) - MIN(view_count)) / MIN(view_count) * 100 AS growth_rate
    FROM youtube_trending
    WHERE trending_date > '2022-09-16' 
    	AND publishedAt_year = 2022
    	AND (title LIKE '%아이폰%' OR title LIKE '%iphone%')
        AND title LIKE '%14%'
    GROUP BY video_id, title, publishedAt
    ORDER BY publishedAt
)
SELECT AVG(growth_rate) AS average
FROM iphone14;

# iphone15 출시 전 조회수 평균 증가율
WITH iphone15 AS (
    SELECT
        publishedAt,
        video_id,
        title,
        (MAX(view_count) - MIN(view_count)) / MIN(view_count) * 100 AS growth_rate
    FROM youtube_trending
    WHERE trending_date <= '2023-09-22' 
    	AND publishedAt_year = 2023
    	AND (title LIKE '%아이폰%' OR title LIKE '%iphone%')
        AND title LIKE '%15%'
    GROUP BY video_id, title, publishedAt
    ORDER BY publishedAt
)
SELECT AVG(growth_rate) AS average
FROM iphone15;

# iphone15 출시 후 조회수 평균 증가율
WITH iphone15 AS (
    SELECT
        publishedAt,
        video_id,
        title,
        (MAX(view_count) - MIN(view_count)) / MIN(view_count) * 100 AS growth_rate
    FROM youtube_trending
    WHERE trending_date > '2023-09-22' 
    	AND publishedAt_year = 2023
    	AND (title LIKE '%아이폰%' OR title LIKE '%iphone%')
        AND title LIKE '%15%'
    GROUP BY video_id, title, publishedAt
    ORDER BY publishedAt
)
SELECT AVG(growth_rate) AS average
FROM iphone15;