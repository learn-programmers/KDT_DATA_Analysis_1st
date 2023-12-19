## 갤럭시 s21 출시 전
WITH daily_view_counts AS (
    SELECT
        video_id,
        title,
        publishedAt,
        trending_date,
        view_count,
        LAG(view_count) OVER (PARTITION BY video_id ORDER BY trending_date) AS prev_view_count
    FROM
        youtube_trending
    WHERE
        publishedAt like '2021%'      
    AND title LIKE '%S21%'
)
SELECT
    video_id, 
    title,
    publishedAt,
    trending_date, 
    view_count, 
    prev_view_count,
    CASE
        WHEN prev_view_count IS NULL THEN 0 
        ELSE round(((view_count - prev_view_count) / prev_view_count) * 100, 3)
    END AS view_count_increase_rate
FROM
    daily_view_counts
WHERE trending_date <= '2021-01-29%'
ORDER BY
    video_id, trending_date;

## 갤럭시 s21 출시 후
WITH daily_view_counts AS (
    SELECT
        video_id,
        title,
        publishedAt,
        trending_date,
        view_count,
        LAG(view_count) OVER (PARTITION BY video_id ORDER BY trending_date) AS prev_view_count
    FROM
        youtube_trending
    WHERE
        publishedAt like '2021%'      
    HAVING (title LIKE '%갤럭시%' or title LIKE '%galaxy%')
    AND title LIKE '%S21%'
)
SELECT
    video_id, 
    title,
    publishedAt,
    trending_date, 
    view_count, 
    prev_view_count,
    CASE
        WHEN prev_view_count IS NULL THEN 0 
        ELSE round(((view_count - prev_view_count) / prev_view_count) * 100, 3)
    END AS view_count_increase_rate
FROM
    daily_view_counts
WHERE trending_date > '2021-01-29%'
ORDER BY
    video_id, trending_date;

   
## 갤럭시 s22 출시 전
WITH daily_view_counts AS (
    SELECT
        video_id,
        title,
        publishedAt,
        trending_date,
        view_count,
        LAG(view_count) OVER (PARTITION BY video_id ORDER BY trending_date) AS prev_view_count
    FROM
        youtube_trending
    WHERE
        publishedAt like '2022%'      
    HAVING (title LIKE '%갤럭시%' or title LIKE '%galaxy%')
    AND title LIKE '%S22%'
)
SELECT
    video_id, 
    title,
    publishedAt,
    trending_date, 
    view_count, 
    prev_view_count,
    CASE
        WHEN prev_view_count IS NULL THEN 0 
        ELSE round(((view_count - prev_view_count) / prev_view_count) * 100, 3)
    END AS view_count_increase_rate
FROM
    daily_view_counts
WHERE trending_date <= '2022-02-25%'
ORDER BY
    video_id, trending_date;
    
## 갤럭시 s22 출시 후
WITH daily_view_counts AS (
    SELECT
        video_id,
        title,
        publishedAt,
        trending_date,
        view_count,
        LAG(view_count) OVER (PARTITION BY video_id ORDER BY trending_date) AS prev_view_count
    FROM
        youtube_trending
    WHERE
        publishedAt like '2022%' 
    HAVING (title LIKE '%갤럭시%' or title LIKE '%galaxy%')
    AND title LIKE '%S22%'
)
SELECT
    video_id, 
    title,
    publishedAt,
    trending_date, 
    view_count, 
    prev_view_count,
    CASE
        WHEN prev_view_count IS NULL THEN 0 
        ELSE round(((view_count - prev_view_count) / prev_view_count) * 100, 3)
    END AS view_count_increase_rate
FROM
    daily_view_counts
WHERE trending_date > '2022-02-25%'
ORDER BY
    video_id, trending_date;
    
## 갤럭시 s23 출시 전
WITH daily_view_counts AS (
    SELECT
        video_id,
        title,
        publishedAt,
        trending_date,
        view_count,
        LAG(view_count) OVER (PARTITION BY video_id ORDER BY trending_date) AS prev_view_count
    FROM
        youtube_trending
    WHERE
        publishedAt like '2023%'      
    HAVING (title LIKE '%갤럭시%' or title LIKE '%galaxy%')
    AND title LIKE '%S23%'
)
SELECT
    video_id, 
    title,
    publishedAt,
    trending_date, 
    view_count, 
    prev_view_count,
    CASE
        WHEN prev_view_count IS NULL THEN 0 
        ELSE round(((view_count - prev_view_count) / prev_view_count) * 100, 3)
    END AS view_count_increase_rate
FROM
    daily_view_counts
WHERE trending_date <= '2023-02-17%'
ORDER BY
    video_id, trending_date;
    
## 갤럭시 s23 출시 후
WITH daily_view_counts AS (
    SELECT
        video_id,
        title,
        publishedAt,
        trending_date,
        view_count,
        LAG(view_count) OVER (PARTITION BY video_id ORDER BY trending_date) AS prev_view_count
    FROM
        youtube_trending
    WHERE
        publishedAt like '2023%'      
    HAVING (title LIKE '%갤럭시%' or title LIKE '%galaxy%')
    AND title LIKE '%S23%'
)
SELECT
    video_id, 
    title,
    publishedAt,
    trending_date, 
    view_count, 
    prev_view_count,
    CASE
        WHEN prev_view_count IS NULL THEN 0 
        ELSE round(((view_count - prev_view_count) / prev_view_count) * 100, 3)
    END AS view_count_increase_rate
FROM
    daily_view_counts
WHERE trending_date > '2023-02-17%'
ORDER BY
    video_id, trending_date;

## 갤럭시 zf3 출시 전
WITH daily_view_counts AS (
    SELECT
        video_id,
        title,
        publishedAt,
        trending_date,
        view_count,
        LAG(view_count) OVER (PARTITION BY video_id ORDER BY trending_date) AS prev_view_count
    FROM
        youtube_trending
    WHERE
        publishedAt like '2021%'      
    HAVING (title LIKE '%Z%' AND title LIKE '%3%')
	AND (title LIKE '%플립%' OR title LIKE '%flip%' OR title LIKE '%폴드%' OR title LIKE '%fold%')
)
SELECT
    video_id, 
    title,
    publishedAt,
    trending_date, 
    view_count, 
    prev_view_count,
    CASE
        WHEN prev_view_count IS NULL THEN 0 
        ELSE round(((view_count - prev_view_count) / prev_view_count) * 100, 3)
    END AS view_count_increase_rate
FROM
    daily_view_counts
WHERE trending_date <= '2021-08-27%'
ORDER BY
    video_id, trending_date;
   
   
## 갤럭시 zf3 출시 후
WITH daily_view_counts AS (
    SELECT
        video_id,
        title,
        publishedAt,
        trending_date,
        view_count,
        LAG(view_count) OVER (PARTITION BY video_id ORDER BY trending_date) AS prev_view_count
    FROM
        youtube_trending
    WHERE
        publishedAt like '2021%'    
    HAVING (title LIKE '%Z%' AND title LIKE '%3%')
	AND (title LIKE '%플립%' OR title LIKE '%flip%' OR title LIKE '%폴드%' OR title LIKE '%fold%')
)
SELECT
    video_id, 
    title,
    publishedAt,
    trending_date, 
    view_count, 
    prev_view_count,
    CASE
        WHEN prev_view_count IS NULL THEN 0 
        ELSE round(((view_count - prev_view_count) / prev_view_count) * 100, 3)
    END AS view_count_increase_rate
FROM
    daily_view_counts
WHERE trending_date > '2021-08-27%'
ORDER BY
    video_id, trending_date;
   
   
## 갤럭시 zf4 출시 전
WITH daily_view_counts AS (
    SELECT
        video_id,
        title,
        publishedAt,
        trending_date,
        view_count,
        LAG(view_count) OVER (PARTITION BY video_id ORDER BY trending_date) AS prev_view_count
    FROM
        youtube_trending
    WHERE
        publishedAt like '2022%'     
    HAVING (title LIKE '%Z%' AND title LIKE '%4%')
	AND (title LIKE '%플립%' OR title LIKE '%flip%' OR title LIKE '%폴드%' OR title LIKE '%fold%')
)
SELECT
    video_id, 
    title,
    publishedAt,
    trending_date, 
    view_count, 
    prev_view_count,
    CASE
        WHEN prev_view_count IS NULL THEN 0 
        ELSE round(((view_count - prev_view_count) / prev_view_count) * 100, 3)
    END AS view_count_increase_rate
FROM
    daily_view_counts
WHERE trending_date <= '2022-08-27%'
ORDER BY
    video_id, trending_date;

## 갤럭시 zf4 출시 후
WITH daily_view_counts AS (
    SELECT
        video_id,
        title,
        publishedAt,
        trending_date,
        view_count,
        LAG(view_count) OVER (PARTITION BY video_id ORDER BY trending_date) AS prev_view_count
    FROM
        youtube_trending
    WHERE
        publishedAt like '2022%'     
    HAVING (title LIKE '%Z%' AND title LIKE '%4%')
	AND (title LIKE '%플립%' OR title LIKE '%flip%' OR title LIKE '%폴드%' OR title LIKE '%fold%')
)
SELECT
    video_id, 
    title,
    publishedAt,
    trending_date, 
    view_count, 
    prev_view_count,
    CASE
        WHEN prev_view_count IS NULL THEN 0 
        ELSE round(((view_count - prev_view_count) / prev_view_count) * 100, 3)
    END AS view_count_increase_rate
FROM
    daily_view_counts
WHERE trending_date > '2022-08-27%'
ORDER BY
    video_id, trending_date;
   

## 갤럭시 zf5 출시 전
WITH daily_view_counts AS (
    SELECT
        video_id,
        title,
        publishedAt,
        trending_date,
        view_count,
        LAG(view_count) OVER (PARTITION BY video_id ORDER BY trending_date) AS prev_view_count
    FROM
        youtube_trending
    WHERE
        publishedAt like '2023%'     
    HAVING (title LIKE '%Z%' AND title LIKE '%5%')
	AND (title LIKE '%플립%' OR title LIKE '%flip%' OR title LIKE '%폴드%' OR title LIKE '%fold%')
)
SELECT
    video_id, 
    title,
    publishedAt,
    trending_date, 
    view_count, 
    prev_view_count,
    CASE
        WHEN prev_view_count IS NULL THEN 0 
        ELSE round(((view_count - prev_view_count) / prev_view_count) * 100, 3)
    END AS view_count_increase_rate
FROM
    daily_view_counts
WHERE trending_date <= '2023-08-11%'
ORDER BY
    video_id, trending_date;
   
## 갤럭시 zf5 출시 후
WITH daily_view_counts AS (
    SELECT
        video_id,
        title,
        publishedAt,
        trending_date,
        view_count,
        LAG(view_count) OVER (PARTITION BY video_id ORDER BY trending_date) AS prev_view_count
    FROM
        youtube_trending
    WHERE
        publishedAt like '2023%'     
    HAVING (title LIKE '%Z%' AND title LIKE '%5%')
	AND (title LIKE '%플립%' OR title LIKE '%flip%' OR title LIKE '%폴드%' OR title LIKE '%fold%')
)
SELECT
    video_id, 
    title,
    publishedAt,
    trending_date, 
    view_count, 
    prev_view_count,
    CASE
        WHEN prev_view_count IS NULL THEN 0 
        ELSE round(((view_count - prev_view_count) / prev_view_count) * 100, 3)
    END AS view_count_increase_rate
FROM
    daily_view_counts
WHERE trending_date > '2023-08-11%'
ORDER BY
    video_id, trending_date;
    
   
## 아이폰 13 출시 전
WITH daily_view_counts AS (
    SELECT
        video_id,
        title,
        publishedAt,
        trending_date,
        view_count,
        LAG(view_count) OVER (PARTITION BY video_id ORDER BY trending_date) AS prev_view_count
    FROM
        youtube_trending
    WHERE
        publishedAt like '2021%' 
    HAVING (title LIKE '%아이폰%' OR title LIKE '%iphone%')
	AND title LIKE '%13%'
)
SELECT
    video_id, 
    title,
    publishedAt,
    trending_date, 
    view_count, 
    prev_view_count,
    CASE
        WHEN prev_view_count IS NULL THEN 0 
        ELSE round(((view_count - prev_view_count) / prev_view_count) * 100, 3)
    END AS view_count_increase_rate
FROM
    daily_view_counts
WHERE trending_date <= '2021-09-24%'
ORDER BY
    video_id, trending_date;
    
## 아이폰 13 출시 후
WITH daily_view_counts AS (
    SELECT
        video_id,
        title,
        publishedAt,
        trending_date,
        view_count,
        LAG(view_count) OVER (PARTITION BY video_id ORDER BY trending_date) AS prev_view_count
    FROM
        youtube_trending
    WHERE
        publishedAt like '2021%'  
    HAVING (title LIKE '%아이폰%' OR title LIKE '%iphone%')
	AND title LIKE '%13%'
)
SELECT
    video_id, 
    title,
    publishedAt,
    trending_date, 
    view_count, 
    prev_view_count,
    CASE
        WHEN prev_view_count IS NULL THEN 0 
        ELSE round(((view_count - prev_view_count) / prev_view_count) * 100, 3)
    END AS view_count_increase_rate
FROM
    daily_view_counts
WHERE trending_date > '2021-09-24%'
ORDER BY
    video_id, trending_date;
    
## 아이폰 14 출시 전
WITH daily_view_counts AS (
    SELECT
        video_id,
        title,
        publishedAt,
        trending_date,
        view_count,
        LAG(view_count) OVER (PARTITION BY video_id ORDER BY trending_date) AS prev_view_count
    FROM
        youtube_trending
    WHERE
        publishedAt like '2022%'   
    HAVING (title LIKE '%아이폰%' OR title LIKE '%iphone%')
	AND title LIKE '%14%'
)
SELECT
    video_id, 
    title,
    publishedAt,
    trending_date, 
    view_count,
    prev_view_count,
    CASE
        WHEN prev_view_count IS NULL THEN 0 
        ELSE round(((view_count - prev_view_count) / prev_view_count) * 100, 3)
    END AS view_count_increase_rate
FROM
    daily_view_counts
WHERE trending_date <= '2022-09-16%'
ORDER BY
    video_id, trending_date;

## 아이폰 14 출시 후
WITH daily_view_counts AS (
    SELECT
        video_id,
        title,
        publishedAt,
        trending_date,
        view_count,
        LAG(view_count) OVER (PARTITION BY video_id ORDER BY trending_date) AS prev_view_count
    FROM
        youtube_trending
    WHERE
        publishedAt like '2022%'   
    HAVING (title LIKE '%아이폰%' OR title LIKE '%iphone%')
	AND title LIKE '%14%'
)
SELECT
    video_id, 
    title,
    publishedAt,
    trending_date, 
    view_count, 
    prev_view_count,
    CASE
        WHEN prev_view_count IS NULL THEN 0 
        ELSE round(((view_count - prev_view_count) / prev_view_count) * 100, 3)
    END AS view_count_increase_rate
FROM
    daily_view_counts
WHERE trending_date > '2022-09-16%'
ORDER BY
    video_id, trending_date;
    
   
## 아이폰 15 출시 전
WITH daily_view_counts AS (
    SELECT
        video_id,
        title,
        publishedAt,
        trending_date,
        view_count,
        LAG(view_count) OVER (PARTITION BY video_id ORDER BY trending_date) AS prev_view_count
    FROM
        youtube_trending
    WHERE
        publishedAt like '2023%' 
    HAVING (title LIKE '%아이폰%' OR title LIKE '%iphone%')
	AND title LIKE '%15%'
)
SELECT
    video_id, 
    title,
    publishedAt,
    trending_date, 
    view_count, 
    prev_view_count,
    CASE
        WHEN prev_view_count IS NULL THEN 0 
        ELSE round(((view_count - prev_view_count) / prev_view_count) * 100, 3)
    END AS view_count_increase_rate
FROM
    daily_view_counts
WHERE trending_date <= '2023-09-22%'
ORDER BY
    video_id, trending_date;

   
## 아이폰 15 출시 후
WITH daily_view_counts AS (
    SELECT
        video_id,
        title,
        publishedAt,
        trending_date,
        view_count,
        LAG(view_count) OVER (PARTITION BY video_id ORDER BY trending_date) AS prev_view_count
    FROM
        youtube_trending
    WHERE
        publishedAt like '2023%'    
    HAVING (title LIKE '%아이폰%' OR title LIKE '%iphone%')
	AND title LIKE '%15%'
)
SELECT
    video_id, 
    title,
    publishedAt,
    trending_date, 
    view_count, 
    prev_view_count,
    CASE
        WHEN prev_view_count IS NULL THEN 0 
        ELSE round(((view_count - prev_view_count) / prev_view_count) * 100, 3)
    END AS view_count_increase_rate
FROM
    daily_view_counts
WHERE trending_date > '2023-09-22%'
ORDER BY
    video_id, trending_date;