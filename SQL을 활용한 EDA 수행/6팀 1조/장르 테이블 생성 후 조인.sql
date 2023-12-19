-- 데이터베이스 확인
SHOW DATABASES;

-- 특정 데이터베이스 사용
USE test;

-- 테이블 확인
SHOW TABLES;

-- 해당 테이블 스키마 확인
DESC youtube_trending; ## 몇 개의 컬럼을 제거한 유튜브 테이블

-- 데이터 확인 - 총 236265 행
SELECT youtube_trending.*
FROM youtube_trending;

-- category 테이블 생성
CREATE TABLE IF NOT EXISTS `category` (
  `categoryId` INT UNSIGNED NOT NULL,
  `category` VARCHAR(30) NOT NULL,
  PRIMARY KEY (`categoryId`)
) DEFAULT CHARSET=UTF8;

-- 해당 테이블 스키마 확인
DESC category; ## 장르 테이블

-- 데이터 삽입
INSERT INTO `category` (`categoryId`, `category`) VALUES
  (1, "Film & Animation"),
  (2, "Autos & Vehicles"),
  (10, "Music"),
  (15, "Pets & Animals"),
  (17, "Sports"),
  (18, "Short Movies"),
  (19, "Travel & Events"),
  (20, "Gaming"),
  (21, "Videoblogging"),
  (22, "People & Blogs"),
  (23, "Comedy"),
  (24, "Entertainment"),
  (25, "News & Politics"),
  (26, "Howto & Style"),
  (27, "Education"),
  (28, "Science & Technology"),
  (30, "Movies"),
  (31, "Anime/Animation"),
  (32, "Action/Adventure"),
  (33, "Classics"),
  (34, "Comedy"),
  (35, "Documentary"),
  (36, "Drama"),
  (37, "Family"),
  (38, "Foreign"),
  (39, "Horror"),
  (40, "Sci-Fi/Fantasy"),
  (41, "Thriller"),
  (42, "Shorts"),
  (43, "Shows"),
  (44, "Trailers");

-- 데이터 확인
SELECT category.*
FROM category
LIMIT 10;

-- youtube_trending 테이블에 category 컬럼 추가
ALTER TABLE youtube_trending
ADD COLUMN category VARCHAR(30);

-- 해당 테이블 스키마 확인
DESC youtube_trending; ## 몇 개의 컬럼을 제거한 유튜브 테이블

-- category 테이블을 사용하여 데이터 업데이트
UPDATE youtube_trending
JOIN category ON youtube_trending.categoryId = category.categoryId
SET youtube_trending.category = category.category;