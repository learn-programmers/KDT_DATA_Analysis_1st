-- 작성자 : 최정혜

USE edadb;

-- 모든 데이터 확인
SELECT *
FROM student_data;

-- 총 레코드 개수 확인
SELECT count(*)
FROM student_data;

-- 결측치 확인 
SELECT *
FROM student_data
WHERE 1=1
AND school IS NULL AND sex IS NULL AND age IS NULL
AND address IS NULL AND famsize IS NULL AND Pstatus IS NULL
AND Medu IS NULL AND Fedu IS NULL AND Mjob IS NULL
AND Fjob IS NULL AND reason IS NULL AND guardian IS NULL
AND traveltime IS NULL AND studytime IS NULL AND failures IS NULL
AND schoolsup IS NULL AND famsup IS NULL AND paid IS NULL
AND activities IS NULL AND nursery IS NULL AND higher IS NULL
AND internet IS NULL AND romantic IS NULL AND famrel IS NULL
AND freetime IS NULL AND goout IS NULL AND Dalc IS NULL
AND Walc IS NULL AND health IS NULL AND absences IS NULL
AND G1 IS NULL AND G2 IS NULL AND G3 IS NULL;

-- 성별에 따른 학생 수
SELECT (CASE WHEN sex = 'M' THEN 'Male' ELSE 'Female' END) as '성별', 
COUNT(*) as '학생 수'
FROM student_data
GROUP BY 1;

-- 학교에 따른 학생 수 
SELECT school, COUNT(*) as num_of_students
FROM student_data
GROUP BY school;

-- 나이 별 학생 수  
SELECT age, COUNT(*) as num_of_students
FROM student_data
GROUP By 1
ORDER BY 1;

-- 나이 별 평균 성적  
SELECT age, COUNT(*) as num_of_students, AVG(g3) as avg_score
FROM student_data
GROUP BY 1
ORDER BY 1;

-- 성별 별 평균 성적 
SELECT sex as '성별', AVG(g3) as '평균 성적'
FROM student_data
GROUP BY 1; 

-- 주소 별 학생 수 
SELECT address, COUNT(*) as num_of_students
FROM student_data
GROUP BY 1;



-- 부모 관련 요인 

-- 부모의 교육수준 별 평균 성적

-- 아버지의 교육수준
SELECT fedu as fathers_edu, AVG(g3) as avg_score
FROM student_data
GROUP BY 1
ORDER BY 2 DESC;

-- 어머니의 교육수준 
SELECT medu as mothers_edu, AVG(g3) as avg_score
FROM student_data
GROUP BY medu
ORDER BY avg_score DESC;

-- 부모의 동거 여부 별 평균 성적 
SELECT (CASE
	WHEN Pstatus = 'A' THEN '별거' 
    ELSE '동거' END) as parental_stat, 
    AVG(g3) as avg_score
FROM student_data
GROUP BY parental_stat
ORDER BY avg_score DESC;

-- 성별 별 부모의 동거 여부에 따른 평균 성적 
SELECT Pstatus as parental_stat, sex, AVG(g3) as avg_score
FROM student_data
GROUP BY 1, 2
ORDER BY avg_score DESC;

-- 부모가 별거 중인 남학생의 보호자 별 평균 성적 
SELECT Pstatus as parental_stat, sex, guardian, 
AVG(g3) as avg_score
FROM student_data
WHERE sex = 'M' AND Pstatus = 'A'
GROUP BY 1, 3
ORDER BY avg_score DESC;

-- 부모가 별거 중인 여학생의 보호자 별 평균 성적 
SELECT Pstatus as parental_stat, sex, guardian, 
AVG(g3) as avg_score
FROM student_data
WHERE sex = 'F' AND Pstatus = 'A'
GROUP BY 1, 3
ORDER BY avg_score DESC;



-- 가정환경 관련 요인 

-- 주소의 특성에 따른 성별 별 평균 성적 
SELECT (CASE WHEN address = 'U' THEN '도시' ELSE '시골' END) as '주소',
	sex as '성별',
	AVG(g3) as avg_score
FROM student_data
GROUP BY 1, 2
ORDER BY 2;

-- 등하교 시간에 따른 성별 별 평균 성적 
SELECT traveltime as '등하교시간', sex as '성별', AVG(g3) as '평균 성적'
FROM student_data
GROUP BY 1, 2
ORDER BY 1;



-- 경제적 요인 

-- 과외 유무에 따른 성별 별 평균 성적 
SELECT paid, sex, AVG(g3) as avg_score
FROM student_data
GROUP BY 1, 2
ORDER BY avg_score DESC;

-- 대외활동 여부에 따른 성별 별 평균 성적 
SELECT activities, sex, AVG(g3) as avg_score
FROM student_data
GROUP BY 1, 2;



-- 최종 학기에 0점을 받은 학생들 

-- G3가 0인 학생의 수 
SELECT COUNT(*) as '0을 받은 학생의 수'
FROM student_data
WHERE g3 = 0;

-- G2도 0인 학생의 수
SELECT COUNT(*) as '0을 받은 학생의 수'
FROM student_data
WHERE g3 = 0 AND g2 = 0;

-- G3가 0점인 학생들의 성별에 따른 G1(1학기) 평균 성적 
WITH zero_g3 as (
	SELECT * FROM student_data
    WHERE g3 = 0
)
SELECT sex as '성별', AVG(g1) as '1학기 평균 성적'
FROM zero_g3
GROUP BY 1
ORDER BY 2;

-- 가족관계와 결석일수에 따른 학생 수 
WITH zero_g3 as (
	SELECT * FROM student_data
    WHERE g3 = 0
)
SELECT famrel as '가족관계', absences as '결석일수', COUNT(*) as '학생 수'
FROM zero_g3
GROUP BY 1, 2
ORDER BY 1;

-- G3가 0점인 학생들 중 과거 F를 받은 수업의 수에 따른 학생 수 
WITH zero_g3 as (
	SELECT * FROM student_data
    WHERE g3 = 0
)
SELECT failures as '과거 F를 받은 수업의 수', 
COUNT(*) as '학생의 수'
FROM zero_g3
GROUP BY 1
ORDER BY 1;

-- G3가 0점이 아닌 학생들 중 과거 F를 받은 수업의 수에 따른 학생 수
WITH existing_g3 as (
	SELECT * FROM student_data
    WHERE g3 != 0
)
SELECT failures as '과거 F를 받은 수업의 수', 
COUNT(*) as '학생의 수'
FROM existing_g3
GROUP BY 1
ORDER BY 1;

