-- 필요한 컬럼만 가져오기
-- index 컬럼은 이후 분석에서 테이블 join할 때 조건 컬럼으로 사용하기 위함
select f.index, f.spotify_genre, f.danceability, f.energy, f.key, f.loudness, f.mode, f.speechiness, f.acousticness, f.instrumentalness, f.liveness, f.valence, f.tempo, f.time_signature
from devcourse_db.features f
-- 장르 데이터 없는 행 제거
where not(f.spotify_genre is null or f.spotify_genre = '[]')
-- 오디오 정보 데이터 없는 행 제거
and not (f.danceability is null
	or f.energy is null
	or f.key is null
	or f.loudness is null
	or f.mode is null
	or f.speechiness is null
	or f.acousticness is null
	or f.instrumentalness is null
	or f.liveness is null
	or f.valence is null
	or f.tempo is null
	or f.time_signature is null)
-- 분석 데이터 행 개수 : 22739
