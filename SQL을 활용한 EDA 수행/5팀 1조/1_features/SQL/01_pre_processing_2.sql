create table if not exists devcourse_db.genre_and_features
-- 소수점 데이터 정리, index & key & mode 컬럼명 변경 (MySQL 예약어여서 사용불가)
select f.index as index0,
	f.spotify_genre,
	round(f.danceability, 3) as danceability,
	round(f.energy, 3) as energy,
	round(f.key, 3) as key0,
	round(f.loudness, 3) as loudness,
	round(f.mode, 3) as mode0,
	round(f.speechiness, 3) as speechiness,
	round(f.acousticness, 3) as acousticness,
	round(f.instrumentalness, 3) as instrumentalness,
	round(f.liveness, 3) as liveness,
	round(f.valence, 3) as valence,
	round(f.tempo, 3) as tempo,
	round(f.time_signature, 3) as time_signature
from billboard_db.features f
where not(f.spotify_genre is null or f.spotify_genre = '[]')
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
	or f.time_signature is null);
