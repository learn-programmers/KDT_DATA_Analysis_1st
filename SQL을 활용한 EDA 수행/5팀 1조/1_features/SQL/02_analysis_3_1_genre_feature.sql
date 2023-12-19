with pop as(
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%'pop'%"
),
rock as(
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%'rock'%"
),
indie as(
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%indie%"
),
hippop as(
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%'hip pop'%"
	or f.spotify_genre like "%'rap'%"
	or f.spotify_genre like "%'trap'%"
),
jazz as(
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%jazz%"
),
metal as(
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%metal%"
),
punk as(
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%punk%"
),
country as(
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%'country'%"
),
soul as(
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%'blues'%"
	or f.spotify_genre like "%soul%"
),
house as(
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%house%"
	or f.spotify_genre like "%electronic%"
),
folk as(
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%punk%"
),
funk as(
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%funk%"
),
christian as(
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%christian%"
	or f.spotify_genre like "%gospel%"
),
rnb as(
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%r&b%"
	or f.spotify_genre like "%rhythm and blues%"
),
dance as(
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%dance%"
),
latin as(
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%latin%"
),
adult_standards as(
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%'adult standards'%"
),
mellow_gold as(
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%'mellow gold'%"
),
brill_building_pop as(
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%'brill building pop'%"
),
easy_listening as(
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%'easy listening'%"
),
motown as (
	select *
	from devcourse_db.genre_and_features f
	where f.spotify_genre like "%'motown'%"
)
-- select * from pop;
select energy as motown from motown;