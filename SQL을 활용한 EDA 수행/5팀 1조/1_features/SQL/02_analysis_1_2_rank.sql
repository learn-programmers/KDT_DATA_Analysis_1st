with high as(
	select f.spotify_genre, f.danceability, dense_rank() over (order by f.danceability desc) as ranknum
	from devcourse_db.genre_and_features f
),
low as(
	select f.spotify_genre, f.danceability, dense_rank() over (order by f.danceability) as ranknum
	from devcourse_db.genre_and_features f
)
select *
from high
where ranknum < 83;
