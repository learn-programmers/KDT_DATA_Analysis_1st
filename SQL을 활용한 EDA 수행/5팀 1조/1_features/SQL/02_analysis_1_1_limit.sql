select f.spotify_genre, f.danceability
from devcourse_db.genre_and_features f
order by f.danceability -- DESC
limit 20;

select f.spotify_genre, f.danceability
from devcourse_db.genre_and_features f
where f.danceability>= 0.9
order by f.danceability DESC;

select f.spotify_genre, f.danceability
from devcourse_db.genre_and_features f
where f.danceability < 0.2
order by f.danceability;