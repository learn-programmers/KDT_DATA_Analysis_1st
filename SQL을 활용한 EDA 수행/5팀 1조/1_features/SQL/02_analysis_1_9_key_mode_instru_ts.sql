select f.spotify_genre, f.key0
from devcourse_db.genre_and_features f
where f.key0 = 0;


select f.spotify_genre, f.mode0
from devcourse_db.genre_and_features f
-- where f.mode0 = 1
where f.mode0 = 0;


select f.spotify_genre, f.instrumentalness
from devcourse_db.genre_and_features f
-- where f.instrumentalness = 0
where f.instrumentalness != 0;


select f.spotify_genre, f.time_signature
from devcourse_db.genre_and_features f
where f.time_signature != 4;