select max(f.danceability) as max_dance,
	max(f.energy) as max_energy,
	max(f.loudness) as max_loud,
	max(f.speechiness) as max_speech,
	max(f.acousticness) as max_acoustic,
	max(f.liveness) as max_live,
	max(f.valence) as max_valence,
	max(f.tempo) as max_tempo
from devcourse_db.genre_and_features f;

select min(f.danceability) as min_dance,
	min(f.energy) as min_energy,
	min(f.loudness) as min_loud,
	min(f.speechiness) as min_speech,
	min(f.acousticness) as min_acoustic,
	min(f.liveness) as min_live,
	min(f.valence) as min_valence,
	min(f.tempo) as min_tempo
from devcourse_db.genre_and_features f;
