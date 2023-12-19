select round(avg(f.danceability), 3) as avg_dance,
	round(avg(f.energy), 3) as avg_energy,
    round(avg(f.loudness), 3) as avg_loud,
    round(avg(f.speechiness), 3) as avg_speech,
    round(avg(f.acousticness), 3) as avg_acoustic,
    round(avg(f.liveness), 3) as avg_live,
    round(avg(f.valence), 3) as avg_valence,
    round(avg(f.tempo), 3) as avg_tempo
from devcourse_db.genre_and_features f;