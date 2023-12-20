select f.song, s.weekID, f.tempo, f.spotify_genre,
    CASE
        WHEN f.spotify_genre LIKE '%dance%' THEN 'dance'
        WHEN f.spotify_genre LIKE '%pop%' THEN 'pop'
        WHEN f.spotify_genre LIKE '%indie%' THEN 'indie'
        WHEN f.spotify_genre LIKE '%hip hop%' or '%rap%' or '%trap%'THEN 'hip pop'
        WHEN f.spotify_genre LIKE '%jazz%' THEN 'jazz'
        WHEN f.spotify_genre LIKE '%metal%' THEN 'metal'
        WHEN f.spotify_genre LIKE '%punk%' THEN 'punk'
        WHEN f.spotify_genre LIKE '%country%' THEN 'country'
        WHEN f.spotify_genre LIKE '%blues%' or '%soul%'THEN 'blues'
        WHEN f.spotify_genre LIKE '%house%' or '%electronic%'THEN 'house'
        WHEN f.spotify_genre LIKE '%folk%' THEN 'folk'
        WHEN f.spotify_genre LIKE '%funk%' THEN 'funk'
        WHEN f.spotify_genre LIKE '%christian%' or '%gospel%'THEN 'christian'
        WHEN f.spotify_genre LIKE '%r&b%' or '%rhythm and blues%'THEN 'r&b'
        WHEN f.spotify_genre LIKE '%dance%' THEN 'dance'
        WHEN f.spotify_genre LIKE '%latin%' THEN 'latin'
        WHEN f.spotify_genre LIKE '%adult standards%' THEN 'adult standards'
        WHEN f.spotify_genre LIKE '%nellow gold%' THEN 'nellow gold'
        WHEN f.spotify_genre LIKE '%brill building pop%' THEN 'latin'
        WHEN f.spotify_genre LIKE '%adult standards%' THEN 'adult standards'
        WHEN f.spotify_genre LIKE '%motown%' THEN 'motown'
        
        ELSE '기타'
    END AS genre
    
from devcourse_db.features f inner join devcourse_db.stuff s
	on f.song = s.song
where 1=1
and not f.spotify_genre is null
and not f.tempo is null
