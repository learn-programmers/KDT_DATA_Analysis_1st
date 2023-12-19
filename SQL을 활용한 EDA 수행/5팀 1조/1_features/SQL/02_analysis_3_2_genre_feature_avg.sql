create table if not exists devcourse_db.genre_avg (
    genre	varchar(30),
    cnt		int,
    avg_dance	double,
    avg_energy	double,
    avg_key		int,
    avg_loud	double,
    avg_mode	tinyint,
    avg_speech	double,
    avg_acou	double,
    avg_instru	double,
    avg_live	double,
    avg_val		double,
    avg_tempo	double,
    avg_ts		int,
    primary key(genre)
);

insert into devcourse_db.genre_avg(
	with tb_genre as (
	select *
	from devcourse_db.filterdata f
	where f.spotify_genre like "%'pop'%"
	)
    select 'pop',
		count(tb_genre.danceability),
		round(avg(tb_genre.danceability), 3),
		round(avg(tb_genre.energy), 3),
		avg(tb_genre.key0),
		round(avg(tb_genre.loudness), 3),
		avg(tb_genre.mode0),
		round(avg(tb_genre.speechiness), 3),
		round(avg(tb_genre.acousticness), 3),
		round(avg(tb_genre.instrumentalness), 3),
		round(avg(tb_genre.liveness), 3),
		round(avg(tb_genre.valence), 3),
		round(avg(tb_genre.tempo), 3),
		avg(tb_genre.time_signature)
	from tb_genre
);

select * from billboard_db.genre_avg;

select round(avg(avg_dance), 3) from devcourse_db.genre_avg;

select genre, avg_dance
from billboard_db.genre_avg
order by avg_dance desc;