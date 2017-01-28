-- Load up the million song dataset from S3 (see data spec at: ht
SET mapreduce.map.java.opts -Xmx2048m;
SET mapreduce.reduce.java.opts -Xmx2048m;
set mapred.child.java.opts -Xmx2048m;
set pig.exec.nocombiner true;
set default_parallel 35;

-- Load up the million song dataset from S3 (see data spec at: http://bit.ly/vOBKPe)
-- 's3n://tbmmsd/A.tsv.*'
songs = LOAD '/Users/Zipfian/datasets/YearPredictionMSD.txt' USING PigStorage('\t') AS (
         track_id:chararray, analysis_sample_rate:chararray, artist_7digitalid:chararray,
         artist_familiarity:chararray, artist_hotness:double, artist_id:chararray, artist_latitude:chararray, 
         artist_location:chararray, artist_longitude:chararray, artist_mbid:chararray, artist_mbtags:chararray, 
         artist_mbtags_count:chararray, artist_name:chararray, artist_playmeid:chararray, artist_terms:chararray, 
         artist_terms_freq:chararray, artist_terms_weight:chararray, audio_md5:chararray, bars_confidence:chararray, 
         bars_start:chararray, beats_confidence:chararray, beats_start:chararray, danceability:double, 
         duration:float, end_of_fade_in:chararray, energy:chararray, key:chararray, key_confidence:chararray, 
         loudness:chararray, mode:chararray, mode_confidence:chararray, release:chararray, 
         release_7digitalid:chararray, sections_confidence:chararray, sections_start:chararray, 
         segments_confidence:chararray, segments_loudness_max:chararray, segments_loudness_max_time:chararray, 
         segments_loudness_max_start:chararray, segments_pitches:chararray, segments_start:chararray, 
         segments_timbre:chararray, similar_artists:chararray, song_hotness:double, song_id:chararray, 
         start_of_fade_out:chararray, tatums_confidence:chararray, tatums_start:chararray, tempo:double, 
         time_signature:chararray, time_signature_confidence:chararray, title:chararray, track_7digitalid:chararray, 
         year:int );

-- Filter out the uneccessary columns
filtered = foreach songs generate track_id as id, STRSPLIT(artist_terms, ',') as genre, analysis_sample_rate as sample_rate, artist_location as loc, artist_name as artist, danceability as dance, duration as duration, energy as energy, key as key, similar_artists as sim_artist, tempo as tempo, time_signature as signature, loudness as loudness, mode as mode, song_hotness as song_hotness, artist_hotness as artist_hotness, title, year;

-- extract the most prevalent genre
genres = foreach filtered generate id as id, genre.$0 as genre, $2..;

-- group each song by common genre
grouped = group genres by genre;

-- count how many of each genre there is
counts = foreach grouped generate group as key, COUNT(genres) as count;

-- Order songs by most frequent genre 
ordered = order counts by count desc;

-- Only look at the top 50 
top50 = limit ordered 50; 

-- store our data
store ordered into 's3n://zipfiandata/order';
store top50 into 's3n://zipfiandata/top50';
