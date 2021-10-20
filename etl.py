import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
import pyspark.sql.functions as F


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    '''
        create spark session with hadoop package
        which will be able to interact with S3. 
    '''
    
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    '''
        pull song data from S3, process data to create songs, artists table 
        then load these tables to S3 as parquet format.   

        Args:
            spark: spark session with hadoop package
            input_data: the path of S3 buckect which stores song data
            output_data: the path of S3 buckect which stores processed data  
    '''
    
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data', '*', '*', '*')
    
    # read song data file
    df_song = spark.read.json(song_data) 

    # extract columns to create songs table
    songs_table = df_song\
                .select(['song_id', 'title', 'artist_id', 'year', 'duration'])\
                .dropDuplicates(['song_id']) 
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('artist_id', 'year').parquet(path = os.path.join(output_data, 'song.parquet'), mode = 'overwrite')

    # extract columns to create artists table
    artists_table = df_song\
                .selectExpr('artist_id', 
                            'artist_name AS name', 
                            'artist_location AS location', 
                            'artist_latitude AS latitude', 
                            'artist_longitude AS longtitude')\
                .dropDuplicates(['artist_id'])
    
    # write artists table to parquet files
    artists_table.write.parquet(path = os.path.join(output_data, 'artist.parquet'), mode = 'overwrite')


def process_log_data(spark, input_data, output_data):
    '''
        pull log data from S3, process data to create users, time, songplays table 
        then load these tables to S3 as parquet format.   

        Args:
            spark: spark session with hadoop package
            input_data: the path of S3 buckect which stores log data
            output_data: the path of S3 buckect which stores processed data  
    '''
    
    # get filepath to song & log data file
    song_data = os.path.join(input_data, 'song_data', '*', '*', '*')
    log_data = os.path.join(input_data, 'log_data', '*', '*')

    # read song & log data file
    df_song = spark.read.json(song_data)
    df_log = spark.read.json(log_data)
    
    # define get_datetime UDF
    get_datetime = F.udf(lambda x: datetime.fromtimestamp(x/1000).isoformat())
    
    # filter by actions for song plays
    # create columns songplay_id as primary key
    # create colummn start_time as timestamp
    df_log = df_log\
                .where("page = 'NextSong'")\
                .withColumn('songplay_id', F.monotonically_increasing_id())\
                .withColumn('start_time', get_datetime('ts').cast('timestamp')) 

    # extract columns for users table    
    users_table = df_log\
                    .selectExpr('userId AS user_id',
                                'firstName AS first_name',
                                'lastName AS last_name',
                                'gender',
                                'level')\
                    .dropDuplicates(['user_id'])
    
    # write users table to parquet files
    users_table.write.parquet(os.path.join(output_data, 'user.parquet'), mode = 'overwrite')
    
    # extract columns to create time table
    time_table = df_log\
                .select('start_time')\
                .withColumn('hour', F.hour('start_time'))\
                .withColumn('day', F.dayofmonth('start_time'))\
                .withColumn('week', F.weekofyear('start_time'))\
                .withColumn('month', F.month('start_time'))\
                .withColumn('year', F.year('start_time'))\
                .withColumn('weekday', F.dayofweek('start_time'))\
                .dropDuplicates(['start_time']) 
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy('year', 'month').parquet(os.path.join(output_data, 'time.parquet'), mode='overwrite')

    # extract columns from joined song and log datasets to create songplays table 
    df_song.createOrReplaceTempView('song')
    df_log.createOrReplaceTempView('log')
    
    songplays_table = spark.sql('''
                                SELECT 
                                log.songplay_id,
                                log.start_time,
                                log.userId AS user_id,
                                log.level,
                                song.song_id,
                                song.artist_id,
                                log.sessionId AS session_id,
                                log.location,
                                log.userAgent AS user_agent,
                                MONTH(log.start_time) AS month,
                                YEAR(log.start_time) AS year
                                FROM log
                                LEFT JOIN song 
                                 ON song.title = log.song
                                 AND song.artist_name = log.artist
                            ''') 

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy('year', 'month').parquet(os.path.join(output_data, 'songPlay.parquet'), mode = 'overwrite')


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://song-log-benson/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
