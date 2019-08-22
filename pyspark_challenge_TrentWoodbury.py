# Setting up Pyspark locally took ~30 minutes
# The code took me 1 hour and 20 minutes to complete

from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as f

spark = SparkSession.builder.config(conf=SparkConf()).getOrCreate()

player_summaries = spark.read.csv('data/Player_Summaries.csv',header=True)
game_publishers = spark.read.csv('data/Games_Publishers.csv', header=True)
game_genres = spark.read.csv('data/Games_Genres.csv', header=True)
game_developers = spark.read.csv('data/Games_Developers.csv', header=True)
games_1 = spark.read.csv('data/Games_1.csv', header=True)

# Join all the games together
joined_games = (
    game_publishers
    .join(game_genres, game_publishers.appid == game_genres.appid)
    .join(game_developers, game_publishers.appid == game_developers.appid)
    .join(games_1, game_publishers.appid == games_1.appid)
)

# Count the number of games per publisher
games_per_publisher = (
    game_publishers
    .groupBy("Publisher")
    .agg(f.countDistinct("appid"))
    .select(
        f.col("Publisher"),
        f.col("count(DISTINCT appid)").alias("NumberOfGames")
    )
)

# Count the number of games per genre
games_per_genre = (
    game_genres
    .groupBy("Genre")
    .agg(f.countDistinct("appid"))
    .select(
        f.col("Genre"),
        f.col("count(DISTINCT appid)").alias("NumberOfGames")
    )
)

# Get the hour where the most accounts were created
creation_times = player_summaries.select(
        f.col("steamid"),
        f.date_trunc('hour', f.to_timestamp(player_summaries.timecreated)).alias("hourcreated")
)
popular_times = (
    creation_times
    .filter(creation_times.hourcreated.isNotNull())
    .select(
        f.col("steamid"),
        f.from_unixtime(f.unix_timestamp(f.col("hourcreated")), "yyyy-MM-dd HH:mm:ss").alias("hourcreated")
    )
    .groupBy("hourcreated").count()
    .orderBy(f.desc("count"))
)
# popular_times.head() gives us Row(hourcreated='2012-12-25 10:00:00', count=292)
# The most accounts were created on 10AM December 25th 2012 (small dataset)
