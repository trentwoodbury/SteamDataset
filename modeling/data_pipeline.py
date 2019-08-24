# This took 45 minutes to create
# This script runs a sequence of functions to create the final training, validation, and test
# sets that we will use for modeling
import datetime
import numpy as np
import pandas as pd

def read_in_games():
    '''
    Creates dataframe for all of the tables we're going to use
    and joins those tables together.
    '''
    games_1 = pd.read_csv('../data/Games_1.csv')
    games_2 = pd.read_csv('../data/Games_2.csv')
    app_id_info = pd.read_csv('../data/App_ID_Info.csv')
    game_developers = pd.read_csv('../data/Games_Developers.csv')
    game_genres = pd.read_csv('../data/Games_Genres.csv')

    all_games = pd.concat((games_1, games_2), axis=0)
    all_games = (
        all_games
        .merge(app_id_info, on='appid', how='left')
        .merge(game_developers, on='appid', how='left')
        .merge(game_genres, on='appid', how='left')
    )

    return  all_games

def create_train_test_split(games_df):
    '''
    This function performs a temporal train-test split.
    '''
    cutoff_date = datetime.datetime.strptime('2014-09-01', '%Y-%m-%d')

    games_df['dateretrieved'] = (
        games_df['dateretrieved']
        .apply(lambda x: datetime.datetime.strptime(x[:10], '%Y-%m-%d'))
    )
    training_data = games_df.loc[games_df.dateretrieved < cutoff_date]
    test_data = games_df.loc[games_df.dateretrieved > cutoff_date]
    return training_data, test_data


def remove_test_data_from_training(training_data, test_data):
    '''
    This function looks for all the pairs of (steamid, appid)
    and removes any rows with a given pair if that pair is in the test set.
    This also sets steamid and appid as the index of each dataframe.
    '''
    test_data.set_index(['steamid', 'appid'])
    training_data.set_index(['steamid', 'appid'])

    training_data = training_data.loc[~training_data.index.isin(test_data.index)]

    return training_data, test_data

if __name__ == "__main__":
    all_games = read_in_games()
    training_data, test_data = create_train_test_split(all_games)
    training_data, test_data = remove_test_data_from_training(training_data, test_data)
