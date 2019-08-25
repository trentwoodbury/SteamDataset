# This took 2 hours to create
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


def dummify_genres(all_games):
    '''
    This function
        (1) Converts the game genre into dummy variables
        (2) aggregates rows on a customer/game basis (e.g. when there were multiple genres for a game)
        (3) Filters out rows with 0 playtime
    '''
    dummy_genres = pd.get_dummies(all_games['Genre'])
    all_games = pd.concat((all_games, dummy_genres), axis=1)
    all_games = all_games.drop(['Genre'], axis=1).reset_index().drop(['index'], axis=1)

    aggregations = {val: 'max' for val in all_games.columns[2:]}
    all_games = all_games.groupby(['steamid', 'appid']).agg(aggregations).reset_index()
    all_games = all_games.loc[all_games.playtime_forever > 0, :]

    return all_games


def get_holdout_games(dummied_games):
    '''
    Adds a label and the associated appid to each customer's row.
    '''
    current_steamid = -1
    customer_prediction_apps = {}
    customer_labels = {}
    # Get the last row since the dataframe is sorted by customer and then date.
    for idx in range(dummied_games.shape[0])[::-1]:
        row = dummied_games.iloc[idx, :]
        if row.steamid != current_steamid:
            customer_prediction_apps[row.steamid] = row.appid
            customer_labels[row.steamid] = row.playtime_forever
            current_steamid = row.steamid

    labels = []
    appids = []

    for idx, row in dummied_games.iterrows():
        labels.append(customer_labels[row.steamid])
        appids.append(customer_prediction_apps[row.steamid])

    dummied_games['labels'] = labels
    dummied_games['label_appid'] = appids

    return dummied_games


def add_cutomer_level_aggregate_statistics(labeled_games):
    '''
    Get Customer's Mean and Median playtime overall and add to the
    labeled_games dataframe.
    '''
    # now let's create a dataframe for aggregating all the rows representing other games the player played
    other_games = labeled_games.loc[labeled_games.label_appid != labeled_games.appid]
    other_games = other_games.loc[:, ['steamid', 'playtime_forever']]
    other_games_mean = (
        other_games
        .groupby('steamid')
        .mean()
    )
    other_games_mean.columns = ['MeanPlaytime']
    other_games_median = (
        other_games
        .groupby('steamid')
        .median()
    )
    other_games_median.columns = ['MedianPlaytime']

    other_games_agg = other_games_mean.join(other_games_median, how='inner')
    other_games_agg.reset_index()
    labeled_games = labeled_games.merge(other_games_agg, on='steamid', how='left')
    labeled_games = labeled_games.loc[labeled_games.appid == labeled_games.label_appid]

    return labeled_games


def create_train_test_split(labeled_games):
    '''
    This function performs a temporal train-test split.
    '''
    cutoff_date = datetime.datetime.strptime('2014-09-22', '%Y-%m-%d')

    games_df = labeled_games.copy()
    games_df['dateretrieved'] = (
        games_df['dateretrieved']
        .apply(lambda x: datetime.datetime.strptime(x[:10], '%Y-%m-%d'))
    )
    training_data = games_df.loc[games_df.dateretrieved < cutoff_date]
    test_data = games_df.loc[games_df.dateretrieved > cutoff_date]
    filter_columns = [
        'steamid',
        'appid',
        'dateretrieved',
        'labels',
        'label_appid',
        'Title',
        'Type',
        'Price',
        'Release_Date',
        'Developer',
        'playtime_2weeks',
        'Genre',
        'playtime_forever'
    ]
    X_train = training_data.loc[:, [col for col in training_data.columns if col not in filter_columns]]
    X_test = test_data.loc[:, [col for col in training_data.columns if col not in filter_columns]]
    y_train = training_data.loc[:, 'labels']
    y_test = test_data.loc[:, 'labels']

    return X_train, X_test, y_train, y_test


def save_results_to_csvs(X_train, X_test, y_train, y_test):
    X_train.to_csv('../data/X_train.csv', index=False, header=True)
    X_test.to_csv('../data/X_test.csv', index=False, header=True)
    y_train.to_csv('../data/y_train.csv', index=False, header=True)
    y_test.to_csv('../data/y_test.csv', index=False, header=True)


if __name__ == "__main__":

    all_games = read_in_games()
    print("Read in and joined games dataframes.")

    dummied_games = dummify_genres(all_games)
    print("Transformed game genres into dummy variables.")

    labeled_games = get_holdout_games(dummied_games)
    print("Collected holdout games.")

    aggregated_games = add_cutomer_level_aggregate_statistics(labeled_games)
    print("Got aggregate statistics for each customer.")

    X_train, X_test, y_train, y_test = create_train_test_split(aggregated_games)
    print("Performed Train-Test split.")

    save_results_to_csvs(X_train, X_test, y_train, y_test)
    print("Saved results to csv files.")
