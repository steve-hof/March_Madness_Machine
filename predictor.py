#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler as MM
import matplotlib.pyplot as plt

def normalize_dataframe(df, starting_col):
    vals = df.values[:, starting_col:]
    cols = df.columns[starting_col:].tolist()
    normed_vals = preprocessing.normalize(vals, norm='l2')
    df[cols] = normed_vals
    return df


def scale_dataframe(df, starting_col):
    vals = df.values[:, starting_col:]
    cols = df.columns[starting_col:].tolist()
    scaler = MM()
    scaled_vals = scaler.fit_transform(vals)
    # scaled_vals = preprocessing.scale(vals)
    df[cols] = scaled_vals
    return df


def get_ranking(df, season):
    data_cols = df.columns[4:]
    df['Sum'] = df[data_cols].sum(axis=1)
    df.sort_values(by=['Sum'], inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    return df


def get_info(team_w, team_l, season, df):
    data_cols = df.columns[3:]
    tm_w_info = df[(df['Season'] == season) & (df['TeamID'] == team_w)][data_cols].values
    tm_l_info = df[(df['Season'] == season) & (df['TeamID'] == team_l)][data_cols].values
    return tm_w_info, tm_l_info


def main():
    season = 2015
    reg_seas = pd.read_csv('input_data.csv', index_col=0)
    tourney_compact_df = pd.read_csv('training_data/DataFiles/NCAATourneyCompactResults.csv')
    normalized_df = normalize_dataframe(reg_seas, starting_col=4)
    scaled_df = scale_dataframe(reg_seas, starting_col=4)

    ###############
    # PREDICTIONS #
    ###############
    scaled_ranking = get_ranking(scaled_df, season)
    normed_ranking = get_ranking(normalized_df, season)


    end = season

if __name__ == '__main__':
    main()