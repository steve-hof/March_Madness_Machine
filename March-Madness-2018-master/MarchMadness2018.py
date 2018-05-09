############################## IMPORTS ##############################

from __future__ import division
import sklearn
import pandas as pd
import numpy as np
import collections
import os.path
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import tree
from sklearn.model_selection import cross_val_score
from keras.utils import np_utils
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sys
from sklearn.ensemble import GradientBoostingRegressor
import math
import csv
import pickle
from sklearn import preprocessing
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
import urllib
from sklearn.svm import LinearSVC
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import random

learning_rate = 0.1
n_estimators = 72
max_depth = 5

############################## LOAD TRAINING SET ##############################

if os.path.exists("Data/test_PrecomputedMatrices/x_train.npy") and os.path.exists(
        "Data/test_PrecomputedMatrices/y_train.npy"):
    x_train = np.load("Data/test_PrecomputedMatrices/x_train.npy")
    y_train = np.load("Data/test_PrecomputedMatrices/y_train.npy")
    print(f"x_train shape = {x_train.shape}")
    temp = np.column_stack([x_train, y_train])
    temp = temp[~np.isnan(temp).any(axis=1)]
    y_train = temp[:, -1:]
    x_train = temp[:, :-1]
    y_train = np.ravel(y_train)
    pow_bool = x_train[:, -1:]
    x_train = x_train[:, :-1]
    # x_train = preprocessing.normalize(x_train, norm='l2', axis=0)
    x_train = np.column_stack([x_train, pow_bool])
    print(f"normalized x_train: {x_train} with shape {x_train.shape}")
else:
    print('We need a training set! Run DataPreprocessing.py')
    sys.exit()

############################## LOAD CSV FILES ##############################

sample_sub_pd = pd.read_csv('Data/KaggleData/SampleSubmissionStage1.csv')
sample_sub_pd2 = pd.read_csv('Data/KaggleData/SampleSubmissionStage2.csv')
teams_pd = pd.read_csv('Data/KaggleData/Teams.csv')

############################## TRAIN MODEL ##############################
# filename = "models/lr_0.1num_300md_5"
model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
# model = pickle.load(open(filename, 'rb'))
# categories = ['Wins', 'PPG', 'PPGA', 'PowerConf', '3PG', 'APG', 'TOP', 'Conference Champ', 'Tourney Conference Champ',
#               'Seed', 'SOS', 'SRS', 'RPG', 'SPG', 'Tourney Appearances', 'National Championships', 'Location']

categories = ['OffRtg', 'DefRtg', 'NetRtg', 'AstR', 'TOR', 'TSP', 'eFGP',
              'FTAR', 'ORP', 'DRP', 'RP', 'PIE', 'Ending_Elo']
accuracy = []
numTrials = 0

for i in range(numTrials):
    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train)
    print(f"Xshape = {X_train.shape}, YShape = {Y_train.shape}")
    startTime = datetime.now()  # For some timing stuff
    results = model.fit(X_train, Y_train)
    preds = model.predict(X_test)

    preds[preds < .5] = 0
    preds[preds >= .5] = 1
    localAccuracy = np.mean(preds == Y_test)
    accuracy.append(localAccuracy)
    print("Finished run #" + str(i) + ". Accuracy = " + str(localAccuracy))
    print("Time taken: " + str(datetime.now() - startTime))
if numTrials != 0:
    print("The average accuracy is", sum(accuracy) / len(accuracy))


############################## TEST MODEL ##############################

def predictGame(team_1_vector, team_2_vector, home, model_used):
    diff = [a - b for a, b in zip(team_1_vector, team_2_vector)]
    # print(f"diff: {diff}")
    diff.append(home)
    # print(f"diff after append: {diff}")
    # Depending on the model you use, you will either need to return model.predict_proba or model.predict
    # predict_proba = Linear Reg, Linear SVC
    # predict = Gradient Boosted, Ridge, HuberRegressor
    # return model_used.predict_proba([diff])[0][1]
    return model_used.predict([diff])[0]


############################## CREATE KAGGLE SUBMISSION ##############################

def load_team_vectors(years):
    list_dictionaries = []
    for year in years:
        curVectors = np.load("Data/test_PrecomputedMatrices/TeamVectors/" + str(year) + "TeamVectors.npy",
                             encoding='latin1').item()
        # curVectors = curVectors[:, :-1]
        list_dictionaries.append(curVectors)
    return list_dictionaries


def create_prediction(stage2=False):
    if stage2:
        years = [2018]
        localPd = sample_sub_pd2
    else:
        # The years that we want to predict for
        years = range(2014, 2018)
        localPd = sample_sub_pd

    if os.path.exists("result.csv"):
        os.remove("result.csv")
    list_dictionaries = load_team_vectors(years)
    print("Loaded the team vectors")
    results = [[0 for x in range(2)] for x in range(len(localPd.index))]

    prediction_model = GradientBoostingRegressor(n_estimators=100, max_depth=5)
    prediction_model.fit(x_train, y_train)

    for index, row in localPd.iterrows():
        match_id = row['ID']
        year = int(match_id[0:4])
        team_vectors = list_dictionaries[year - years[0]]
        team_1_id = int(match_id[5:9])
        team_2_id = int(match_id[10:14])
        team_1_vector = team_vectors[team_1_id]
        team_2_vector = team_vectors[team_2_id]
        pred1 = predictGame(team_1_vector, team_2_vector, 0, prediction_model)
        pred = pred1.clip(0., 1.)
        results[index][0] = match_id
        results[index][1] = pred
    results = pd.np.array(results)
    firstRow = [[0 for x in range(2)] for x in range(1)]
    firstRow[0][0] = 'ID'
    firstRow[0][1] = 'Pred'
    with open("result.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(firstRow)
        writer.writerows(results)


# create_prediction()
# create_prediction(stage2=True)

############################## PREDICTING 2018 BRACKET ##############################

def train_model(learning_rate, n_estimators, max_depth):
    model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
    model.fit(x_train, y_train)
    return model


def random_winner(team1, team2, model_used):
    year = [2018]
    team_vectors = load_team_vectors(year)[0]
    team_1_vector = team_vectors[int(teams_pd[teams_pd['TeamName'] == team1].values[0][0])]
    team_2_vector = team_vectors[int(teams_pd[teams_pd['TeamName'] == team2].values[0][0])]
    # Normalize
    team_1_vector = preprocessing.normalize(team_1_vector, norm='l2', axis=0)
    team_2_vector = preprocessing.normalize(team_2_vector, norm='l2', axis=0)

    prediction = predictGame(team_1_vector, team_2_vector, 0, model_used)
    for i in range(10):
        if prediction > random.random():
            print(f"{team1} Wins")
        else:
            print(f"{team2} Wins")


def find_winner(team1, team2, model_used):
    year = [2018]
    team_vectors = load_team_vectors(year)[0]
    team_1_vector = team_vectors[int(teams_pd[teams_pd['TeamName'] == team1].values[0][0])]
    team_1_vector = team_1_vector[:-1]
    team_2_vector = team_vectors[int(teams_pd[teams_pd['TeamName'] == team2].values[0][0])]
    team_2_vector = team_2_vector[:-1]
    prediction = predictGame(team_1_vector, team_2_vector, 0, model_used)
    if prediction < 0.5:
        print(f"Probability that {team2} wins: {1-prediction}")
    else:
        print(f"Probability that {team1} wins: {prediction}")
    with open('Predictions/predictions.txt', 'a') as f:
        if prediction < 0.5:
            f.write(f"I am {1-prediction} sure that {team2} will beat {team1}\n")
        else:
            f.write(f"I am {prediction} sure that {team1} will beat {team2}\n")


trainedModel = train_model(learning_rate, n_estimators, max_depth)
sav_string = f"lr_{learning_rate}num_{n_estimators}md_{max_depth}"
pickle.dump(trainedModel, open('models/' + sav_string, 'wb'))

# As an example, below prints out the probability that Michigan
# will beat Villanova (or vice versa)
find_winner('Michigan', 'Villanova', trainedModel)
