
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
n_estimators = 10
max_depth = 5
############################## LOAD TRAINING SET ##############################

if os.path.exists("Data/test_PrecomputedMatrices/xTrain.npy") and os.path.exists("Data/test_PrecomputedMatrices/yTrain.npy"):
    xTrain = np.load("Data/test_PrecomputedMatrices/xTrain.npy")
    yTrain = np.load("Data/test_PrecomputedMatrices/yTrain.npy")
    print(f"xTrain shape = {xTrain.shape}")
    temp = np.column_stack([xTrain, yTrain])
    temp = temp[~np.isnan(temp).any(axis=1)]
    yTrain = temp[:, -1:]
    xTrain = temp[:, :-1]
    yTrain = np.ravel(yTrain)
    pow_bool = xTrain[:, -1:]
    xTrain = xTrain[:, :-1]
    xTrain = preprocessing.normalize(xTrain, norm='l2', axis=0)
    # xTrain = np.column_stack([xTrain, pow_bool])
    print(f"normalized xTrain: {xTrain} with shape {xTrain.shape}")
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
numTrials = 1

for i in range(numTrials):
    X_train, X_test, Y_train, Y_test = train_test_split(xTrain, yTrain)
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

def predictGame(team_1_vector, team_2_vector, home, modelUsed):
    diff = [a - b for a, b in zip(team_1_vector, team_2_vector)]
    # print(f"diff: {diff}")
    # diff.append(home)
    # print(f"diff after append: {diff}")
    # Depending on the model you use, you will either need to return model.predict_proba or model.predict
    # predict_proba = Linear Reg, Linear SVC
    # predict = Gradient Boosted, Ridge, HuberRegressor
    filler = diff
    # return modelUsed.predict_proba([diff])[0][1]
    return modelUsed.predict([diff])[0]


############################## CREATE KAGGLE SUBMISSION ##############################

def loadTeamVectors(years):
    listDictionaries = []
    for year in years:
        curVectors = np.load("Data/test_PrecomputedMatrices/TeamVectors/" + str(year) + "TeamVectors.npy",
                             encoding='latin1').item()
        # curVectors = curVectors[:, :-1]
        listDictionaries.append(curVectors)
    return listDictionaries


def createPrediction(stage2=False):
    if stage2:
        years = [2018]
        localPd = sample_sub_pd2
    else:
        # The years that we want to predict for
        years = range(2014, 2018)
        localPd = sample_sub_pd

    if os.path.exists("result.csv"):
        os.remove("result.csv")
    listDictionaries = loadTeamVectors(years)
    print("Loaded the team vectors")
    results = [[0 for x in range(2)] for x in range(len(localPd.index))]

    predictionModel = GradientBoostingRegressor(n_estimators=100, max_depth=5)
    predictionModel.fit(xTrain, yTrain)

    for index, row in localPd.iterrows():
        matchupId = row['ID']
        year = int(matchupId[0:4])
        teamVectors = listDictionaries[year - years[0]]
        team1Id = int(matchupId[5:9])
        team2Id = int(matchupId[10:14])
        team1Vector = teamVectors[team1Id]
        team2Vector = teamVectors[team2Id]
        pred1 = predictGame(team1Vector, team2Vector, 0, predictionModel)
        pred = pred1.clip(0., 1.)
        results[index][0] = matchupId
        results[index][1] = pred
    results = pd.np.array(results)
    firstRow = [[0 for x in range(2)] for x in range(1)]
    firstRow[0][0] = 'ID'
    firstRow[0][1] = 'Pred'
    with open("result.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(firstRow)
        writer.writerows(results)


# createPrediction()
# createPrediction(stage2=True)

############################## PREDICTING 2018 BRACKET ##############################

def trainModel(learning_rate, n_estimators, max_depth):
    model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
    model.fit(xTrain, yTrain)
    return model


def randomWinner(team1, team2, modelUsed):
    year = [2018]
    teamVectors = loadTeamVectors(year)[0]
    team1Vector = teamVectors[int(teams_pd[teams_pd['TeamName'] == team1].values[0][0])]
    team2Vector = teamVectors[int(teams_pd[teams_pd['TeamName'] == team2].values[0][0])]
    # Normalize
    team1Vector = preprocessing.normalize(team1Vector, norm='l2', axis=0)
    team2Vector = preprocessing.normalize(team2Vector, norm='l2', axis=0)


    prediction = predictGame(team1Vector, team2Vector, 0, modelUsed)
    for i in range(10):
        if (prediction > random.random()):
            print("{0} Wins".format(team1))
        else:
            print("{0} Wins".format(team2))


def findWinner(team1, team2, modelUsed):
    year = [2018]
    teamVectors = loadTeamVectors(year)[0]
    team1Vector = teamVectors[int(teams_pd[teams_pd['TeamName'] == team1].values[0][0])]
    team1Vector = team1Vector[:-1]
    team2Vector = teamVectors[int(teams_pd[teams_pd['TeamName'] == team2].values[0][0])]
    team2Vector = team2Vector[:-1]
    prediction = predictGame(team1Vector, team2Vector, 0, modelUsed)
    if (prediction < 0.5):
        print("Probability that {0} wins: {1}".format(team2, 1 - prediction))
    else:
        print("Probability that {0} wins: {1}".format(team1, prediction))


trainedModel = trainModel(learning_rate, n_estimators, max_depth)
sav_string = f"lr_{learning_rate}num_{n_estimators}md_{max_depth}"
pickle.dump(trainedModel, open('models/' + sav_string, 'wb'))


# First round games in the South for example
findWinner('Virginia', 'UMBC', trainedModel)
findWinner('Creighton', 'Kansas St', trainedModel)
findWinner('Kentucky', 'Davidson', trainedModel)
findWinner('Arizona', 'Buffalo', trainedModel)
findWinner('Miami FL', 'Loyola-Chicago', trainedModel)
findWinner('Tennessee', 'Wright St', trainedModel)
findWinner('Nevada', 'Texas', trainedModel)
findWinner('Cincinnati', 'Georgia St', trainedModel)
findWinner('Xavier', 'Gonzaga', trainedModel)
findWinner('Duke', 'Michigan St', trainedModel)

x = 10
