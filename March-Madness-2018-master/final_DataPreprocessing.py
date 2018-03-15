# Format:
# 1) Imports
# 2) Load CSVs
# 3) Data Structures
# 4) Data Preprocessing Helper Functions
# 5) Create Training Set

############################## IMPORTS ##############################

from __future__ import division
import pandas as pd
import numpy as np
import os.path
import math
import collections

############################## LOAD CSVs ##############################

reg_season_compact_pd = pd.read_csv('Data/KaggleData/RegularSeasonCompactResults.csv')
teams_pd = pd.read_csv('Data/KaggleData/Teams.csv')
tourney_compact_pd = pd.read_csv('Data/KaggleData/NCAATourneyCompactResults.csv')
conference_pd = pd.read_csv('Data/KaggleData/Conference.csv')
tourney_results_pd = pd.read_csv('Data/KaggleData/TourneyResults.csv')
tourney_seeds_pd = pd.read_csv('Data/KaggleData/NCAATourneySeeds.csv')
team_conferences_pd = pd.read_csv('Data/KaggleData/TeamConferences.csv')
reg_season_detail_pd = pd.read_csv('../training_data/DataFiles/RegularSeasonDetailedResults.csv')

############################## DATA STRUCTURES ##############################

teamList = teams_pd['TeamName'].tolist()
NCAAChampionsList = tourney_results_pd['NCAA Champion'].tolist()


############################## HELPER FUNCTIONS ##############################

def checkPower6Conference(team_id):
    team_pd = team_conferences_pd[(team_conferences_pd['Season'] == 2018) & (team_conferences_pd['TeamID'] == team_id)]
    # Can't find the team
    if (len(team_pd) == 0):
        return 0
    confName = team_pd.iloc[0]['ConfAbbrev']
    return int(
        confName == 'sec' or confName == 'acc' or confName == 'big_ten' or confName == 'big_twelve' or confName == 'big_east' or confName == 'pac_twelve')


def getTeamID(name):
    return teams_pd[teams_pd['TeamName'] == name].values[0][0]


def getTeamName(team_id):
    return teams_pd[teams_pd['TeamID'] == team_id].values[0][1]


def getNumChampionships(team_id):
    name = getTeamName(team_id)
    return NCAAChampionsList.count(name)


def getListForURL(team_list):
    team_list = [x.lower() for x in team_list]
    team_list = [t.replace(' ', '-') for t in team_list]
    team_list = [t.replace('st', 'state') for t in team_list]
    team_list = [t.replace('northern-dakota', 'north-dakota') for t in team_list]
    team_list = [t.replace('nc-', 'north-carolina-') for t in team_list]
    team_list = [t.replace('fl-', 'florida-') for t in team_list]
    team_list = [t.replace('ga-', 'georgia-') for t in team_list]
    team_list = [t.replace('lsu', 'louisiana-state') for t in team_list]
    team_list = [t.replace('maristate', 'marist') for t in team_list]
    team_list = [t.replace('stateate', 'state') for t in team_list]
    team_list = [t.replace('northernorthern', 'northern') for t in team_list]
    team_list = [t.replace('usc', 'southern-california') for t in team_list]
    base = 'http://www.sports-reference.com/cbb/schools/'
    for team in team_list:
        url = base + team + '/'


getListForURL(teamList);


def handleCases(arr):
    indices = []
    listLen = len(arr)
    for i in range(listLen):
        if (arr[i] == 'St' or arr[i] == 'FL'):
            indices.append(i)
    for p in indices:
        arr[p - 1] = arr[p - 1] + ' ' + arr[p]
    for i in range(len(indices)):
        arr.remove(arr[indices[i] - i])
    return arr


def checkConferenceChamp(team_id, year):
    year_conf_pd = conference_pd[conference_pd['Year'] == year]
    champs = year_conf_pd['Regular Season Champ'].tolist()
    # For handling cases where there is more than one champion
    champs_separated = [words for segments in champs for words in segments.split()]
    name = getTeamName(team_id)
    champs_separated = handleCases(champs_separated)
    if (name in champs_separated):
        return 1
    else:
        return 0


def checkConferenceTourneyChamp(team_id, year):
    year_conf_pd = conference_pd[conference_pd['Year'] == year]
    champs = year_conf_pd['Tournament Champ'].tolist()
    name = getTeamName(team_id)
    if (name in champs):
        return 1
    else:
        return 0


def getTourneyAppearances(team_id):
    return len(tourney_seeds_pd[tourney_seeds_pd['TeamID'] == team_id].index)


def handleDifferentCSV(df):
    # The stats CSV is a lit different in terms of naming so below is just some data cleaning
    df['School'] = df['School'].replace('(State)', 'St', regex=True)
    df['School'] = df['School'].replace('Albany (NY)', 'Albany NY')
    df['School'] = df['School'].replace('Boston University', 'Boston Univ')
    df['School'] = df['School'].replace('Central Michigan', 'C Michigan')
    df['School'] = df['School'].replace('(Eastern)', 'E', regex=True)
    df['School'] = df['School'].replace('Louisiana St', 'LSU')
    df['School'] = df['School'].replace('North Carolina St', 'NC State')
    df['School'] = df['School'].replace('Southern California', 'USC')
    df['School'] = df['School'].replace('University of California', 'California', regex=True)
    df['School'] = df['School'].replace('American', 'American Univ')
    df['School'] = df['School'].replace('Arkansas-Little Rock', 'Ark Little Rock')
    df['School'] = df['School'].replace('Arkansas-Pine Bluff', 'Ark Pine Bluff')
    df['School'] = df['School'].replace('Bowling Green St', 'Bowling Green')
    df['School'] = df['School'].replace('Brigham Young', 'BYU')
    df['School'] = df['School'].replace('Cal Poly', 'Cal Poly SLO')
    df['School'] = df['School'].replace('Centenary (LA)', 'Centenary')
    df['School'] = df['School'].replace('Central Connecticut St', 'Central Conn')
    df['School'] = df['School'].replace('Charleston Southern', 'Charleston So')
    df['School'] = df['School'].replace('Coastal Carolina', 'Coastal Car')
    df['School'] = df['School'].replace('College of Charleston', 'Col Charleston')
    df['School'] = df['School'].replace('Cal St Fullerton', 'CS Fullerton')
    df['School'] = df['School'].replace('Cal St Sacramento', 'CS Sacramento')
    df['School'] = df['School'].replace('Cal St Bakersfield', 'CS Bakersfield')
    df['School'] = df['School'].replace('Cal St Northridge', 'CS Northridge')
    df['School'] = df['School'].replace('East Tennessee St', 'ETSU')
    df['School'] = df['School'].replace('Detroit Mercy', 'Detroit')
    df['School'] = df['School'].replace('Fairleigh Dickinson', 'F Dickinson')
    df['School'] = df['School'].replace('Florida Atlantic', 'FL Atlantic')
    df['School'] = df['School'].replace('Florida Gulf Coast', 'FL Gulf Coast')
    df['School'] = df['School'].replace('Florida International', 'Florida Intl')
    df['School'] = df['School'].replace('George Washington', 'G Washington')
    df['School'] = df['School'].replace('Georgia Southern', 'Ga Southern')
    df['School'] = df['School'].replace('Gardner-Webb', 'Gardner Webb')
    df['School'] = df['School'].replace('Illinois-Chicago', 'IL Chicago')
    df['School'] = df['School'].replace('Kent St', 'Kent')
    df['School'] = df['School'].replace('Long Island University', 'Long Island')
    df['School'] = df['School'].replace('Loyola Marymount', 'Loy Marymount')
    df['School'] = df['School'].replace('Loyola (MD)', 'Loyola MD')
    df['School'] = df['School'].replace('Loyola (IL)', 'Loyola-Chicago')
    df['School'] = df['School'].replace('Massachusetts', 'MA Lowell')
    df['School'] = df['School'].replace('Maryland-Eastern Shore', 'MD E Shore')
    df['School'] = df['School'].replace('Miami (FL)', 'Miami FL')
    df['School'] = df['School'].replace('Miami (OH)', 'Miami OH')
    df['School'] = df['School'].replace('Missouri-Kansas City', 'Missouri KC')
    df['School'] = df['School'].replace('Monmouth', 'Monmouth NJ')
    df['School'] = df['School'].replace('Mississippi Valley St', 'MS Valley St')
    df['School'] = df['School'].replace('Montana St', 'MTSU')
    df['School'] = df['School'].replace('Northern Colorado', 'N Colorado')
    df['School'] = df['School'].replace('North Dakota St', 'N Dakota St')
    df['School'] = df['School'].replace('Northern Illinois', 'N Illinois')
    df['School'] = df['School'].replace('Northern Kentucky', 'N Kentucky')
    df['School'] = df['School'].replace('North Carolina A&T', 'NC A&T')
    df['School'] = df['School'].replace('North Carolina Central', 'NC Central')
    df['School'] = df['School'].replace('Pennsylvania', 'Penn')
    df['School'] = df['School'].replace('South Carolina St', 'S Carolina St')
    df['School'] = df['School'].replace('Southern Illinois', 'S Illinois')
    df['School'] = df['School'].replace('UC-Santa Barbara', 'Santa Barbara')
    df['School'] = df['School'].replace('Southeastern Louisiana', 'SE Louisiana')
    df['School'] = df['School'].replace('Southeast Missouri St', 'SE Missouri St')
    df['School'] = df['School'].replace('Stephen F. Austin', 'SF Austin')
    df['School'] = df['School'].replace('Southern Methodist', 'SMU')
    df['School'] = df['School'].replace('Southern Mississippi', 'Southern Miss')
    df['School'] = df['School'].replace('Southern', 'Southern Univ')
    df['School'] = df['School'].replace('St. Bonaventure', 'St Bonaventure')
    df['School'] = df['School'].replace('St. Francis (NY)', 'St Francis NY')
    df['School'] = df['School'].replace('Saint Francis (PA)', 'St Francis PA')
    df['School'] = df['School'].replace('St. John\'s (NY)', 'St John\'s')
    df['School'] = df['School'].replace('Saint Joseph\'s', 'St Joseph\'s PA')
    df['School'] = df['School'].replace('Saint Louis', 'St Louis')
    df['School'] = df['School'].replace('Saint Mary\'s (CA)', 'St Mary\'s CA')
    df['School'] = df['School'].replace('Mount Saint Mary\'s', 'Mt St Mary\'s')
    df['School'] = df['School'].replace('Saint Peter\'s', 'St Peter\'s')
    df['School'] = df['School'].replace('Texas A&M-Corpus Christian', 'TAM C. Christian')
    df['School'] = df['School'].replace('Texas Christian', 'TCU')
    df['School'] = df['School'].replace('Tennessee-Martin', 'TN Martin')
    df['School'] = df['School'].replace('Texas-Rio Grande Valley', 'UTRGV')
    df['School'] = df['School'].replace('Texas Southern', 'TX Southern')
    df['School'] = df['School'].replace('Alabama-Birmingham', 'UAB')
    df['School'] = df['School'].replace('UC-Davis', 'UC Davis')
    df['School'] = df['School'].replace('UC-Irvine', 'UC Irvine')
    df['School'] = df['School'].replace('UC-Riverside', 'UC Riverside')
    df['School'] = df['School'].replace('Central Florida', 'UCF')
    df['School'] = df['School'].replace('Louisiana-Lafayette', 'ULL')
    df['School'] = df['School'].replace('Louisiana-Monroe', 'ULM')
    df['School'] = df['School'].replace('Maryland-Baltimore County', 'UMBC')
    df['School'] = df['School'].replace('North Carolina-Asheville', 'UNC Asheville')
    df['School'] = df['School'].replace('North Carolina-Greensboro', 'UNC Greensboro')
    df['School'] = df['School'].replace('North Carolina-Wilmington', 'UNC Wilmington')
    df['School'] = df['School'].replace('Nevada-Las Vegas', 'UNLV')
    df['School'] = df['School'].replace('Texas-Arlington', 'UT Arlington')
    df['School'] = df['School'].replace('Texas-San Antonio', 'UT San Antonio')
    df['School'] = df['School'].replace('Texas-El Paso', 'UTEP')
    df['School'] = df['School'].replace('Virginia Commonwealth', 'VA Commonwealth')
    df['School'] = df['School'].replace('Western Carolina', 'W Carolina')
    df['School'] = df['School'].replace('Western Illinois', 'W Illinois')
    df['School'] = df['School'].replace('Western Kentucky', 'WKU')
    df['School'] = df['School'].replace('Western Michigan', 'W Michigan')
    df['School'] = df['School'].replace('Abilene Christian', 'Abilene Chr')
    df['School'] = df['School'].replace('Montana State', 'Montana St')
    df['School'] = df['School'].replace('Central Arkansas', 'Cent Arkansas')
    df['School'] = df['School'].replace('Houston Baptist', 'Houston Bap')
    df['School'] = df['School'].replace('South Dakota St', 'S Dakota St')
    df['School'] = df['School'].replace('Maryland-Eastern Shore', 'MD E Shore')
    return df


def getHomeStat(row):
    if (row == 'H'):
        home = 1
    if (row == 'A'):
        home = -1
    if (row == 'N'):
        home = 0
    return home


def compareTwoTeams(id_1, id_2, year):
    team_1 = getSeasonData(id_1, year)
    team_2 = getSeasonData(id_2, year)
    diff = [a - b for a, b in zip(team_1, team_2)]
    return diff


def normalizeInput(arr):
    for i in range(arr.shape[1]):
        minVal = min(arr[:, i])
        maxVal = max(arr[:, i])
        arr[:, i] = (arr[:, i] - minVal) / (maxVal - minVal)
    return arr


def normalizeInput2(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def create_adv_stats(df, tm_id, year):
    gamesWon = df[df.WTeamID == tm_id]
    totalPointsScored = gamesWon['WScore'].sum()
    gamesLost = df[df.LTeamID == tm_id]
    totalGames = gamesWon.append(gamesLost)
    numGames = len(totalGames.index)

    if numGames == 0:
        vals = np.zeros(13).tolist()
        return vals
    else:
        ##########################
        # CALCULATING POSSESSION #
        ##########################
        df = df[(df['WTeamID'] == tm_id) | (df['LTeamID'] == tm_id)]
        df = df[df['Season'] == year]
        fill = 2
        # Points Winning/Losing Team
        df['WPts'] = df.apply(lambda row: 2*row.WFGM + row.WFGM3 + row.WFTM, axis=1)
        df['LPts'] = df.apply(lambda row: 2*row.LFGM + row.LFGM3 + row.LFTM, axis=1)

        # Calculate Winning/losing Team Possession Feature
        w_pos = df.apply(lambda row: 0.96*(row.WFGA + row.WTO + 0.44*row.WFTA - row.WOR), axis=1)
        l_pos = df.apply(lambda row: 0.96*(row.LFGA + row.LTO + 0.44*row.LFTA - row.LOR), axis=1)

        # two teams use almost the same number of possessions in a game
        # (plus/minus one or two - depending on how quarters end)
        # so let's just take the average
        df['Pos'] = (w_pos+l_pos)/2

        ####################
        # ADVANCED METRICS #
        ####################

        # Offensive efficiency (OffRtg) = 100 x (Points / Possessions)
        df['WOffRtg'] = df.apply(lambda row: 100 * (row.WPts / row.Pos), axis=1)
        df['LOffRtg'] = df.apply(lambda row: 100 * (row.LPts / row.Pos), axis=1)

        # Defensive efficiency (DefRtg) = 100 x (Opponent points / Opponent possessions)
        df['WDefRtg'] = df.LOffRtg
        df['LDefRtg'] = df.WOffRtg

        # Net Rating = Off.Rtg - Def.Rtg
        df['WNetRtg'] = df.apply(lambda row:(row.WOffRtg - row.WDefRtg), axis=1)
        df['LNetRtg'] = df.apply(lambda row:(row.LOffRtg - row.LDefRtg), axis=1)

        # Assist Ratio : Percentage of team possessions that end in assists
        df['WAstR'] = df.apply(lambda row: 100 * row.WAst / (row.WFGA + 0.44*row.WFTA + row.WAst + row.WTO), axis=1)
        df['LAstR'] = df.apply(lambda row: 100 * row.LAst / (row.LFGA + 0.44*row.LFTA + row.LAst + row.LTO), axis=1)

        # Turnover Ratio: Number of turnovers of a team per 100 possessions used.
        # (TO * 100) / (FGA + (FTA * 0.44) + AST + TO)
        df['WTOR'] = df.apply(lambda row: 100 * row.WTO / (row.WFGA + 0.44*row.WFTA + row.WAst + row.WTO), axis=1)
        df['LTOR'] = df.apply(lambda row: 100 * row.LTO / (row.LFGA + 0.44*row.LFTA + row.LAst + row.LTO), axis=1)

        # True Shooting Percentage : Measure of Shooting Efficiency (FGA/FGA3, FTA)
        df['WTSP'] = df.apply(lambda row: 100 * row.WPts / (2 * (row.WFGA + 0.44 * row.WFTA)), axis=1)
        df['LTSP'] = df.apply(lambda row: 100 * row.LPts / (2 * (row.LFGA + 0.44 * row.LFTA)), axis=1)

        # eFG% : Effective Field Goal Percentage adjusting for the fact that 3pt shots are more valuable
        df['WeFGP'] = df.apply(lambda row: (row.WFGM + 0.5 * row.WFGM3) / row.WFGA, axis=1)
        df['LeFGP'] = df.apply(lambda row: (row.LFGM + 0.5 * row.LFGM3) / row.LFGA, axis=1)

        # FTA Rate : How good a team is at drawing fouls.
        df['WFTAR'] = df.apply(lambda row: row.WFTA / row.WFGA, axis=1)
        df['LFTAR'] = df.apply(lambda row: row.LFTA / row.LFGA, axis=1)

        # OREB% : Percentage of team offensive rebounds
        df['WORP'] = df.apply(lambda row: row.WOR / (row.WOR + row.LDR), axis=1)
        df['LORP'] = df.apply(lambda row: row.LOR / (row.LOR + row.WDR), axis=1)

        # DREB% : Percentage of team defensive rebounds
        df['WDRP'] = df.apply(lambda row: row.WDR / (row.WDR + row.LOR), axis=1)
        df['LDRP'] = df.apply(lambda row: row.LDR / (row.LDR + row.WOR), axis=1)

        # REB% : Percentage of team total rebounds
        df['WRP'] = df.apply(lambda row: (row.WDR + row.WOR) / (row.WDR + row.WOR + row.LDR + row.LOR), axis=1)
        df['LRP'] = df.apply(lambda row: (row.LDR + row.LOR) / (row.WDR + row.WOR + row.LDR + row.LOR), axis=1)

        # PIE% : Player Impact Estimate (but calculated for team)
        wtmp = df.apply(lambda row: row.WPts + row.WFGM + row.WFTM - row.WFGA - row.WFTA + row.WDR + 0.5*row.WOR + row.WAst +row.WStl + 0.5*row.WBlk - row.WPF - row.WTO, axis=1)
        ltmp = df.apply(lambda row: row.LPts + row.LFGM + row.LFTM - row.LFGA - row.LFTA + row.LDR + 0.5*row.LOR + row.LAst +row.LStl + 0.5*row.LBlk - row.LPF - row.LTO, axis=1)
        df['WPIE'] = wtmp/(wtmp + ltmp)
        df['LPIE'] = ltmp/(wtmp + ltmp)

        # Build predictive model on advanced stats only
        adv_stats_df = df.drop(['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF'], axis=1)

        adv_stats_df.drop(['LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF'], axis=1, inplace=True)

        categories = adv_stats_df.columns.tolist()
        win_columns = list(filter(lambda s: s[0] == 'W', categories))
        complete_columns = [cat for cat in win_columns if cat not in ['WScore', 'WPts', 'WLoc', 'WTeamID']]
        complete_columns = list(map(lambda x: x.lstrip('W'), complete_columns))
        loss_columns = list(filter(lambda s: s[0] == 'L', categories))

        adv_dict = {}
        win_stats = df[df['WTeamID'] == tm_id]
        loss_stats = df[df['LTeamID'] == tm_id]
        num_win = win_stats.shape[0]
        num_loss = loss_stats.shape[0]
        games_played = num_win + num_loss

        win_stats = win_stats[win_columns]
        loss_stats = loss_stats[loss_columns]
        win_stats.drop(['WScore', 'WPts', 'WLoc', 'WTeamID'], axis=1, inplace=True)
        loss_stats.drop(['LScore', 'LPts', 'LTeamID'], axis=1, inplace=True)

        totals = list(map(sum, zip(win_stats.sum().tolist(), loss_stats.sum().tolist())))
        # adv_dicts.append({tm_id: list(map(lambda x: x / games_played, totals))})
        adv_dict[tm_id] = list(map(lambda x: x / games_played, totals))

        end_df = pd.DataFrame.from_dict(adv_dict, orient='index')
        end_df.dropna(axis=0, how='any', inplace=True)
        end_df.columns = complete_columns
        end_df.insert(loc=0, column='TeamID', value=end_df.index.tolist())
        end_df.sort_values(by=['TeamID'], inplace=True)
        end_df.reset_index(drop=True, inplace=True)
        filler = 2
        adv_stats_vals = end_df.values.tolist()[0]

        return adv_stats_vals


def GetElo(tm_id, year):
    elo_year = pd.read_csv('Data/RegSeasonStats/EloStats_' + str(year) + '.csv')
    elo_stat = elo_year[elo_year['TeamID'] == tm_id]['Ending_Elo'].values[0]
    return elo_stat


############################## MAIN PREPROCESSING FUNCTIONS ##############################

def getSeasonData(team_id, year):
    # The data frame below holds stats for every single game in the given year
    year_data_compact_pd = reg_season_compact_pd[reg_season_compact_pd['Season'] == year]
    year_data_pd = reg_season_detail_pd[reg_season_detail_pd['Season'] == year]

    # Advanced Stats calculations
    adv_stats = create_adv_stats(year_data_pd, team_id, year)
    adv_stat = adv_stats[1:]

    # ELO Calculations
    elo_stat = GetElo(team_id, year)
    filler = elo_stat
    # Combine all stats into final vector
    adv_stat.append(elo_stat)
    adv_stat.append(checkPower6Conference(team_id))
    return adv_stat

def createSeasonDict(year):
    seasonDictionary = collections.defaultdict(list)
    for team in teamList:
        team_id = teams_pd[teams_pd['TeamName'] == team].values[0][0]
        team_vector = getSeasonData(team_id, year)
        seasonDictionary[team_id] = team_vector
    return seasonDictionary


def createTrainingSet(years, saveYears):
    totalNumGames = 0
    for year in years:
        season = reg_season_compact_pd[reg_season_compact_pd['Season'] == year]
        totalNumGames += len(season.index)
        tourney = tourney_compact_pd[tourney_compact_pd['Season'] == year]
        totalNumGames += len(tourney.index)
    numFeatures = len(
        getSeasonData(1181, 2017))  # Just choosing a random team and seeing the dimensionality of the vector
    print(f"numFeatures: {numFeatures} based on 2017")
    xTrain = np.zeros((totalNumGames, numFeatures))# + 1))
    yTrain = np.zeros((totalNumGames))
    print(f"xTrain shape: {xTrain.shape}, yTrain shape: {yTrain.shape}, these are for the zero matrices")
    indexCounter = 0
    for year in years:
        team_vectors = createSeasonDict(year)
        season = reg_season_compact_pd[reg_season_compact_pd['Season'] == year]


        if (year in saveYears):
            np.save('Data/test_PrecomputedMatrices/TeamVectors/' + str(year) + 'TeamVectors', team_vectors)
    return xTrain, yTrain


def createAndSave(years, saveYears):
    xTrain, yTrain = createTrainingSet(years, saveYears)
    np.save('Data/test_PrecomputedMatrices/xTrain', xTrain)
    np.save('Data/test_PrecomputedMatrices/yTrain', yTrain)


############################## CREATE TRAINING SET ##############################

years = range(2003, 2019) #1993 - 2019
# Saves the team vectors for the following years
saveYears = range(2014, 2019)
# if os.path.exists("Data/test_PrecomputedMatrices/xTrain.npy") and os.path.exists("Data/test_PrecomputedMatrices/yTrain.npy"):
#     print('There is already a precomputed xTrain and yTrain.')
#     response = raw_input('Do you want to remove these files and create a new training set? (y/n) ')
#     # response = 'y'
#     if (response == 'y'):
#         os.remove("Data/test_PrecomputedMatrices/xTrain.npy")
#         os.remove("Data/test_PrecomputedMatrices/yTrain.npy")
#         createAndSave(years, saveYears)
#     else:
#         print('Okay, going to exit now.')
# else:
#     createAndSave(years, saveYears)
createAndSave(years, saveYears)