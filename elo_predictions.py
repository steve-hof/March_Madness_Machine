import pandas as pd
import numpy as np


K = 20.
HOME_ADVANTAGE = 100.
ELO_WIDTH = 400
season = 2015


####################################################
# FUNCTIONS FOR CALCULATING ELO THROUGH REG SEASON #
####################################################
def elo_pred(elo1, elo2):
    return(1. / (10. ** (-(elo1 - elo2) / 400.) + 1.))


def expected_margin(elo_diff):
    return((7.5 + 0.006 * elo_diff))


def elo_update(w_elo, l_elo, margin):
    elo_diff = w_elo - l_elo
    pred = elo_pred(w_elo, l_elo)
    mult = ((margin + 3.) ** 0.8) / expected_margin(elo_diff)
    update = K * mult * (1 - pred)
    return(pred, update)


###########################
# MISCELLANEOUS FUNCTIONS #
###########################
def get_team_ID(team_name, teams_df):
    id = teams_df[teams_df['TeamName'] == team_name]['TeamID']
    return id.iloc[0]


def add_team_names(df, team_df):
    plus_team =pd.merge(df, team_df, how="left", on="TeamID")
    plus_team.drop(["FirstD1Season", "LastD1Season"], axis=1, inplace=True)
    return plus_team


#######################
# PREDICTION FUNCTION #
#######################
def pred_a(dictionary, team_a_ID, team_b_ID):
    a = dictionary[team_a_ID]
    b = dictionary[team_b_ID]
    return 1.0 / (1 + 10**((b - a) / ELO_WIDTH))


def main():
    reg_season = pd.read_csv("training_data/DataFiles/RegularSeasonCompactResults.csv")
    teams_df = pd.read_csv('training_data/DataFiles/Teams.csv')
    team_ids = teams_df[teams_df['LastD1Season'] >= season]['TeamID'].tolist()
    seeds = pd.read_csv('training_data/DataFiles/NCAATourneySeeds.csv')
    # tourn_compact_df = pd.read_csv('training_data/DataFiles/NCAATourneyCompactResults.csv')

    #######################################
    # CALCULATE FINAL REGULAR SEASON ELOs #
    #######################################
    # This dictionary will be used as a lookup for current
    # scores while the algorithm is iterating through each game
    elo_dict = dict(zip(list(team_ids), [1500] * len(team_ids)))

    # New columns to help us iteratively update elos
    reg_season['margin'] = reg_season.WScore - reg_season.LScore
    reg_season['w_elo'] = None
    reg_season['l_elo'] = None

    # I'm going to iterate over the games dataframe using
    # index numbereg_season, so want to check that nothing is out
    # of order before I do that.
    assert np.all(reg_season.index.values == np.array(range(reg_season.shape[0]))), "Index is out of order."

    preds = []
    reg_season = reg_season[reg_season['Season'] == season]
    reg_season.index = pd.RangeIndex(len(reg_season.index))
    reg_season.index = range(len(reg_season.index))

    # Loop over all rows of the games dataframe
    for i in range(reg_season.shape[0]):

        # Get key data from current row
        w = reg_season.at[i, 'WTeamID']
        l = reg_season.at[i, 'LTeamID']
        margin = reg_season.at[i, 'margin']
        wloc = reg_season.at[i, 'WLoc']

        # Does either team get a home-court advantage?
        w_ad, l_ad, = 0., 0.
        if wloc == "H":
            w_ad += HOME_ADVANTAGE
        elif wloc == "A":
            l_ad += HOME_ADVANTAGE

        # Get elo updates as a result of the game
        pred, update = elo_update(elo_dict[w] + w_ad,
                                  elo_dict[l] + l_ad,
                                  margin)
        elo_dict[w] += update
        elo_dict[l] -= update
        preds.append(pred)

        # Stores new elos in the games dataframe
        reg_season.loc[i, 'w_elo'] = elo_dict[w]
        reg_season.loc[i, 'l_elo'] = elo_dict[l]

    final_elo_dict = {}
    team_ids = teams_df[teams_df['LastD1Season'] >= season]['TeamID'].tolist()
    for id in team_ids:
        df = reg_season
        df = df[(df['WTeamID'] == id) | (df['LTeamID'] == id)]
        df = df.loc[df['DayNum'].idxmax()]
        w_mask = df.WTeamID == id
        # l_mask = df.LTeamID == id
        if w_mask:
            final_elo_dict[id] = df.loc['w_elo']
        else:
            final_elo_dict[id] = df.loc['l_elo']

    elo_df = pd.DataFrame(list(final_elo_dict.items()), columns=['TeamID', 'Ending_Elo'])
    elo_df = add_team_names(teams_df, elo_df)
    reg_season_standings = elo_df.sort_values(['Ending_Elo'], ascending=False)


    rate = dukewinner

    #########################################
    # DETERMINE IF PART OF POWER CONFERENCE #
    #########################################

if __name__ == '__main__':
    main()