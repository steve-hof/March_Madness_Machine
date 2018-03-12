import pandas as pd
import numpy as np
from sklearn import preprocessing
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


################################
# POWER CONFERENCE FUNCTION(S) #
################################
def determine_power(curr_df):
    conf_df = pd.read_csv('training_data/DataFiles/TeamConferences.csv')
    power_conf = ['acc', 'big_twelve', 'big_ten', 'pac_twelve', 'sec', 'big_east']
    conf_df = conf_df[conf_df['Season'] == season]
    conf_df['pow_bool'] = np.where(conf_df['ConfAbbrev'].isin(power_conf), 1, 0)

    team_li = conf_df['TeamID'].tolist()
    bool_li = conf_df['pow_bool'].tolist()
    pow_dict = dict(zip(team_li, bool_li))
    temp = [k for k, v in pow_dict.items() if v == 1]

    curr_df['pow_bool'] = np.where(curr_df['TeamID'].isin(temp), 1, 0)

    cols = curr_df.columns.tolist()
    ord_cols = cols[:2] + cols[-1:] + cols[2:3]
    curr_df = curr_df[ord_cols]

    return curr_df


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


def only_teams_in_tourney(df, season, seeds_df):
    seeds_df = seeds_df[seeds_df['Season'] == season]
    df = df[df['TeamID'].isin(seeds_df['TeamID'])]
    df.reset_index(drop=True, inplace=True)
    return df


def list_teams_in_tourney(seeds_df, season):
    return seeds_df[seeds_df['Season'] == season]['TeamID'].tolist()


def normalize_dataframe(df, starting_col):
    vals = df.values[:, starting_col:]
    cols = df.columns[starting_col:].tolist()
    normed_vals = preprocessing.normalize(vals, norm='l2')
    df[cols] = normed_vals
    return df


def scale_dataframe(df, starting_col):
    vals = df.values[:, starting_col:]
    cols = df.columns[starting_col:].tolist()
    normed_vals = preprocessing.scale(vals)
    df[cols] = normed_vals
    return df

#######################
# PREDICTION FUNCTION #
#######################
def pred_a(dictionary, team_a_ID, team_b_ID):
    a = dictionary[team_a_ID]
    b = dictionary[team_b_ID]
    return 1.0 / (1 + 10**((b - a) / ELO_WIDTH))


############################
# ADVANCED STATS FUNCTIONS #
############################
def create_adv_stats(df):
    ##########################
    # CALCULATING POSSESSION #
    ##########################

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

    return adv_stats_df


# Summarize advanced stats
def get_adv_stats(df, seeds_df, season):
    teams = list_teams_in_tourney(seeds_df, season)
    teams = teams
    categories = df.columns.tolist()
    win_columns = list(filter(lambda s: s[0] == 'W', categories))
    complete_columns = [cat for cat in win_columns if cat not in ['WScore', 'WPts', 'WLoc', 'WTeamID']]
    complete_columns = list(map(lambda x: x.lstrip('W'), complete_columns))
    loss_columns = list(filter(lambda s: s[0] == 'L', categories))

    # tm_id = 1181
    adv_dict = {}
    for tm_id in teams:
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
        end_df.columns = complete_columns
        end_df.insert(loc=0, column='TeamID', value=end_df.index.tolist())
        end_df.sort_values(by=['TeamID'], inplace=True)
        end_df.reset_index(drop=True, inplace=True)

    return end_df


########################
# PREDICTION FUNCTIONS #
########################
def get_ranking(df, season):
    data_cols = df.columns[4:]
    df['Sum'] = df[data_cols].sum(axis=1)
    df.sort_values(by=['Sum'], inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    return df


def main():
    reg_season = pd.read_csv("training_data/DataFiles/RegularSeasonCompactResults.csv")
    teams_df = pd.read_csv('training_data/DataFiles/Teams.csv')
    team_ids = teams_df[teams_df['LastD1Season'] >= season]['TeamID'].tolist()
    tourn_compact_df = pd.read_csv('training_data/DataFiles/NCAATourneyCompactResults.csv')
    reg_detail_df = pd.read_csv('training_data/DataFiles/RegularSeasonDetailedResults.csv')
    seeds_df = pd.read_csv('training_data/DataFiles/NCAATourneySeeds.csv')

    #######################################
    # CALCULATE AND RETURN ADVANCED STATS #
    #######################################
    reg_detail_df = reg_detail_df[reg_detail_df['Season'] == season]
    reg_adv_stats = create_adv_stats(reg_detail_df)
    final_adv_stats = get_adv_stats(reg_adv_stats, seeds_df, season)

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

    final_df = pd.DataFrame(list(final_elo_dict.items()), columns=['TeamID', 'Ending_Elo'])
    final_df = add_team_names(teams_df, final_df)
    # reg_season_standings = final_df.sort_values(['Ending_Elo'], ascending=False)

    # Add power conference
    final_df = determine_power(final_df)

    # Clean NaN before adv stats computations
    final_df = final_df[np.isfinite(final_df['Ending_Elo'])]
    # Only teams in tourney
    final_df = only_teams_in_tourney(final_df, season, seeds_df)

    # Combine dataframes
    reg_season_final_df = pd.merge(final_df, final_adv_stats, how='left', on='TeamID')
    reg_season_final_df.insert(loc=1, column='Season', value=season)
    reg_season_final_df.to_csv('input_data')
    # Normalize data
    normalized_df = normalize_dataframe(reg_season_final_df, starting_col=4)
    scaled_df = scale_dataframe(reg_season_final_df, starting_col=4)

    ###############
    # PREDICTIONS #
    ###############
    scaled_ranking = get_ranking(scaled_df, season)
    normed_ranking = get_ranking(normalized_df, season)


    foo = 12


if __name__ == '__main__':
    main()