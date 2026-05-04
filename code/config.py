path_clean = '../data/csv/foot_v4.csv'
path_clean_cell = 'data/csv/foot_v4.csv'

targets = ["FTR", "Hvs", 'Avs', 'Dvs']

features = [
    'FT_Hforme', 'FT_Aforme', 'FT_Hatt', 'FT_Aatt',
    'FT_Hdef', 'FT_Adef', 'FT_forme_diff', 'FT_att_diff',
    'FT_def_diff', 'HT_Hforme', 'HT_Aforme', 'HT_Hdef',
    'HT_Adef', 'HT_forme_diff', 'HT_def_diff', 'HRepos',
    'ARepos', 'Repos_diff', 'FT_Hprecision', 'FT_Aprecision',
    'FT_Hprec_weight', 'FT_Aprec_weight', 'FT_prec_diff',
    'FT_prec_weight_diff', 'FT_Elo_H', 'FT_Elo_A', 'FT_Elo_dif',
    'HT_Elo_H', 'HT_Elo_A', 'HT_Elo_dif', 'FT_HavgF', 'FT_AavgF',
    'FT_avgF_diff', 'FT_HavgY', 'FT_AavgY', 'FT_avgY_diff', 'FT_HavgR',
    'FT_AavgR', 'FT_avgR_diff', 'FT_forme_ratio', 'FT_att_ratio', 'FT_def_ratio',
    'HT_forme_ratio', 'HT_def_ratio', 'Repos_ratio', 'FT_prec_ratio',
    'FT_prec_weight_ratio', 'FT_Elo_ratio', 'HT_Elo_ratio', 'FT_avgF_ratio',
    'FT_avgY_ratio', 'FT_avgR_ratio', 'H_WinStreak', 'A_WinStreak', 'H_LoseStreak', 'A_LoseStreak']

# grace au random forest, on trouve les features classées par importance.
# On choisis arbitrairement les 15 premières pour faire les modèles.
features_importante = ['FT_Elo_dif', 'FT_Elo_ratio', 'HT_Elo_dif', 'HT_Elo_ratio', 'HT_Elo_H',
                        'FT_Elo_A', 'FT_Elo_H', 'HT_Elo_A', 'FT_att_ratio', 'FT_def_ratio']

betting = ['B365H', 'B365D', 'B365A']
