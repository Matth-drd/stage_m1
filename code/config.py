# import sys
# import os
#
# sys.path.append(os.path.abspath('code'))
# import config as conf

split_ratio = .8
path_clean = '../data/csv/foot_v4.csv'
path_clean_cell = 'data/csv/foot_v4.csv'

targets = ["FTR", "Hvs", 'Avs', 'Dvs']

features = [
    'FT_Hforme', 'FT_Aforme', 'FT_Hatt', 'FT_Aatt', 'FT_Hdef', 'FT_Adef',
    'FT_forme_diff', 'FT_att_diff', 'FT_def_diff', 'HT_Hforme', 'HT_Aforme',
    'HT_Hdef', 'HT_Adef', 'HT_forme_diff', 'HT_def_diff', 'HRepos',
    'ARepos', 'Repos_diff', 'FT_Hprecision', 'FT_Aprecision',
    'FT_Hprec_weight', 'FT_Aprec_weight', 'FT_prec_diff',
    'FT_prec_weight_diff', 'FT_Elo_H', 'FT_Elo_A', 'FT_Elo_dif', 'HT_Elo_H',
    'HT_Elo_A', 'HT_Elo_dif', 'FT_HavgF', 'FT_AavgF', 'FT_avgF_diff',
    'FT_HavgY', 'FT_AavgY', 'FT_avgY_diff', 'FT_HavgR', 'FT_AavgR',
    'FT_avgR_diff', 'FT_forme_ratio', 'FT_att_ratio', 'FT_def_ratio',
    'HT_forme_ratio', 'HT_def_ratio', 'Repos_ratio', 'FT_prec_ratio',
    'FT_prec_weight_ratio', 'FT_Elo_ratio', 'HT_Elo_ratio', 'FT_avgF_ratio',
    'FT_avgY_ratio', 'FT_avgR_ratio', 'H_WinStreak', 'A_WinStreak',
    'H_LoseStreak', 'A_LoseStreak', 'WinStreak_diff', 'LoseStreak_diff', 'H_Rank', 'A_Rank', 'Rank_diff']
# print(len(features))
# grace au random forest, on trouve les features classées par importance.
# On choisis arbitrairement les 15 premières pour faire les modèles.
# features_importante = ['FT_Elo_dif', 'FT_Elo_ratio', 'HT_Elo_dif', 'HT_Elo_ratio', 'HT_Elo_H',
#                        'FT_Elo_A', 'FT_Elo_H', 'HT_Elo_A', 'FT_att_ratio', 'FT_def_ratio']
# On utilise la méthode RFECV pour extraire les features les plus importantes.
# Nous avons conservé les 10 plus importantes en commun entre Regression et Random Forest
"""
Utilisation de RFECV, SelectKBest, f_classif, mutual_info_classif pour 
séléctionner les meilleurs features, total de 12 features retenues.
"""
features_importante = [
    'FT_Aforme', 'FT_Elo_A', 'FT_Elo_H', 'FT_Elo_dif', 'FT_Hforme', 'FT_forme_diff',
    'FT_forme_ratio', 'HT_Elo_H', 'HT_Elo_dif', 'HT_Elo_ratio', 'HT_forme_diff', 'HT_forme_ratio']

# ['FT_Hprecision', 'FT_Hprec_weight', 'FT_Aprec_weight', 'FT_prec_diff', 'FT_prec_weight_diff',
#  'FT_Elo_H',
#  'FT_Elo_A', 'FT_Elo_dif', 'HT_Elo_H', 'HT_Elo_A', 'HT_Elo_dif', 'HT_Elo_ratio', 'FT_avgY_ratio']

RFECV_34 = ['FT_Aatt', 'FT_Adef', 'FT_Aforme', 'FT_Aprec_weight', 'FT_Aprecision', 'FT_Elo_A', 'FT_Elo_H',
            'FT_Elo_dif', 'FT_Elo_ratio', 'FT_Hdef', 'FT_Hforme', 'FT_Hprec_weight', 'FT_Hprecision',
            'FT_avgR_diff', 'FT_avgR_ratio', 'FT_avgY_diff', 'FT_avgY_ratio', 'FT_forme_diff', 'FT_forme_ratio',
            'FT_prec_diff', 'FT_prec_ratio', 'FT_prec_weight_ratio', 'HRepos', 'HT_Aforme', 'HT_Elo_A',
            'HT_Elo_H', 'HT_Elo_dif', 'HT_Elo_ratio', 'HT_Hdef', 'HT_Hforme', 'HT_def_ratio', 'HT_forme_diff',
            'HT_forme_ratio', 'Rank_diff']

ANOVA_34 = ['A_Rank', 'FT_Aatt', 'FT_Adef', 'FT_Aforme', 'FT_Elo_A', 'FT_Elo_H', 'FT_Elo_dif', 'FT_Elo_ratio',
            'FT_Hatt', 'FT_Hdef', 'FT_Hforme', 'FT_att_diff', 'FT_att_ratio', 'FT_def_diff', 'FT_def_ratio',
            'FT_forme_diff', 'FT_forme_ratio', 'FT_prec_diff', 'FT_prec_ratio', 'FT_prec_weight_diff',
            'FT_prec_weight_ratio', 'HT_Elo_A', 'HT_Elo_H', 'HT_Elo_dif', 'HT_Elo_ratio', 'HT_Hforme', 'HT_def_diff',
            'HT_def_ratio', 'HT_forme_diff', 'HT_forme_ratio', 'H_Rank', 'H_WinStreak', 'Rank_diff', 'WinStreak_diff']

Mutual_Info_34 = ['A_Rank', 'A_WinStreak', 'FT_Adef', 'FT_Aforme', 'FT_Aprec_weight', 'FT_Aprecision', 'FT_Elo_A',
                  'FT_Elo_H',
                  'FT_Elo_dif', 'FT_Elo_ratio', 'FT_Hatt', 'FT_Hforme', 'FT_Hprec_weight', 'FT_Hprecision',
                  'FT_att_diff',
                  'FT_att_ratio', 'FT_avgF_diff', 'FT_def_diff', 'FT_def_ratio', 'FT_forme_diff', 'FT_forme_ratio',
                  'FT_prec_diff', 'FT_prec_ratio', 'FT_prec_weight_diff', 'HT_Elo_A', 'HT_Elo_H', 'HT_Elo_dif',
                  'HT_Elo_ratio',
                  'HT_Hforme', 'HT_def_ratio', 'HT_forme_diff', 'HT_forme_ratio', 'Rank_diff', 'WinStreak_diff']

ECD = ['FT_Aprec_weight', 'FT_avgY_diff', 'HT_Elo_A', 'HT_Elo_dif', 'HT_Hdef', 'WinStreak_diff']

betting = ['B365H', 'B365D', 'B365A']
