# Ordre de compilation des codes


EDA des données (analyce exploratoire) :
    
    1. 'get_data.py': récupération + nettoyage des données
    2. 'preprocess_data.py' : nettoyage + création de features
    3. 'selecet_features.py' : sélection de features
    4. 'visu_data.py' : réduction et visualisation des données (via UMAP + ACP)
    5. 'boxplot.py' : affichage des boxplots pour visualiser les features
    6. 'analyse_feature.py' : vérifie la pertinence des features via une régression logistique simple
    7. 'ranking_bookmaker.py' : classe les bookmakers, permettra également de comparer notre modèle aux bookmakers

Implémentation des modèles :

    "poisson.py" : implémentation du modèle de Poisson (Maher, Dixon-Coles)
    "modeles_et_comparaison.py" : implémentation et comparaison des modèles de ML

Comparaison des modèles :

    "comparaison_ML_stat.py" : compare les modèles ML/Stat entre eux et avec le bookmaker Bwin


[//]: # (Outil de prédiction : )

[//]: # (    )
[//]: # (    'pred_stat.py' : récupère les data et prédit l'issue d'un match)

