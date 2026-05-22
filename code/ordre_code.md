# Ordre de compilation des codes


EDA des données (analyce exploratoire) :
    
    1. 'get_data.py': récupération + nettoyage des données
    2. 'preprocess_data.py' : nettoyage + création de features
    3. 'visu_data.py' : réduction et visualisation des données (via UMAP + ACP)
    4. 'selecet_features.py' : sélection de features 
    5. 'analyse_feature.py' : vérifie la pertinence des features via une régression logistique simple
    6. 'ranking_bookmaker.py' : classe les bookmakers, permettra également de comparer notre modèle aux bookmakers

Implémentation des modèles :

    "modeles_et_comparaison.py" : comparaison des modèles


