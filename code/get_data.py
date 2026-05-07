import pandas as pd


# Documentation des colonnes disponible sur : https://www.football-data.co.uk/notes.txt

def top_5_leagues():
    """
    Récupère les données historiques des 5 grands championnats européens.

    Parcourt les 10 dernières saisons (de 2014/2015 à 2023/2024) pour la Premier League,
    Ligue 1, Bundesliga, Serie A et La Liga via l'API de football-data.co.uk.

    Returns:
        pd.DataFrame: Un seul DataFrame contenant l'ensemble des matchs compilés.
    """
    # Configuration des ligues (Code source: Nom lisible)
    leagues = {
        'E0': 'Premier League',
        'F1': 'Ligue 1',
        'D1': 'Bundesliga',
        'I1': 'Serie A',
        'SP1': 'La Liga'
    }

    # Liste des saisons au format AA/AA
    seasons = ['1415', '1516', '1617', '1718', '1819',
               '1920', '2021', '2122', '2223', '2324']

    base_url = "https://www.football-data.co.uk/mmz4281/"
    all_dataframes = []

    for s in seasons:
        for code, name in leagues.items():
            url = f"{base_url}{s}/{code}.csv"
            try:
                # 'unicode_escape' gère les caractères spéciaux dans les noms d'équipes
                df = pd.read_csv(url, encoding='unicode_escape')

                # Ajout des métadonnées de contexte
                df['Season'] = s
                df['League'] = name
                all_dataframes.append(df)
            except Exception as e:
                print(f"Erreur lors du téléchargement de {name} ({s}) : {e}")

    # Fusion de tous les fichiers CSV en un seul objet
    dataset = pd.concat(all_dataframes, ignore_index=True)
    return dataset


def clean_data(df):
    """
    Nettoie le DataFrame en supprimant le bruit et en gérant les valeurs manquantes.

    Logique de nettoyage :
    1. Supprime les colonnes ayant plus de 8 900 NaN (colonnes très peu renseignées).
    2. Supprime des colonnes spécifiques jugées non pertinentes.
    3. Impute les valeurs manquantes de la colonne 'Div' en fonction de l'équipe à domicile.
    4. Supprime les lignes résiduelles contenant des NaN.

    Args:
        df (pd.DataFrame): Le DataFrame brut issu de top_5_leagues.

    Returns:
        pd.DataFrame: Le DataFrame nettoyé.
    """
    df = df.copy()

    # 1. Suppression des colonnes vides ou quasi-vides
    for c in df.columns:
        if df[c].isna().sum() >= 8_900:
            df.drop(columns=[c], inplace=True)
        # 2. Suppression de colonnes de paris spécifiques peu remplies
        elif c in ['IWD', "IWA", "IWH"]:
            df.drop(columns=[c], inplace=True)

    # 3. Récupération de la division ('Div') manquante
    # On crée un dictionnaire associant chaque division à ses équipes connues
    homeT = {}
    for div in df['Div'].dropna().unique():
        team_par_div = df.loc[df['Div'] == div, 'HomeTeam'].unique()
        homeT[div] = team_par_div

    # Si 'Div' est NaN, on cherche l'équipe à domicile dans
    # notre dictionnaire pour réassigner la division
    for div, equipes in homeT.items():
        mask = df['Div'].isna() & df['HomeTeam'].isin(equipes)
        df.loc[mask, 'Div'] = div

    # 4. Suppression des dernières lignes avec des valeurs manquantes
    df.dropna(axis=0, inplace=True)
    return df


def modif_type(df):
    """
    Convertit les types de données et numérise les variables catégorielles.

    Opérations :
    - Conversion des scores (buts) de float vers int.
    - Mapping des résultats (FTR/HTR) : Domicile=1, Extérieur=2, Nul=0.

    Args:
        df (pd.DataFrame): Le DataFrame nettoyé.
    Returns:
        pd.DataFrame: Le DataFrame avec les types corrigés.
    """
    df = df.copy()

    # Conversion des colonnes de buts pour un format plus propre
    cols_to_int = ["FTHG", "FTAG", "HTAG", "HTHG"]
    for col in cols_to_int:
        df[col] = df[col].astype(int)

    # Encodage numérique des résultats pour faciliter les futures
    # analyses de corrélation
    mapping = {'H': 1, 'A': 2, 'D': 0}
    for col in ["FTR", "HTR"]:
        df[col] = df[col].map(mapping)

    # df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    # Traitement du format date
    df['Date'] = df['Date'].astype(str)
    mask_2digits = df['Date'].str.split('/').str[-1].str.len() == 2
    df.loc[mask_2digits, 'Date'] = pd.to_datetime(df.loc[mask_2digits, 'Date'], format='%d/%m/%y', errors='coerce')
    df.loc[~mask_2digits, 'Date'] = pd.to_datetime(df.loc[~mask_2digits, 'Date'], format='%d/%m/%Y', errors='coerce')

    df.sort_values('Date', inplace=True)
    return df


if __name__ == "__main__":
    df = top_5_leagues()
    df_clean = clean_data(df)
    df_final = modif_type(df_clean)

    # Affichage du bilan
    print(f"Dimensions finales : {df_final.shape[0]} matchs, {df_final.shape[1]} variables.")

    # Exportation
    PATH = "../data/csv/foot_.csv"
    df_final.to_csv(PATH, index=False)
    print(f"Fichier sauvegardé avec succès dans {PATH}")
