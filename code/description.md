# Documentation des Données Football

Ce document détaille la structure des données utilisées pour l'analyse et la prédiction de matchs de football. Les
colonnes sont regroupées par catégories thématiques.

---

## 1. Informations Générales & Identification

Données de base permettant d'identifier le contexte du match.

| Colonne      | Description                           |
|:-------------|:--------------------------------------|
| ``Div``      | Division (championnat)                |
| ``Season``   | Saison                                |
| ``League``   | Nom de la ligue (ex : Premier League) |
| ``Date``     | Date du match (aaaa/mm/jj)            |
| ``HomeTeam`` | Équipe évoluant à domicile            |
| ``AwayTeam`` | Équipe évoluant à l'extérieur         |

---

## 2. Résultats & Statistiques de Match (Brut)

Données constatées à la fin de la rencontre.

### Fin de match (Full Time)

- **FTHG / FTAG** : Buts marqués par l'équipe à domicile / extérieur.
- **FTR** : Résultat final (**1** : Domicile, **2** : Extérieur, **0** : Nul).

### Mi-temps (Half Time)

- **HTHG / HTAG** : Buts à la mi-temps (Dom/Ext).
- **HTR** : Résultat à la mi-temps (**1** : Domicile, **2** : Extérieur, **0** : Nul).

### Statistiques de Jeu

- **HS / AS** : Tirs (Dom/Ext).
- **HST / AST** : Tirs cadrés (Dom/Ext).
- **HC / AC** : Corners (Dom/Ext).
- **HF / AF** : Fautes commises (Dom/Ext).
- **HY / AY** : Cartons jaunes (Dom/Ext).
- **HR / AR** : Cartons rouges (Dom/Ext).

---

## 3. Indicateurs de Performance (Calculés)

Moyennes mobiles et indicateurs d'efficacité basés sur les matchs précédents.
(XX $\in$ {FT,HT})

### Domicile (Home) vs  Extérieur (Away)

| Catégorie             | Domicile          | Extérieur         | Description                                   |
|:----------------------|:------------------|:------------------|:----------------------------------------------|
| **Forme (FT/HT)**     | `XX_Hforme`       | `XX_Aforme`       | Moyenne des buts marqués (5 derniers matchs)  |
| **Attaque (FT)**      | `FT_Hatt`         | `FT_Aatt`         | Moyenne des tirs effectués                    |
| **Défense (FT/HT)**   | `XX_Hdef`         | `XX_Adef`         | Moyenne des tirs concédés                     |
| **Elo Score (FT/HT)** | `XX_Elo_H`        | `XX_Elo_A`        | Niveau de puissance de l'équipe (avant match) |
| **Précision**         | `FT_Hprecision`   | `FT_Aprecision`   | Ratio Tirs cadrés / Tirs totaux               |
| **Efficacité**        | `FT_Hprec_weight` | `FT_Aprec_weight` | Précision pondérée (poids buts = 1.5)         |

### Discipline & Repos (E=H,A pour Home et Away)

| Catégorie        | Domicile(H)  | Extérieur (A) | Description                                                               |
|:-----------------|:-------------|:--------------|:--------------------------------------------------------------------------|
| Fautes           | ``FT_HavgF`` | ``FT_AavgF``  | moyenne des fautes commisent sur 5 matchs                                 |
| Carton Jaunes(Y) | ``FT_HavgY`` | ``FT_AavgY``  | moyenne des cartons jaunes sur 5 matchs                                   |
| Carton Rouges(R) | ``FT_HavgR`` | ``FT_AavgR``  | moyenne des cartons rouges sur 5 matchs                                   |
| Repos            | ``HRepos``   | ``ARepos``    | Nombre de jours entre le dernier match et le match actuel (plafonné à 25) |

---

## 4. Variables de Comparaison (Diff & Ratios)

Indicateurs calculés pour mettre en évidence l'écart entre les deux adversaires.(XX = FT,HT)

> **Formule Différence :** `stat_home - stat_away`  
> **Formule Ratio :** `stat_home / (stat_away + 10^-6)`

| Catégorie    | Différence            | Ratio                    | 
|:-------------|:----------------------|:-------------------------|
| Forme        | `'XX_forme_diff'`     | ``XX_forme_ratio``       |
| Attaque      | `XX_att_diff`         | ``XX_att_ratio``         |
| Défense      | `XX_def_diff`         | ``XX_def_ratio``         |
| Précision    | `FT_prec_diff`        | ``FT_prec_ratio``        |
| Efficacité   | `FT_prec_weight_diff` | ``FT_prec_weight_ratio`` |
| Elo          | `XX_Elo_diff`         | ``XX_Elo_ratio``         |
| Repos        | `Repos_diff`          | ``Repos_ratio``          |
| Fautes       | `FT_avgF_diff`        | ``FT_avgF_ratio``        |
| Carton jaune | `FT_avgY_diff`        | ``FT_avgY_ratio``        |
| Carton rouge | `FT_avgR_diff`        | ``FT_avgR_ratio``        |

## 5. Variables Cibles (Labels)

Colonnes utilisées pour l'entraînement des modèles de Machine Learning.

- **Hvs** : Victoire Domicile (1 ou 0)
- **Avs** : Victoire Extérieur (1 ou 0)
- **Dvs** : Match Nul (1 ou 0)

---

## 6. Cotes des Bookmakers (Supprimées)

> **Note Importante :** Ces colonnes sont supprimées pour éviter le **Data Leakage**. Elles reflètent des probabilités
> de sortie qui fausseraient l'apprentissage réel du modèle.

- **Opérateurs :** Bet365, Bet&Win, Pinnacle, William Hill, VC Bet.
- **Variables concernées :** `B365H, B365D, B365A, BWH, BWD, BWA, PSH, PSD, PSA, WHH, WHD, WHA, VCH, VCD, VCA`.
- **Closing Odds (Ajustées) :** `PSCH, PSCD, PSCA`.
