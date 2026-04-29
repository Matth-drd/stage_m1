#  Documentation des Données Football

---

##  INFORMATIONS GÉNÉRALES

- **Div** : Division (championnat)
- **Date** : Date du match (jj/mm/aa)
- **Season** : Saison
- **League** : Nom de la ligue (ex : Premier League)
- **HomeTeam** : Équipe à domicile
- **AwayTeam** : Équipe à l'extérieur

---

##  RÉSULTATS DU MATCH

- **FTHG** *(Full Time Home Goals)* : Buts équipe à domicile en fin de match  
- **FTAG** *(Full Time Away Goals)* : Buts équipe à l'extérieur en fin de match  
- **FTR** *(Full Time Result)* : Résultat final  
  - Domicile = 1  
  - Extérieur = 2  
  - Nul = 0  

- **HTHG** *(Half Time Home Goals)* : Buts domicile à la mi-temps  
- **HTAG** *(Half Time Away Goals)* : Buts extérieur à la mi-temps  
- **HTR** *(Half Time Result)* : Résultat à la mi-temps  
  - Domicile = 1  
  - Extérieur = 2  
  - Nul = 0  

---

##  STATISTIQUES DE JEU

- **HS** : Tirs domicile  
- **AS** : Tirs extérieur  
- **HST** : Tirs cadrés domicile  
- **AST** : Tirs cadrés extérieur  
- **HF** : Fautes domicile  
- **AF** : Fautes extérieur  
- **HC** : Corners domicile  
- **AC** : Corners extérieur  

---

#  STAT VALEUR AJOUTÉE

---

## INDICATEURS DE PERFORMANCE — FULL TIME

###  Domicile

- **FT_Hforme** : Moyenne des buts marqués (5 derniers matchs)
- **FT_Hatt** : Moyenne des tirs effectués
- **FT_Hdef** : Moyenne des tirs concédés (↓ = meilleure défense)

### ️ Extérieur

- **FT_Aforme** : Moyenne des buts marqués
- **FT_Aatt** : Moyenne des tirs effectués
- **FT_Adef** : Moyenne des tirs concédés (↓ = meilleure défense)

---

##  INDICATEURS MI-TEMPS (HALF TIME)

###  Domicile

- **HT_Hforme** : Moyenne des buts à la mi-temps
- **HT_Hatt** : Pression offensive (proxy basé sur les buts)
- **HT_Hdef** : Moyenne des buts encaissés

### ✈ Extérieur

- **HT_Aforme** : Moyenne des buts à la mi-temps
- **HT_Aatt** : Pression offensive (proxy)
- **HT_Adef** : Moyenne des buts encaissés

---

##  PRÉCISION & EFFICACITÉ

- **FT_Hprecision** : Tirs cadrés / tirs (domicile)
- **FT_Aprecision** : Tirs cadrés / tirs (extérieur)

- **FT_Hprec_weight** : Précision pondérée (poids sur les buts = 1.5)
- **FT_Aprec_weight** : Idem extérieur

---

##  SCORE ELO 

- **FT_Elo_H** : Elo domicile avant match  
- **FT_Elo_A** : Elo extérieur avant match  
- **FT_Elo_dif** : Différence Elo (avantage domicile si positif)

- **HT_Elo_H** : Elo mi-temps domicile  
- **HT_Elo_A** : Elo mi-temps extérieur  
- **HT_Elo_dif** : Différence Elo mi-temps  

---

## ⚔ DISCIPLINE & IMPACT PHYSIQUE

###  Domicile
- **FT_HavgF** : Fautes moyennes  
- **FT_HavgY** : Cartons jaunes moyens  
- **FT_HavgR** : Cartons rouges moyens  

### ️ Extérieur
- **FT_AavgF** : Fautes moyennes  
- **FT_AavgY** : Cartons jaunes moyens  
- **FT_AavgR** : Cartons rouges moyens  

---

##  CONTEXTE & REPOS

- **HRepos** : Jours de repos domicile (max 25)
- **ARepos** : Jours de repos extérieur (max 25)

---

##  VARIABLES CIBLES (LABELS)

- **Hvs** : Victoire domicile (1/0)
- **Avs** : Victoire extérieur (1/0)
- **Dvs** : Match nul (1/0)

---

## ️ CALCULS INTERMÉDIAIRES

- **FT_Hshot / FT_Ashot** : Volume de tirs ajusté  


---

##  SANCTIONS

- **HY** : Cartons jaunes domicile  
- **AY** : Cartons jaunes extérieur  
- **HR** : Cartons rouges domicile  
- **AR** : Cartons rouges extérieur  

---

##  COTES DES BOOKMAKERS

- **Bet365** : B365H, B365D, B365A  
- **Bet&Win** : BWH, BWD, BWA  
- **Pinnacle** : PSH, PSD, PSA  
- **William Hill** : WHH, WHD, WHA  
- **VC Bet** : VCH, VCD, VCA  

---

##  COTES AJUSTÉES (CLOSING)

- **PSCH, PSCD, PSCA** : Cotes Pinnacle ajustées avant match  

---

##  COLONNES DE PARI 

```python
betting_cols = [
  "B365H", "B365A", "B365D", "BWA", "BWH", "BWD",
  "PSH", "PSA", "PSD", "WHD", "WHH", "WHA",
  "VCH", "VCD", "VCA", "PSCD", "PSCH", "PSCA"
]