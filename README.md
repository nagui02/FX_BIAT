# Pipeline Quantitatif de Gestion du Risque de Change — BIAT

Pipeline complet de gestion quantitative du risque de change (TND/USD, TND/EUR), développé dans le cadre d'un stage d'ingénieur à la **BIAT** (Banque Internationale Arabe de Tunisie), Direction des Risques.

> Stage encadré par **M. Nader Trigui**, réalisé par **Youssef Neji**, 2ᵉ année ingénieur MINDS — ENIT (École Nationale d'Ingénieurs de Tunis).
> Projet finalisé et soutenu.

## Objectif

Construire un pipeline de bout en bout — de la donnée brute à la décision de couverture — pour analyser et gérer le risque de change de la banque et de ses clients, structuré autour d'un fil conducteur en trois temps :

**le taux de change est imprévisible → le risque est mesurable → le risque est gérable.**

## Architecture du pipeline

| Étape | Contenu | Résultat clé |
|---|---|---|
| **S1 — Données** | Collecte BCT (fichiers XLS déguisés en HTML) + 12 séries FRED | 5 898 observations journalières (2004–2026), 24 features, lag de publication de 20 jours appliqué aux données mensuelles BCT (anti-look-ahead) |
| **S2 — ARIMA/GARCH** | Modélisation de la moyenne et de la volatilité | Correction d'un bug critique (prévision GARCH statique multi-step → récursion rolling 1-step), avec impact direct sur les primes d'options en aval |
| **S3 — LSTM/MLP** | Comparaison de modèles ML vs. modèles classiques | 36 tests de Diebold-Mariano : aucun modèle ne bat significativement la marche aléatoire ; ARIMAX formellement exclu (0/24 gains significatifs en conditions de production) |
| **S4 — VaR/CVaR** | 4 méthodes de VaR, 5 profils clients, backtesting de Kupiec | La VaR historique est la seule méthode validée sur toutes les configurations testées |
| **S5 — Couverture** | Pricing forward, options Garman-Kohlhagen, optimisation Markowitz | Portefeuille à variance minimale ≈ 69,7 % EUR / 30,3 % USD ; bénéfice de diversification de 43 % pour un desk 50/50 EUR/USD |

Cinq profils clients simulés sont utilisés à travers toutes les étapes : **ALPHA SARL** (importateur EUR), **BETA Export** (exportateur USD), **M. Gamma** (particulier EUR), **Desk BIAT** (multi EUR+USD), **Dette BIAT** (emprunteur USD).

## Structure du dépôt

```
biat_stage/
├── dashboard/                  # Application Streamlit (5 pages)
│   ├── app.py
│   ├── pages/
│   │   ├── 1_Données.py
│   │   ├── 2_Prévisions.py
│   │   ├── 3_Risque.py
│   │   ├── 4_Couverture.py
│   │   └── 5_Markowitz.py
│   └── utils/
├── src/                        # Pipeline de traitement, un module par étape
│   ├── s1_data/                # Collecte et nettoyage BCT + FRED
│   ├── s2_arima_garch/         # ARIMA, ARIMAX, GARCH, Diebold-Mariano
│   ├── s3_lstm_xgboost/        # LSTM, MLP, sélection de features
│   ├── s4_risk/                # VaR/CVaR, stress testing
│   └── s5_hedging/             # Forward, options, Markowitz
├── data/
│   ├── live/                   # Derniers taux et prévisions
│   └── models/                 # Modèles entraînés (ARIMA, ARIMAX, MLP)
├── report/                     # Figures et scripts de génération pour le rapport
├── requirements.txt
└── README.md
```

## Lancer le dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Stack technique

Python (pandas, statsmodels/arch, PyTorch, scikit-learn), Streamlit, LaTeX (rapport, template ENIT), Canva (support de soutenance).

## Principes méthodologiques

- **Discipline anti-look-ahead** : chaque feature a été évaluée sous l'angle de sa disponibilité réelle en production (lag de publication BCT, exclusion d'ARIMAX).
- **Résultats négatifs assumés** : la détection de régimes par modèle de Markov et l'ARIMAX ont été testés, ont échoué à des seuils fixés à l'avance, et sont documentés comme tels plutôt que retirés silencieusement.
- **Interdépendance des étapes** : la correction du bug GARCH en S2 a nécessité une propagation des mises à jour jusqu'en S5.

---

*Les données et résultats numériques référencés dans ce projet sont issus d'un stage académique encadré et s'appuient sur des séries publiques (BCT, FRED) combinées à des profils clients simulés.*