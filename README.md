# Prédiction des Taux de Change & Risque de Change — BIAT
**Stage ingénieur MINDS — ENIT | 6 semaines**

## Objectif
Prédiction des cours de change TND/EUR et TND/USD + mesure et couverture du risque de change à la BIAT.

## Structure du projet
```
biat_stage/
├── data/
│   ├── raw/          ← données brutes (BCT, yfinance, FRED)
│   └── processed/    ← données nettoyées + sigma_t GARCH exporté
├── notebooks/        ← exploration et prototypage (un notebook par semaine)
├── src/
│   ├── data.py       ← collecte + nettoyage des données
│   ├── models.py     ← ARIMA, GARCH, LSTM (PyTorch), XGBoost
│   ├── risk.py       ← VaR, CVaR, stress testing
│   ├── coverage.py   ← Forward pricing, Garman-Kohlhagen
│   └── portfolio.py  ← Optimisation Markowitz
├── dashboard/
│   └── app.py        ← Application Streamlit (importe src/)
├── report/           ← Rapport académique
└── requirements.txt
```

## Stack technique
- **Langage** : Python 3.10+
- **Deep Learning** : PyTorch
- **Séries temporelles** : statsmodels, arch, pmdarima
- **ML** : XGBoost, scikit-learn, shap
- **Risque** : scipy, numpy
- **Portfolio** : PyPortfolioOpt
- **Dashboard** : Streamlit, Plotly
- **Données** : yfinance, pandas-datareader (FRED)

## Données
- **Cours spot** : BCT (Banque Centrale de Tunisie) + yfinance
- **Variables macro** : FRED (Brent, taux directeur, réserves BCT, IPC)
- **Période d'entraînement** : 2015 – 2022
- **Période de test (backtesting)** : 2023 – 2025
- **Profils clients** : simulés (5 profils)

## Axes du projet
1. **Axe 1** — Prédiction des cours de change (ARIMA, GARCH, LSTM, XGBoost)
2. **Axe 2** — Mesure du risque (VaR, CVaR, Stress Testing)
3. **Axe 2+** — Couverture (Forward, Options Garman-Kohlhagen)
4. **Bonus** — Optimisation de portefeuille (Markowitz EUR+USD)

## Équipe
- Youssef Neji
- Binôme

**Encadrant BIAT** : réunion de validation hebdomadaire
