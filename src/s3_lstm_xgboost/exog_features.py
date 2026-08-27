# =============================================================
#  src/s3_lstm_xgboost/exog_features.py
#  Sélection des variables exogènes pour LSTM/MLP — propre à S3,
#  DÉCOUPLÉE d'ARIMAX (src/s2_arima_garch/arimax.py).
#
#  ARIMAX est un modèle exclu de la production (validation croisée
#  + Diebold-Mariano) : sa sélection de variables n'a plus vocation
#  à servir de référence pour LSTM/MLP.
#
#  Historique des révisions :
#    - 2026 (initial) : copie de l'ancienne sélection ARIMAX +
#      features BCT dérivées (reserves_mom_pct, bct_market_share_pct)
#    - 2026-08 (feature_selection.py) : validation empirique par
#      RMSE MLP réel sur fenêtre de validation interne (2022-2023),
#      4 candidats testés par devise/horizon (Actuel / Data-driven
#      top-5 / Aucun / Toutes disponibles). Résultats :
#
#        USD J+1  : Actuel gagne  (RMSE=0.010441, marge <0.1% —
#                   quasi-bruit, INCHANGÉ)
#        USD J+7  : Aucun gagne   (RMSE=0.030269, marge ~0.4% vs
#                   Actuel — signal faible, INCHANGÉ par prudence)
#        USD J+30 : Actuel gagne  (RMSE=0.069840, marge ~0.8% —
#                   signal modéré, INCHANGÉ)
#        EUR J+1  : Data-driven gagne (RMSE=0.007639, marge <0.1%
#                   vs Actuel — quasi-bruit, INCHANGÉ par prudence)
#        EUR J+7  : Data-driven gagne NETTEMENT (RMSE=0.018343 vs
#                   0.018942 Actuel, +3.3% — MODIFIÉ ci-dessous)
#        EUR J+30 : Aucun gagne NETTEMENT (RMSE=0.035038 vs
#                   0.036877 Actuel, +5.2% — MODIFIÉ ci-dessous,
#                   AUCUNE variable macro pour cette config)
#
#      Note importante : le candidat "Data-driven (top-K)" issu du
#      classement corrélation/RF n'est pas systématiquement fiable
#      — il a été le PIRE candidat sur EUR J+30 (+23% de RMSE vs
#      "Aucun"). Ne jamais adopter un classement exploratoire sans
#      validation empirique derrière.
#
#      ⚠️ Basé sur UN SEUL run (5 epochs, seed=42) — les marges
#      <1% (USD J+1/J+7, EUR J+1) n'ont pas été jugées assez
#      fiables pour justifier un changement et sont restées
#      inchangées. Seuls EUR J+7 et EUR J+30, aux marges plus
#      larges, ont été appliqués.
#
#      Prochaine étape obligatoire : relancer run_s3.py (mode
#      complet) pour valider ces 2 changements sur le vrai test
#      (2024-2026) — jamais d'adoption sans ce passage final.
#
#  Auteur : Youssef Neji | MINDS ENIT | 2026
# =============================================================

EXOG_FEATURES_S3 = {
    "USD": {
        1:  ["VIX", "EURUSD", "Brent_USD", "DXY",
             "reserves_mom_pct", "bct_market_share_pct"],
        7:  ["VIX", "EURUSD", "Brent_USD", "DXY",
             "reserves_mom_pct", "bct_market_share_pct"],
        30: ["TauxFed_USD", "IPC_USA", "TauxMMBCT", "US10Y",
             "reserves_mom_pct", "bct_market_share_pct"],
    },
    "EUR": {
        1:  ["EURUSD", "VIX", "TauxBCE_EUR", "EUR10Y",
             "reserves_mom_pct", "bct_market_share_pct"],
        # MODIFIÉ 2026-08 — Data-driven (top-K), RMSE validation
        # 0.018343 vs 0.018942 pour l'ancienne sélection (+3.3%)
        7:  ["IPC_Tunisie", "EURUSD", "reserves_mom_pct",
             "bct_market_share_pct", "reserves_officielles_mdt"],
        # MODIFIÉ 2026-08 — Aucune variable macro, RMSE validation
        # 0.035038 vs 0.036877 pour l'ancienne sélection (+5.2%).
        # Le modèle s'appuie uniquement sur lags/MA/volatilité/
        # momentum/sigma GARCH/dummies de régime pour cette config.
        30: [],
    },
}


def get_exog_features_s3(horizon: int, currency: str) -> list:
    """Retourne la liste des variables macro pour cette devise et
    cet horizon. Liste vide si combinaison non définie OU si la
    validation empirique a montré qu'aucune variable macro n'aide
    (cas EUR J+30) — build_features() gère déjà ce cas proprement."""
    return list(EXOG_FEATURES_S3.get(currency, {}).get(horizon, []))