# =============================================================
#  src/s3_lstm_xgboost/config.py
#  Configuration S3 — LSTM
#  Auteur : Youssef Neji | MINDS ENIT | 2025-2026
# =============================================================

import os

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "..", "data", "processed")
FIGURES_DIR   = os.path.join(BASE_DIR, "..", "..", "report", "figures")
MODELS_DIR    = os.path.join(BASE_DIR, "..", "..", "data", "models")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# -------------------------------------------------------------
#  Fenêtres temporelles
# -------------------------------------------------------------
# IMPORTANT : alignée sur s2_arima_garch.config pour garantir
# que tous les modèles (ARIMA, ARIMAX, GARCH, LSTM) sont
# entraînés/testés sur exactement les mêmes périodes — condition
# nécessaire pour que les comparaisons de RMSE et les tests de
# Diebold-Mariano soient valides. Toute modification ici doit
# être répercutée dans s2_arima_garch/config.py (et vice versa).

TRAIN_START = "2012-01-01"   # était "2004-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2026-07-31"   # mis à jour après téléchargement4"
# ── Reproductibilité ─────────────────────────────────────────
SEED = 42
HORIZONS = [1, 7, 30]

# -------------------------------------------------------------
#  Hyperparamètres LSTM — communs à tous les horizons
# -------------------------------------------------------------
LSTM_CONFIG_BASE = {
    "sequence_length": 30,    # 30 jours ouvrés d'historique
    "batch_size":      32,
    "learning_rate":   1e-3,
    "epochs":          100,
    "patience":        10,
}

# -------------------------------------------------------------
#  Hyperparamètres LSTM — spécifiques par horizon
# -------------------------------------------------------------
# Motivation : avec ~2 870 observations d'entraînement, un LSTM
# de pleine capacité (hidden_size=64, num_layers=2 → ~61-63k
# paramètres) est en danger de surapprentissage, particulièrement
# aux horizons J+7 et J+30 où le nombre de séquences effectivement
# indépendantes est plus faible et le signal macro plus diffus
# (observé empiriquement : écart LSTM vs ARIMA qui se creuse
# fortement aux horizons longs). La capacité du modèle est donc
# réduite progressivement avec l'horizon :
#   - J+1  : capacité complète (signal court terme à exploiter)
#   - J+7  : capacité réduite (~6x moins de paramètres)
#   - J+30 : capacité minimale (~9x moins de paramètres)
# combinée à un weight_decay (régularisation L2) plus fort aux
# horizons longs.
LSTM_CONFIG_BY_HORIZON = {
    1:  {"hidden_size": 64, "num_layers": 2, "dropout": 0.20, "weight_decay": 1e-5},
    7:  {"hidden_size": 32, "num_layers": 1, "dropout": 0.30, "weight_decay": 1e-4},
    30: {"hidden_size": 24, "num_layers": 1, "dropout": 0.35, "weight_decay": 1e-4},
}


def get_lstm_config(horizon: int) -> dict:
    """
    Retourne la configuration LSTM complète pour un horizon donné
    (fusion de LSTM_CONFIG_BASE + overrides spécifiques à l'horizon).
    Si l'horizon n'a pas d'entrée dédiée dans
    LSTM_CONFIG_BY_HORIZON, retombe sur la configuration J+1
    (capacité complète) par sécurité.
    """
    cfg = dict(LSTM_CONFIG_BASE)
    cfg.update(LSTM_CONFIG_BY_HORIZON.get(
        horizon, LSTM_CONFIG_BY_HORIZON[1]
    ))
    return cfg


LAG_FEATURES = [1, 2, 3, 5, 10, 20]  # lags en jours ouvrés