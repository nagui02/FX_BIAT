# =============================================================
#  src/s2_arima_garch/config.py
#  Configuration centralisée S2 — ARIMA + ARIMAX + GARCH
# =============================================================

import os

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "..", "data", "processed")
FIGURES_DIR   = os.path.join(BASE_DIR, "..", "..", "report", "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Fenêtres temporelles — fenêtre finale retenue : exclut le régime
# quasi-fixe 2004-2011 (structurellement différent), inclut la
# transition 2012-2015 pour maximiser la taille d'échantillon.
TRAIN_START = "2012-01-01"   # était "2004-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2026-07-31"

# Horizons de prévision.
# Recommandation : garder les 4 horizons pour couvrir le spectre
# court terme (efficience de marché) / moyen terme / long terme
# (canaux macro-économiques plus lents).
HORIZONS = [1 , 7, 30]

REFIT_BY_HORIZON = {1: 1, 7: 2, 30: 7}

DATASET_PATH = os.path.join(PROCESSED_DIR, "dataset_final.csv")

# NOTE : la sélection des variables exogènes pour ARIMAX ne passe
# plus par une liste statique ici — voir get_exog_features()
# dans arimax.py, qui sélectionne les variables par devise ET
# par horizon (une ancienne liste EXOG_FEATURES statique et
# jamais utilisée a été supprimée de ce fichier).