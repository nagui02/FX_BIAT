# =============================================================
#  src/s4_risk/config.py
#  Configuration du module Risque de Change (S4)
#  Auteur : Youssef Neji | MINDS ENIT | 2026
# =============================================================

import os

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "..", "data", "processed")
LIVE_DIR      = os.path.join(BASE_DIR, "..", "..", "data", "live")

# ── Niveaux de confiance VaR (standard Bâle) ────────────────
from scipy.stats import norm

CONFIDENCE_LEVELS = [0.95, 0.99]
Z_SCORES = {a: float(norm.ppf(a)) for a in CONFIDENCE_LEVELS}
# 0.95 -> 1.6448536..., 0.99 -> 2.3263479...
# (remplace les valeurs arrondies 1.645/2.326 codées en dur)

# ── Fenêtre VaR historique (≈ 1 an de données récentes) ─────
HIST_WINDOW = 252

# ── Monte Carlo ──────────────────────────────────────────────
MC_SIMULATIONS = 10_000
VAR_HORIZON    = 1  # horizon en jours

# ── Stress testing : magnitudes des chocs adverses ───────────
STRESS_SHOCKS = [0.10, 0.20, 0.30]

# ── Scénarios historiques (dates approximatives) ─────────────
HISTORICAL_SCENARIOS = {
    "Dévaluation 2018":  ("2018-01-01", "2018-12-31"),
    "COVID-19":          ("2020-03-01", "2020-06-30"),
    "Guerre Ukraine":    ("2022-02-24", "2022-06-30"),
}

# ── Profils clients simulés ───────────────────────────────────
# direction : +1 = perd si TND se déprécie (cours TND/FCY augmente)
#             -1 = perd si TND s'apprécie (cours TND/FCY diminue)
CLIENT_PROFILES = {
    "ALPHA SARL": {
        "currency":    "EUR",
        "amount":      1_000_000,
        "role":        "Importateur",
        "direction":   +1,
        "description": "Achat d'équipements industriels en zone euro",
    },
    "BETA Export": {
        "currency":    "USD",
        "amount":      500_000,
        "role":        "Exportateur",
        "direction":   -1,
        "description": "Ventes facturées en dollars US",
    },
    "M. GAMMA": {
        "currency":    "EUR",
        "amount":      50_000,
        "role":        "Particulier",
        "direction":   +1,
        "description": "Frais de scolarité à l'étranger",
    },
    "Desk BIAT": {
        "currency":    "MULTI",
        "amount":      8_000_000,
        "split":       {"EUR": 0.5, "USD": 0.5},
        "role":        "Position propre",
        "direction":   +1,
        "description": "Position longue nette multi-devises",
    },
    "Dette BIAT": {
        "currency":    "USD",
        "amount":      10_000_000,
        "role":        "Emprunteur",
        "direction":   +1,
        "description": "Dette externe à rembourser",
    },
}