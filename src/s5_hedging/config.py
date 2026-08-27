# =============================================================
#  src/s5_hedging/config.py
#  Configuration du module Couverture (S5)
#  Auteur : Youssef Neji | MINDS ENIT | 2026
# =============================================================

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(BASE_DIR, "..", "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.s4_risk.config import CLIENT_PROFILES, PROCESSED_DIR, LIVE_DIR
# ── Horizons de couverture ────────────────────────────────────
HEDGE_HORIZONS = [7, 30]  # jours calendaires

# ── Convention de jours pour l'actualisation ──────────────────
DAY_COUNT = 360  # convention monétaire standard (Act/360)

# ── Mapping direction S4 → type d'option ──────────────────────
#  direction=+1 (perd si TND se déprécie) → besoin d'un CALL
#  direction=-1 (perd si TND s'apprécie)  → besoin d'un PUT
DIRECTION_TO_OPTION = {+1: "call", -1: "put"}

# ── Taux domestique (TND) et étrangers pour le pricing ─────────
DOMESTIC_RATE_COL = "TauxMMBCT"
FOREIGN_RATE_COL  = {"EUR": "TauxBCE_EUR", "USD": "TauxFed_USD"}