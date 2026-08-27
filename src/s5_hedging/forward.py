# =============================================================
#  src/s5_hedging/forward.py
#  Pricing Forward — Parité des taux d'intérêt couverts (CIP)
#  Auteur : Youssef Neji | MINDS ENIT | 2026
# =============================================================

import pandas as pd

from .config import (
    PROCESSED_DIR, HEDGE_HORIZONS, DAY_COUNT,
    DOMESTIC_RATE_COL, FOREIGN_RATE_COL, CLIENT_PROFILES
)


def load_latest_rates() -> dict:
    """
    Dernières valeurs disponibles : cours spot TND/USD, TND/EUR,
    taux TND (BCT), taux Fed (USD), taux BCE (EUR).
    """
    df = pd.read_csv(
        f"{PROCESSED_DIR}/dataset_final.csv",
        parse_dates=["Date"], index_col="Date"
    )
    last = df.iloc[-1]
    return {
        "date":     df.index[-1].strftime("%Y-%m-%d"),
        "spot_usd": float(last["TND_USD"]),
        "spot_eur": float(last["TND_EUR"]),
        "r_tnd":    float(last[DOMESTIC_RATE_COL]) / 100,
        "r_usd":    float(last[FOREIGN_RATE_COL["USD"]]) / 100,
        "r_eur":    float(last[FOREIGN_RATE_COL["EUR"]]) / 100,
    }


def forward_price(spot: float, r_domestic: float, r_foreign: float,
                  days: int, day_count: int = DAY_COUNT) -> float:
    """
    Prix Forward par parité des taux d'intérêt couverts (CIP) :

        F = S x (1 + r_TND x T/360) / (1 + r_FCY x T/360)

    Intuition : le point Forward reflète le différentiel de taux
    entre la Tunisie et l'étranger. Si r_TND > r_FCY (cas usuel,
    la Tunisie a des taux plus élevés), le TND se traite avec une
    décote à terme — le Forward TND/devise est PLUS ÉLEVÉ que
    le spot, ce qui pénalise structurellement l'importateur qui
    fixe son cours d'achat en avance.
    """
    T = days / day_count
    return spot * (1 + r_domestic * T) / (1 + r_foreign * T)


def forward_points(spot: float, forward: float) -> dict:
    """Décomposition du prix Forward en points de terme."""
    points = forward - spot
    return {
        "points_absolus": round(points, 5),
        "points_pips":    round(points * 10000, 1),
        "points_pct":     round(points / spot * 100, 4),
        "sens":           "report (Forward > Spot)" if points > 0
                          else "déport (Forward < Spot)",
    }


def price_forward_for_profile(profile_name: str,
                              rates: dict = None) -> pd.DataFrame:
    """
    Prix Forward pour un profil donné, sur tous les horizons.
    Le coût de couverture est la différence entre le montant
    payé/reçu au Forward vs au spot actuel.
    """
    if rates is None:
        rates = load_latest_rates()

    profile = CLIENT_PROFILES[profile_name]
    if profile["currency"] == "MULTI":
        return pd.DataFrame()  # Desk BIAT traité séparément (portefeuille)

    currency, amount, direction = (
        profile["currency"], profile["amount"], profile["direction"]
    )
    spot   = rates[f"spot_{currency.lower()}"]
    r_fcy  = rates[f"r_{currency.lower()}"]
    r_tnd  = rates["r_tnd"]

    rows = []
    for days in HEDGE_HORIZONS:
        F = forward_price(spot, r_tnd, r_fcy, days)
        pts = forward_points(spot, F)

        # Coût de couverture : différence entre le montant fixé
        # au Forward et le montant qu'on aurait au spot actuel,
        # dans le sens défavorable au profil (direction).
        cout_vs_spot = direction * (F - spot) * amount

        rows.append({
            "profile":       profile_name,
            "devise":        currency,
            "horizon_j":     days,
            "spot":          round(spot, 5),
            "forward":       round(F, 5),
            **pts,
            "cout_couverture_tnd": round(cout_vs_spot, 2),
            "cours_fige":    round(F, 5),
        })

    return pd.DataFrame(rows)


def run_all_forwards() -> pd.DataFrame:
    rates = load_latest_rates()
    frames = [
        price_forward_for_profile(name, rates)
        for name in CLIENT_PROFILES
        if CLIENT_PROFILES[name]["currency"] != "MULTI"
    ]
    return pd.concat(frames, ignore_index=True)