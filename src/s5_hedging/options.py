# =============================================================
#  src/s5_hedging/options.py
#  Pricing Options de change — Garman-Kohlhagen (1983)
#  Auteur : Youssef Neji | MINDS ENIT | 2026
# =============================================================

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import (
    PROCESSED_DIR, HEDGE_HORIZONS, DAY_COUNT,
    DIRECTION_TO_OPTION, CLIENT_PROFILES
)
from .forward import load_latest_rates

_sigma_ann_cache: dict = {}
def load_sigma_annualized(currency: str) -> float:
    if currency not in _sigma_ann_cache:
        sigma = pd.read_csv(
            f"{PROCESSED_DIR}/garch_sigma_{currency.lower()}.csv",
            index_col=0, parse_dates=True
        ).squeeze()
        _sigma_ann_cache[currency] = float(sigma.iloc[-1]) * np.sqrt(252)
    return _sigma_ann_cache[currency]


def garman_kohlhagen(S: float, K: float, T: float,
                     r_d: float, r_f: float, sigma: float,
                     option_type: str = "call") -> dict:
    """
    Modèle de Garman-Kohlhagen (1983) — extension de Black-Scholes
    au marché des changes, où les deux devises portent un taux
    d'intérêt (domestique r_d et étranger r_f).

        d1 = [ln(S/K) + (r_d - r_f + sigma²/2)T] / (sigma·√T)
        d2 = d1 - sigma·√T

        Call = S·e^(-r_f·T)·N(d1) - K·e^(-r_d·T)·N(d2)
        Put  = K·e^(-r_d·T)·N(-d2) - S·e^(-r_f·T)·N(-d1)

    Les Grecques (Delta, Gamma, Vega) sont calculées pour piloter
    une couverture dynamique si nécessaire.
    """
    if sigma <= 0 or T <= 0:
        raise ValueError(
            f"garman_kohlhagen : sigma={sigma} et T={T} doivent être "
            f"strictement positifs (division par zéro sinon)."
        )
    d1 = (np.log(S/K) + (r_d - r_f + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    # ... reste inchangé

    if option_type == "call":
        price = S*np.exp(-r_f*T)*norm.cdf(d1) - K*np.exp(-r_d*T)*norm.cdf(d2)
        delta = np.exp(-r_f*T) * norm.cdf(d1)
    else:  # put
        price = K*np.exp(-r_d*T)*norm.cdf(-d2) - S*np.exp(-r_f*T)*norm.cdf(-d1)
        delta = -np.exp(-r_f*T) * norm.cdf(-d1)

    gamma = np.exp(-r_f*T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * np.exp(-r_f*T) * norm.pdf(d1) * np.sqrt(T) / 100  # / 1% vol

    return {
        "prime":   round(price, 6),
        "delta":   round(delta, 4),
        "gamma":   round(gamma, 6),
        "vega":    round(vega, 6),
        "d1": round(d1, 4), "d2": round(d2, 4),
    }


def price_option_for_profile(profile_name: str,
                             rates: dict = None,
                             strike_mode: str = "atm") -> pd.DataFrame:
    """
    Prime d'option pour un profil, sur tous les horizons.
    strike_mode="atm" : strike = spot actuel (at-the-money) —
    protection dès le premier mouvement défavorable.
    """
    if rates is None:
        rates = load_latest_rates()

    profile = CLIENT_PROFILES[profile_name]
    if profile["currency"] == "MULTI":
        return pd.DataFrame()

    currency, amount, direction = (
        profile["currency"], profile["amount"], profile["direction"]
    )
    option_type = DIRECTION_TO_OPTION[direction]

    S     = rates[f"spot_{currency.lower()}"]
    K     = S  # ATM
    r_fcy = rates[f"r_{currency.lower()}"]
    r_tnd = rates["r_tnd"]
    sigma = load_sigma_annualized(currency)

    rows = []
    for days in HEDGE_HORIZONS:
        T = days / DAY_COUNT
        gk = garman_kohlhagen(S, K, T, r_tnd, r_fcy, sigma, option_type)
        prime_totale = gk["prime"] * amount

        rows.append({
            "profile":     profile_name,
            "devise":      currency,
            "type_option": option_type.upper(),
            "horizon_j":   days,
            "spot":        round(S, 5),
            "strike_atm":  round(K, 5),
            "sigma_ann_pct": round(sigma*100, 2),
            "prime_unitaire": gk["prime"],
            "prime_totale_tnd": round(prime_totale, 2),
            "prime_pct_expo":  round(gk["prime"]/S*100, 3),
            "delta":       gk["delta"],
            "gamma":       gk["gamma"],
            "vega":        gk["vega"],
        })

    return pd.DataFrame(rows)


def run_all_options() -> pd.DataFrame:
    rates = load_latest_rates()
    frames = [
        price_option_for_profile(name, rates)
        for name in CLIENT_PROFILES
        if CLIENT_PROFILES[name]["currency"] != "MULTI"
    ]
    return pd.concat(frames, ignore_index=True)