# =============================================================
#  src/s4_risk/var.py
#  Calcul de la VaR et CVaR — 3 méthodes
#  Auteur : Youssef Neji | MINDS ENIT | 2026
# =============================================================

import os
import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import (
    PROCESSED_DIR, CONFIDENCE_LEVELS, Z_SCORES,
    HIST_WINDOW, MC_SIMULATIONS, VAR_HORIZON, CLIENT_PROFILES
)

# FIX (Vague 4, Finding 4) : cache module-level — évite de relire
# les mêmes CSV depuis le disque à chaque appel (5 profils x 2
# niveaux de confiance appellent load_returns/load_sigma en boucle).
_returns_cache: dict = {}
_sigma_cache: dict = {}


def load_returns(currency: str) -> pd.Series:
    """Log-rendements historiques pour une devise."""
    # FIX (Finding 1) : import os manquant + branche morte
    # `if False` supprimée.
    path = os.path.join(PROCESSED_DIR, "dataset_final.csv")
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df[f"LogRet_TND_{currency}"].dropna()


def load_sigma(currency: str) -> pd.Series:
    """Volatilité conditionnelle GARCH(1,1)."""
    path = f"{PROCESSED_DIR}/garch_sigma_{currency.lower()}.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True).squeeze()


def _get_returns(currency: str) -> pd.Series:
    if currency not in _returns_cache:
        _returns_cache[currency] = load_returns(currency)
    return _returns_cache[currency]


def _get_sigma(currency: str) -> pd.Series:
    if currency not in _sigma_cache:
        _sigma_cache[currency] = load_sigma(currency)
    return _sigma_cache[currency]


# =============================================================
#  MÉTHODE 1 — VaR HISTORIQUE
# =============================================================

def var_historical(returns: pd.Series, position_value: float,
                   direction: int, alpha: float,
                   window: int = HIST_WINDOW) -> dict:
    """
    Quantile empirique des rendements récents — ne suppose
    aucune distribution théorique, capture les vraies queues
    de distribution observées (kurtosis élevé sur le TND).
    """
    recent = returns.tail(window)
    losses = direction * recent

    var_pct   = np.quantile(losses, alpha)
    var_value = max(var_pct, 0) * position_value

    tail = losses[losses >= var_pct]
    cvar_pct   = tail.mean() if len(tail) > 0 else var_pct
    cvar_value = max(cvar_pct, 0) * position_value

    return {
        "method": "Historique", "confidence": alpha,
        "VaR_pct": round(var_pct*100, 4),
        "VaR_value": round(var_value, 2),
        "CVaR_pct": round(cvar_pct*100, 4),
        "CVaR_value": round(cvar_value, 2),
        "n_obs": len(recent),
    }


# =============================================================
#  MÉTHODE 2 — VaR PARAMÉTRIQUE (GARCH)
# =============================================================

def var_parametric(sigma_t: float, position_value: float,
                   alpha: float, direction: int = 1,
                   mu: float = 0.0) -> dict:
    """
    Hypothèse de normalité, volatilité conditionnelle GARCH(1,1).

    FIX (Vague 4, Finding 2+3) : intègre désormais la dérive (mu)
    et la direction du profil — l'ancienne version supposait
    implicitement mu=0, ce qui ignore la tendance de dépréciation
    structurelle du TND. Sans direction, le signe de la dérive
    serait faux pour les profils direction=-1 (exportateurs).
    """
    z = Z_SCORES[alpha]
    var_pct   = direction * mu + z * sigma_t
    var_value = var_pct * position_value

    cvar_pct   = direction * mu + sigma_t * norm.pdf(z) / (1 - alpha)
    cvar_value = cvar_pct * position_value

    return {
        "method": "Paramétrique (GARCH)", "confidence": alpha,
        "VaR_pct": round(var_pct*100, 4),
        "VaR_value": round(var_value, 2),
        "CVaR_pct": round(cvar_pct*100, 4),
        "CVaR_value": round(cvar_value, 2),
        "sigma_t": round(sigma_t*100, 4),
    }


# =============================================================
#  MÉTHODE 3 — VaR MONTE CARLO
# =============================================================

def var_monte_carlo(mu: float, sigma_t: float, position_value: float,
                    direction: int, alpha: float,
                    n_sim: int = MC_SIMULATIONS,
                    horizon: int = VAR_HORIZON,
                    seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_sim)
    simulated_returns = mu * horizon + sigma_t * np.sqrt(horizon) * z
    losses = direction * simulated_returns

    var_pct   = np.quantile(losses, alpha)
    var_value = max(var_pct, 0) * position_value

    tail = losses[losses >= var_pct]
    cvar_pct   = tail.mean() if len(tail) > 0 else var_pct
    cvar_value = max(cvar_pct, 0) * position_value

    return {
        "method": "Monte Carlo", "confidence": alpha,
        "VaR_pct": round(var_pct*100, 4),
        "VaR_value": round(var_value, 2),
        "CVaR_pct": round(cvar_pct*100, 4),
        "CVaR_value": round(cvar_value, 2),
        "n_sim": n_sim,
    }


def var_monte_carlo_student_t(returns: pd.Series, mu: float,
                              sigma_t: float, position_value: float,
                              direction: int, alpha: float,
                              n_sim: int = MC_SIMULATIONS,
                              horizon: int = VAR_HORIZON,
                              seed: int = 42) -> dict:
    from scipy.stats import kurtosis as scipy_kurtosis

    excess_kurt = scipy_kurtosis(returns.dropna(), fisher=True)

    if excess_kurt <= 0.1:
        nu = 30.0
    else:
        nu = 4 + 6 / excess_kurt
        nu = float(np.clip(nu, 4.5, 30.0))

    rng = np.random.default_rng(seed)
    t_raw = rng.standard_t(df=nu, size=n_sim)
    t_std = t_raw * np.sqrt((nu - 2) / nu)

    simulated_returns = mu * horizon + sigma_t * np.sqrt(horizon) * t_std
    losses = direction * simulated_returns

    var_pct   = np.quantile(losses, alpha)
    var_value = max(var_pct, 0) * position_value

    tail = losses[losses >= var_pct]
    cvar_pct   = tail.mean() if len(tail) > 0 else var_pct
    cvar_value = max(cvar_pct, 0) * position_value

    return {
        "method": "Monte Carlo Student-t", "confidence": alpha,
        "VaR_pct": round(var_pct*100, 4),
        "VaR_value": round(var_value, 2),
        "CVaR_pct": round(cvar_pct*100, 4),
        "CVaR_value": round(cvar_value, 2),
        "n_sim": n_sim,
        "degres_liberte_nu": round(nu, 2),
        "kurtosis_excedentaire": round(excess_kurt, 3),
    }


# =============================================================
#  PORTEFEUILLE MULTI-DEVISES (Desk BIAT)
# =============================================================

def portfolio_var_multi_currency(profile_name: str,
                                 alpha: float) -> dict:
    profile = CLIENT_PROFILES[profile_name]
    split, total = profile["split"], profile["amount"]
    z = Z_SCORES[alpha]

    var_components = {}
    for cur, weight in split.items():
        sigma_t = float(_get_sigma(cur).iloc[-1])
        var_components[cur] = z * sigma_t * (total * weight)

    r_eur = _get_returns("EUR").tail(HIST_WINDOW)
    r_usd = _get_returns("USD").tail(HIST_WINDOW)
    common = r_eur.index.intersection(r_usd.index)
    rho = r_eur.loc[common].corr(r_usd.loc[common])

    var_eur, var_usd = var_components.get("EUR",0), var_components.get("USD",0)
    var_undiv = var_eur + var_usd
    var_div = np.sqrt(var_eur**2 + var_usd**2 + 2*rho*var_eur*var_usd)
    benefit = var_undiv - var_div

    return {
        "method": "Paramétrique multi-devises", "confidence": alpha,
        "VaR_EUR": round(var_eur, 2), "VaR_USD": round(var_usd, 2),
        "correlation": round(rho, 4),
        "VaR_non_diversifie": round(var_undiv, 2),
        "VaR_diversifie": round(var_div, 2),
        "benefice_diversification": round(benefit, 2),
        "benefice_pct": round(benefit/var_undiv*100, 2) if var_undiv>0 else 0,
    }


# =============================================================
#  ORCHESTRATION
# =============================================================

def compute_all_var(profile_name: str) -> pd.DataFrame:
    """VaR + CVaR par les 4 méthodes × 2 niveaux, pour un profil."""
    profile = CLIENT_PROFILES[profile_name]
    results = []

    if profile["currency"] == "MULTI":
        for alpha in CONFIDENCE_LEVELS:
            r = portfolio_var_multi_currency(profile_name, alpha)
            r["profile"] = profile_name
            results.append(r)
        return pd.DataFrame(results)

    currency, amount, direction = (profile["currency"],
                                   profile["amount"], profile["direction"])
    returns = _get_returns(currency)
    sigma   = _get_sigma(currency)
    sigma_t = float(sigma.iloc[-1])
    mu      = float(returns.tail(HIST_WINDOW).mean())

    for alpha in CONFIDENCE_LEVELS:
        for r in [
            var_historical(returns, amount, direction, alpha),
            var_parametric(sigma_t, amount, alpha,
                          direction=direction, mu=mu),
            var_monte_carlo(mu, sigma_t, amount, direction, alpha),
            var_monte_carlo_student_t(returns, mu, sigma_t, amount,
                                      direction, alpha),
        ]:
            r["profile"], r["currency"] = profile_name, currency
            results.append(r)

    return pd.DataFrame(results)


def run_all_profiles() -> pd.DataFrame:
    return pd.concat(
        [compute_all_var(name) for name in CLIENT_PROFILES],
        ignore_index=True
    )