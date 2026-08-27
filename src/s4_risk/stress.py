## =============================================================
#  src/s4_risk/stress.py
#  Stress testing, scénarios historiques, backtesting Kupiec
#  Auteur : Youssef Neji | MINDS ENIT | 2026
# =============================================================

import numpy as np
import pandas as pd
from scipy.stats import chi2, t as student_t, kurtosis as scipy_kurtosis

from .config import (
    PROCESSED_DIR, STRESS_SHOCKS, HISTORICAL_SCENARIOS,
    CLIENT_PROFILES, CONFIDENCE_LEVELS, Z_SCORES, HIST_WINDOW
)
from .var import load_returns, load_sigma
from ..s2_arima_garch.config import TEST_START, TEST_END





# =============================================================
#  STRESS TESTING — chocs synthétiques
# =============================================================

def stress_shock(profile_name: str, shock_magnitude: float) -> dict:
    """
    Choc de shock_magnitude (ex: 0.20 = 20%) appliqué dans le sens
    défavorable au profil. Exposition linéaire (spot, sans
    optionalité) → perte directement proportionnelle au choc.
    """
    profile = CLIENT_PROFILES[profile_name]
    amount, direction = profile["amount"], profile["direction"]
    sens = "hausse" if direction == 1 else "baisse"

    loss_value = shock_magnitude * amount

    return {
        "profile": profile_name,
        "choc_pct": f"{shock_magnitude*100:.0f}%",
        "sens_du_cours": f"{sens} de {shock_magnitude*100:.0f}% "
                         f"du cours TND/{profile['currency']}",
        "perte_estimee": round(loss_value, 2),
        "perte_pct_exposition": round(shock_magnitude*100, 2),
    }


def run_stress_grid() -> pd.DataFrame:
    results = []
    for name, profile in CLIENT_PROFILES.items():
        if profile["currency"] == "MULTI":
            continue
        for shock in STRESS_SHOCKS:
            results.append(stress_shock(name, shock))
    return pd.DataFrame(results)


# =============================================================
#  SCÉNARIOS HISTORIQUES RÉELS
# =============================================================

def historical_scenario_impact(profile_name: str,
                               scenario_name: str) -> dict | None:
    """Impact réel d'un choc de marché passé sur un profil donné."""
    profile = CLIENT_PROFILES[profile_name]
    if profile["currency"] == "MULTI":
        return None

    currency, amount, direction = (profile["currency"],
                                   profile["amount"], profile["direction"])
    start, end = HISTORICAL_SCENARIOS[scenario_name]

    path = f"{PROCESSED_DIR}/dataset_final.csv"
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    window = df.loc[start:end, f"TND_{currency}"].dropna()
    if len(window) < 2:
        return None

    total_change  = (window.iloc[-1]-window.iloc[0]) / window.iloc[0]
    adverse_change = direction * total_change
    loss_pct   = max(adverse_change, 0)
    loss_value = loss_pct * amount

    return {
        "profile": profile_name, "scenario": scenario_name,
        "periode": f"{start} → {end}",
        "variation_cours_pct": round(total_change*100, 2),
        "perte_estimee": round(loss_value, 2),
        "perte_pct_exposition": round(loss_pct*100, 2),
    }


def run_historical_scenarios() -> pd.DataFrame:
    results = []
    for name in CLIENT_PROFILES:
        for scenario in HISTORICAL_SCENARIOS:
            r = historical_scenario_impact(name, scenario)
            if r: results.append(r)
    return pd.DataFrame(results)

def kupiec_test(losses: pd.Series, var_series, alpha: float) -> dict:
    """
    Test de Kupiec (1995).

    FIX : aux bornes (0 ou n violations), la formule originale
    forçait LR=0 — ce qui peut MASQUER un modèle qui devrait
    échouer le backtest. La vraie limite (0*ln(0) -> 0 par
    continuité) donne une forme fermée non-nulle, pas 0.
    """
    violations = (losses > var_series).sum()
    n = len(losses)
    p_theo = 1 - alpha
    p_hat  = violations / n if n > 0 else 0

    if n == 0:
        lr_stat = 0.0
    elif violations == 0:
        lr_stat = -2 * n * np.log(1 - p_theo)
    elif violations == n:
        lr_stat = -2 * n * np.log(p_theo)
    else:
        lr_stat = -2 * (
            (n-violations)*np.log(1-p_theo) + violations*np.log(p_theo)
            - (n-violations)*np.log(1-p_hat) - violations*np.log(p_hat)
        )
    critical = chi2.ppf(0.95, df=1)

    return {
        "n_obs": n, "n_violations": int(violations),
        "violations_attendues": round(n*p_theo, 1),
        "taux_violation_pct": round(p_hat*100, 3),
        "taux_theorique_pct": round(p_theo*100, 3),
        "LR_stat": round(lr_stat, 4),
        "seuil_critique_5pct": round(critical, 4),
        "modele_valide": lr_stat <= critical,
    }
# =============================================================
#  Backtesting — Kupiec, 4 méthodes, restreint à 2024-2026
# =============================================================



def historical_var_daily(returns: pd.Series, alpha: float,
                         window: int = HIST_WINDOW) -> pd.Series:
    """
    VaR historique glissante : recalculée chaque jour à partir
    des `window` jours PRÉCÉDENTS (shift(1) évite tout lookahead).
    La fenêtre peut légitimement remonter avant 2024 — c'est la
    méthodologie standard, pas un biais.
    """
    return returns.rolling(window).quantile(alpha).shift(1)


def parametric_var_daily(sigma: pd.Series, alpha: float) -> pd.Series:
    """VaR paramétrique : z x sigma_t (GARCH, non-lookahead par construction)."""
    return Z_SCORES[alpha] * sigma


def mc_normal_var_daily(returns: pd.Series, sigma: pd.Series,
                        alpha: float, window: int = HIST_WINDOW,
                        n_sim: int = 20000, seed: int = 42) -> pd.Series:
    """
    VaR Monte Carlo gaussienne, recalculée chaque jour.
    Mathématiquement équivalente à la VaR paramétrique pour n_sim
    grand — la comparer permet de vérifier la convergence numérique
    plutôt que d'apporter une méthode réellement différente.
    """
    rng = np.random.default_rng(seed)
    mu_roll = returns.rolling(window).mean().shift(1)
    common = mu_roll.index.intersection(sigma.index)
    result = {}
    for d in common:
        mu_t, sig_t = mu_roll.loc[d], sigma.loc[d]
        if pd.isna(mu_t) or pd.isna(sig_t):
            continue
        sim = rng.normal(mu_t, sig_t, n_sim)
        result[d] = np.quantile(sim, alpha)
    return pd.Series(result)


def student_t_var_daily(returns: pd.Series, sigma: pd.Series,
                        alpha: float, window: int = HIST_WINDOW) -> pd.Series:
    """
    VaR Student-t en forme fermée (pas de simulation — plus rapide
    et exact) : quantile de Student-t standardisé x sigma_t + drift.
    nu calibré une fois sur la kurtosis de l'historique complet.
    """
    excess_kurt = scipy_kurtosis(returns.dropna(), fisher=True)
    nu = 30.0 if excess_kurt <= 0.1 else float(
        np.clip(4 + 6/excess_kurt, 4.5, 30.0)
    )
    q_std = student_t.ppf(alpha, df=nu) * np.sqrt((nu-2)/nu)

    mu_roll = returns.rolling(window).mean().shift(1)
    common = mu_roll.index.intersection(sigma.index)
    return pd.Series({
        d: mu_roll.loc[d] + sigma.loc[d]*q_std
        for d in common
        if not pd.isna(mu_roll.loc[d]) and not pd.isna(sigma.loc[d])
    })


def run_backtesting() -> pd.DataFrame:
    """
    Backtest de Kupiec pour les 4 méthodes de VaR, restreint
    strictement à la période de test 2024-2026 (hors échantillon
    par rapport à l'estimation GARCH sur 2004-2023).
    """
    results = []
    methods = {
        "Historique":            historical_var_daily,
        "Paramétrique (GARCH)":  parametric_var_daily,
        "Monte Carlo":           mc_normal_var_daily,
        "Monte Carlo Student-t": student_t_var_daily,
    }

    for currency in ["USD", "EUR"]:
        returns_full = load_returns(currency)
        sigma_full   = load_sigma(currency)

        for alpha in CONFIDENCE_LEVELS:
            for method_name, fn in methods.items():
                if method_name == "Paramétrique (GARCH)":
                    var_series = fn(sigma_full, alpha)
                else:
                    var_series = fn(returns_full, sigma_full, alpha) \
                        if method_name != "Historique" else \
                        fn(returns_full, alpha)

                # Restriction stricte à la période test
                var_series = var_series.loc[TEST_START:TEST_END].dropna()
                losses = returns_full.loc[var_series.index]

                if len(losses) < 30:
                    continue

                k = kupiec_test(losses, var_series, alpha)
                k["currency"]   = f"TND/{currency}"
                k["confidence"] = alpha
                k["method"]     = method_name
                k["periode_test"] = f"{TEST_START} → {TEST_END}"
                results.append(k)

    return pd.DataFrame(results)