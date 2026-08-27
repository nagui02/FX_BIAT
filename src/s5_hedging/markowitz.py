# =============================================================
#  src/s5_hedging/markowitz.py
#  Optimisation de portefeuille EUR+USD — Markowitz
#  Auteur : Youssef Neji | MINDS ENIT | 2026
# =============================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import PROCESSED_DIR, CLIENT_PROFILES

from ..s2_arima_garch.config import TEST_START, TEST_END
TRADING_DAYS = 252


# Ajout dans src/s5_hedging/markowitz.py — remplace load_portfolio_data

def load_portfolio_data(include_carry: bool = True) -> dict:
    """
    Charge les rendements pour Markowitz.

    include_carry=True : rendement total = rendement de change
    + taux d'intérêt de la devise détenue (carry), cohérent avec
    la logique de parité des taux déjà utilisée en S5 pour le
    pricing Forward. C'est le vrai rendement économique de
    détenir un dépôt en devise étrangère, pas juste sa variation
    de change.

    include_carry=False : ancienne version, change seul
    (conservée pour comparaison directe avant/après correction).
    """
    df = pd.read_csv(
        f"{PROCESSED_DIR}/dataset_final.csv",
        parse_dates=["Date"], index_col="Date"
    )
    window = df.loc[TEST_START:TEST_END]

    r_eur = window["LogRet_TND_EUR"].dropna()
    r_usd = window["LogRet_TND_USD"].dropna()
    common = r_eur.index.intersection(r_usd.index)
    r_eur, r_usd = r_eur.loc[common], r_usd.loc[common]

    returns = pd.DataFrame({"EUR": r_eur, "USD": r_usd})
    mu_change = returns.mean().values * TRADING_DAYS
    cov = returns.cov().values * TRADING_DAYS

    r_eur_rate = float(window["TauxBCE_EUR"].iloc[-1]) / 100
    r_usd_rate = float(window["TauxFed_USD"].iloc[-1]) / 100
    rf = float(window["TauxMMBCT"].iloc[-1]) / 100

    if include_carry:
        mu = mu_change + np.array([r_eur_rate, r_usd_rate])
    else:
        mu = mu_change

    return {
        "returns": returns, "mu": mu, "cov": cov, "rf": rf,
        "mu_change_only": mu_change,
        "carry_rates": {"EUR": r_eur_rate, "USD": r_usd_rate},
        "n_obs": len(returns), "assets": ["EUR", "USD"],
        "period": f"{TEST_START} → {TEST_END}",
        "include_carry": include_carry,
    }


# =============================================================
#  MÉTRIQUES DE PORTEFEUILLE
# =============================================================

def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    return float(weights @ mu)


def portfolio_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov @ weights))


def sharpe_ratio(weights: np.ndarray, mu: np.ndarray,
                 cov: np.ndarray, rf: float) -> float:
    ret = portfolio_return(weights, mu)
    vol = portfolio_vol(weights, cov)
    return (ret - rf) / vol if vol > 0 else 0.0


# =============================================================
#  OPTIMISATION — poids long-only, somme = 1
# =============================================================

def minimize_variance(mu: np.ndarray, cov: np.ndarray,
                      target_return: float = None) -> dict:
    """
    Portefeuille à variance minimale, éventuellement sous
    contrainte de rendement cible (pour tracer la frontière).
    Contraintes : poids ∈ [0,1], somme = 1 (long-only, pas de levier).
    """
    n = len(mu)
    x0 = np.ones(n) / n
    bounds = [(0, 1)] * n

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if target_return is not None:
        constraints.append({
            "type": "eq",
            "fun": lambda w: portfolio_return(w, mu) - target_return
        })

    result = minimize(
        lambda w: portfolio_vol(w, cov), x0,
        method="SLSQP", bounds=bounds, constraints=constraints
    )
    w = result.x
    return {
        "weights": w,
        "return": portfolio_return(w, mu),
        "vol": portfolio_vol(w, cov),
        "success": result.success,
    }


def maximize_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float) -> dict:
    """Portefeuille tangent — ratio de Sharpe maximal."""
    n = len(mu)
    x0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(
        lambda w: -sharpe_ratio(w, mu, cov, rf), x0,
        method="SLSQP", bounds=bounds, constraints=constraints
    )
    w = result.x
    return {
        "weights": w,
        "return": portfolio_return(w, mu),
        "vol": portfolio_vol(w, cov),
        "sharpe": sharpe_ratio(w, mu, cov, rf),
        "success": result.success,
    }


# =============================================================
#  FRONTIÈRE EFFICIENTE
# =============================================================

def efficient_frontier(mu: np.ndarray, cov: np.ndarray,
                       n_points: int = 50) -> pd.DataFrame:
    """
    Trace la frontière efficiente : pour chaque niveau de
    rendement cible entre le min et le max atteignable, trouve
    le portefeuille de variance minimale correspondant.
    """
    min_ret, max_ret = mu.min(), mu.max()
    targets = np.linspace(min_ret, max_ret, n_points)

    rows = []
    for target in targets:
        res = minimize_variance(mu, cov, target_return=target)
        if res["success"]:
            rows.append({
                "rendement": res["return"],
                "volatilite": res["vol"],
                "poids_EUR": res["weights"][0],
                "poids_USD": res["weights"][1],
            })
    return pd.DataFrame(rows)

def capital_market_line_analysis(data: dict, max_sharpe_res: dict) -> dict:
    """
    Formalise pourquoi aucune combinaison EUR/USD long-only ne bat
    la position 100% TND — construit la Ligne de Marché des
    Capitaux (CML) entre l'actif sans risque (TND, rf, vol=0) et
    le portefeuille tangent EUR/USD atteignable.

    Interprétation : la pente de la CML = le "prix du risque"
    (rendement additionnel obtenu par unité de volatilité prise).
    Une pente NÉGATIVE signifie que prendre du risque FX ne
    rapporte rien de plus que de garder du TND pur — au contraire,
    ça coûte du rendement pour rien.
    """
    rf = data["rf"]
    tangent_ret = max_sharpe_res["return"]
    tangent_vol = max_sharpe_res["vol"]

    cml_slope = (tangent_ret - rf) / tangent_vol if tangent_vol > 0 else 0

    # Vérification : est-ce qu'un SEUL point de la frontière FX
    # atteint ou dépasse le taux TND ?
    max_fx_return = max(data["mu"])
    dominance_totale = max_fx_return < rf

    return {
        "taux_TND": rf,
        "rendement_tangent_FX": tangent_ret,
        "volatilite_tangent_FX": tangent_vol,
        "pente_CML": cml_slope,
        "rendement_FX_max_possible": max_fx_return,
        "TND_domine_toute_la_frontiere": dominance_totale,
        "ecart_pts": (rf - max_fx_return) * 100,
    }


def print_cml_conclusion(cml: dict):
    print("\n" + "=" * 70)
    print("  CAPITAL MARKET LINE — TND vs Frontière EUR/USD")
    print("=" * 70)
    print(f"Taux TND (actif sans risque)      : {cml['taux_TND']*100:.2f}%")
    print(f"Meilleur rendement FX atteignable  : "
          f"{cml['rendement_FX_max_possible']*100:.2f}% "
          f"(100% EUR, sans risque de change diversifié)")
    print(f"Écart                              : "
          f"{cml['ecart_pts']:.2f} points de %")
    print(f"Pente de la CML                    : {cml['pente_CML']:.4f} "
          f"({'négative' if cml['pente_CML']<0 else 'positive'})")

    if cml["TND_domine_toute_la_frontiere"]:
        print(f"\n✅ CONFIRMÉ : le taux TND domine INTÉGRALEMENT la "
              f"frontière EUR/USD sur 2024-2026.")
        print(f"   Aucune combinaison de devises, quelle qu'elle soit, "
              f"n'égale le rendement du cash TND à ce niveau de risque.")
        print(f"   → Cohérent avec la politique de taux élevés de la "
              f"BCT visant à défendre le TND.")
# =============================================================
#  CAS D'ÉTUDE — Desk BIAT (position actuelle 50/50)
# =============================================================

def analyze_desk_biat() -> dict:
    """
    Compare la position actuelle de Desk BIAT (50/50 EUR/USD,
    8M TND, définie en S4) au portefeuille optimal Markowitz —
    minimum variance ET Sharpe maximal.
    """
    data = load_portfolio_data()
    mu, cov, rf = data["mu"], data["cov"], data["rf"]
    profile = CLIENT_PROFILES["Desk BIAT"]
    total = profile["amount"]

    # Position actuelle
    w_current = np.array([
        profile["split"]["EUR"], profile["split"]["USD"]
    ])
    current = {
        "weights": w_current,
        "return": portfolio_return(w_current, mu),
        "vol": portfolio_vol(w_current, cov),
        "sharpe": sharpe_ratio(w_current, mu, cov, rf),
    }

    # Optimaux
    min_var = minimize_variance(mu, cov)
    min_var["sharpe"] = sharpe_ratio(min_var["weights"], mu, cov, rf)
    max_sharpe = maximize_sharpe(mu, cov, rf)

    frontier = efficient_frontier(mu, cov)

    return {
        "data": data, "total_amount": total,
        "current": current, "min_variance": min_var,
        "max_sharpe": max_sharpe, "frontier": frontier,
    }


def summary_table(analysis: dict) -> pd.DataFrame:
    total = analysis["total_amount"]
    rows = []
    for label, res in [
        ("Position actuelle (50/50)", analysis["current"]),
        ("Variance minimale", analysis["min_variance"]),
        ("Sharpe maximal (tangent)", analysis["max_sharpe"]),
    ]:
        rows.append({
            "portefeuille": label,
            "poids_EUR_pct": round(res["weights"][0]*100, 1),
            "poids_USD_pct": round(res["weights"][1]*100, 1),
            "montant_EUR_tnd": round(res["weights"][0]*total, 0),
            "montant_USD_tnd": round(res["weights"][1]*total, 0),
            "rendement_ann_pct": round(res["return"]*100, 3),
            "volatilite_ann_pct": round(res["vol"]*100, 3),
            "sharpe_ratio": round(res["sharpe"], 4),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import os
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("=" * 70)
    print("  S5 — OPTIMISATION MARKOWITZ : DESK BIAT EUR/USD")
    print("=" * 70)

    analysis = analyze_desk_biat()
    data = analysis["data"]

    print(f"\nDonnées : {data['n_obs']} obs | {data['period']}")
    print(f"Rendement annualisé : EUR={data['mu'][0]*100:.3f}% | "
          f"USD={data['mu'][1]*100:.3f}%")
    print(f"Volatilité annualisée : EUR={np.sqrt(data['cov'][0,0])*100:.2f}% | "
          f"USD={np.sqrt(data['cov'][1,1])*100:.2f}%")
    corr = data["cov"][0,1] / np.sqrt(data["cov"][0,0]*data["cov"][1,1])
    print(f"Corrélation EUR/USD : {corr:.4f}")
    print(f"Taux sans risque (BCT) : {data['rf']*100:.2f}%")

    print("\n📊 Comparaison des portefeuilles :")
    summary = summary_table(analysis)
    print(summary.to_string(index=False))
    cml = capital_market_line_analysis(
        analysis["data"], analysis["max_sharpe"]
    )
    print_cml_conclusion(cml)

    cml_path = os.path.join(PROCESSED_DIR, "markowitz_cml.csv")
    pd.DataFrame([cml]).to_csv(cml_path, index=False)
    print(f"\n💾 {cml_path}")
    
    summary_path = os.path.join(PROCESSED_DIR, "markowitz_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n💾 {summary_path}")

    frontier_path = os.path.join(PROCESSED_DIR, "markowitz_frontier.csv")
    analysis["frontier"].to_csv(frontier_path, index=False)
    print(f"💾 {frontier_path}")
    print("\n" + "=" * 70)
    print("  COMPARAISON : SANS vs AVEC CARRY")
    print("=" * 70)
    d = analysis["data"]
    print(f"Rendement change seul  : EUR={d['mu_change_only'][0]*100:+.3f}% "
          f"| USD={d['mu_change_only'][1]*100:+.3f}%")
    print(f"Taux devise (carry)    : EUR={d['carry_rates']['EUR']*100:.2f}% "
          f"| USD={d['carry_rates']['USD']*100:.2f}%")
    print(f"Rendement total (avec carry) : "
          f"EUR={d['mu'][0]*100:+.3f}% | USD={d['mu'][1]*100:+.3f}%")
    print(f"Taux sans risque TND   : {d['rf']*100:.2f}%")
    print(f"\nSharpe 50/50 : {analysis['current']['sharpe']:.4f}")
    print(f"Sharpe min-var : {analysis['min_variance']['sharpe']:.4f}")
    print(f"Sharpe max-Sharpe : {analysis['max_sharpe']['sharpe']:.4f}")
    print(f"Poids max-Sharpe : EUR={analysis['max_sharpe']['weights'][0]*100:.1f}% "
          f"USD={analysis['max_sharpe']['weights'][1]*100:.1f}%")
    
    print("\n✅ Markowitz terminé")