# =============================================================
#  src/s4_risk/bct_diagnostics.py
#  Diagnostic : intensité d'intervention BCT vs persistance GARCH
#  Auteur : Youssef Neji | MINDS ENIT | 2026
#
#  Objectif : vérifier si bct_market_share_pct (part de la BCT
#  dans les transactions au comptant totales) aide à expliquer la
#  persistance GARCH élevée (α+β≈0.996) observée post-Ukraine —
#  hypothèse : une intervention active de la BCT amortit
#  artificiellement la variance réalisée, ce qui peut se traduire
#  par une persistance élevée dans un GARCH(1,1) univarié qui ne
#  "voit" pas cette intervention.
#
#  ⚠️  Ce script est un DIAGNOSTIC (corrélation), pas une
#  réestimation du GARCH. Si la corrélation s'avère forte et
#  significative, la suite logique serait un GARCH-X (variance
#  exogène) — décision méthodologique à valider séparément,
#  comme cela a été fait pour ARIMA vs ARIMAX (DM test), plutôt
#  qu'à ajouter silencieusement ici.
#
#  Usage : python -m src.s4_risk.bct_diagnostics
# =============================================================

import os
import pandas as pd
import numpy as np

from .config import PROCESSED_DIR
from .var import load_sigma


def load_bct_market_share() -> pd.Series:
    """
    Charge bct_market_share_pct depuis dataset_final.csv (déjà
    fusionné avec décalage de publication anti look-ahead bias
    dans data.py::load_bct_extra).
    """
    path = os.path.join(PROCESSED_DIR, "dataset_final.csv")
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    if "bct_market_share_pct" not in df.columns:
        raise ValueError(
            "Colonne 'bct_market_share_pct' absente de dataset_final.csv "
            "— vérifier que data.py a bien été relancé après l'ajout "
            "des fichiers BCT complémentaires."
        )
    return df["bct_market_share_pct"].dropna()


def correlation_share_vs_sigma(currency: str,
                               period: tuple | None = None) -> dict:
    """
    Corrélation entre bct_market_share_pct et la volatilité
    conditionnelle GARCH sigma_t, sur une période donnée (par
    défaut : toute la période disponible en commun).
    """
    share = load_bct_market_share()
    sigma = load_sigma(currency)

    common = share.index.intersection(sigma.index)
    s, sg = share.loc[common], sigma.loc[common]

    if period is not None:
        start, end = period
        s, sg = s.loc[start:end], sg.loc[start:end]

    if len(s) < 10:
        return {
            "currency": f"TND/{currency}",
            "period": period if period else "toute la période",
            "n_obs": len(s),
            "correlation_bct_share_vs_sigma": None,
        }

    corr = s.corr(sg)
    return {
        "currency": f"TND/{currency}",
        "period": period if period else "toute la période",
        "n_obs": len(s),
        "correlation_bct_share_vs_sigma": round(corr, 4),
    }


def run_diagnostics() -> pd.DataFrame:
    """
    Corrélation bct_market_share_pct vs sigma GARCH, pour USD et
    EUR, sur toute la période ET restreinte à la fenêtre post-Ukraine
    (2022-02-24 → aujourd'hui), où la persistance élevée a été
    observée — cf. HISTORICAL_SCENARIOS["Guerre Ukraine"] dans
    s4_risk/config.py pour la date de départ du choc.
    """
    results = []
    for currency in ["USD", "EUR"]:
        results.append(correlation_share_vs_sigma(currency))
        results.append(correlation_share_vs_sigma(
            currency, period=("2022-02-24", None)
        ))
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=" * 60)
    print("  DIAGNOSTIC — BCT market share vs persistance GARCH")
    print("=" * 60)

    df = run_diagnostics()
    print(df.to_string(index=False))

    out = os.path.join(PROCESSED_DIR, "bct_diagnostics.csv")
    df.to_csv(out, index=False)
    print(f"\n💾 {out}")