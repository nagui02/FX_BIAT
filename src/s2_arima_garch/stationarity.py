# =============================================================
#  src/s2_arima_garch/stationarity.py
#  Tests de stationnarité ADF + KPSS
#
#  Test rapide : python -m src.s2_arima_garch.stationarity
# =============================================================

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from .config import FIGURES_DIR, TRAIN_START, TRAIN_END
import os


def test_stationarity(series: pd.Series, name: str) -> dict:
    """
    Applique les tests ADF et KPSS sur une série temporelle.

    ADF  : H0 = racine unitaire (non-stationnaire). Rejeter H0 (p<0.05) → stationnaire
    KPSS : H0 = stationnaire. Rejeter H0 (p<0.05) → non-stationnaire
    """
    print(f"\n   📊 Tests de stationnarité — {name}")

    adf_result = adfuller(series.dropna(), autolag="AIC")
    adf_stat, adf_pvalue = adf_result[0], adf_result[1]
    adf_conclusion = "Stationnaire ✅" if adf_pvalue < 0.05 else "Non-stationnaire ❌"

    kpss_result = kpss(series.dropna(), regression="c", nlags="auto")
    kpss_stat, kpss_pvalue = kpss_result[0], kpss_result[1]
    kpss_conclusion = "Stationnaire ✅" if kpss_pvalue > 0.05 else "Non-stationnaire ❌"

    print(f"   ADF  : stat={adf_stat:.4f} | p={adf_pvalue:.4f} → {adf_conclusion}")
    print(f"   KPSS : stat={kpss_stat:.4f} | p={kpss_pvalue:.4f} → {kpss_conclusion}")

    return {
        "serie": name,
        "adf_stat": adf_stat, "adf_pvalue": adf_pvalue,
        "adf_stationnaire": adf_pvalue < 0.05,
        "kpss_stat": kpss_stat, "kpss_pvalue": kpss_pvalue,
        "kpss_stationnaire": kpss_pvalue > 0.05,
    }


def plot_acf_pacf(series: pd.Series, currency: str):
    """Trace ACF et PACF pour identifier visuellement p et q."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(), lags=40, ax=axes[0])
    plot_pacf(series.dropna(), lags=40, ax=axes[1])
    axes[0].set_title(f"ACF — LogRet TND/{currency}")
    axes[1].set_title(f"PACF — LogRet TND/{currency}")
    plt.tight_layout()

    fname = os.path.join(FIGURES_DIR, f"acf_pacf_{currency.lower()}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 Figure sauvegardée → {fname}")

from statsmodels.stats.diagnostic import acorr_ljungbox

def test_residuals(residuals: pd.Series, model_name: str) -> bool:
    """
    Test de Ljung-Box sur les résidus d'un modèle ARIMA.

    H0 : les résidus sont un bruit blanc (pas d'autocorrélation résiduelle)
    Si p > 0.05 sur tous les lags → résidus OK → modèle bien spécifié
    Si p < 0.05 → résidus encore autocorrélés → modèle mal spécifié

    Args:
        residuals  : résidus du modèle ARIMA (result.resid)
        model_name : nom affiché dans le log (ex: "ARIMA(1,1,1) TND/USD")

    Returns:
        True si résidus OK, False sinon
    """
    print(f"\n   🔍 Test Ljung-Box — résidus {model_name}")

    lb = acorr_ljungbox(residuals.dropna(), lags=[10, 20], return_df=True)

    for lag in [10, 20]:
        row    = lb.loc[lag]
        status = "✅" if row["lb_pvalue"] > 0.05 else "⚠️ "
        print(f"   Lag {lag:>2} : stat={row['lb_stat']:.4f} | p={row['lb_pvalue']:.4f} {status}")

    ok = (lb["lb_pvalue"] > 0.05).all()
    if ok:
        print(f"   ✅ Résidus OK — bruit blanc confirmé — modèle bien spécifié")
    else:
        print(f"   ⚠️  Résidus autocorrélés — modèle potentiellement mal spécifié")
        print(f"   → Envisager d'augmenter p ou q")

    return ok


# -------------------------------------------------------------
#  Test rapide
# -------------------------------------------------------------
if __name__ == "__main__":
    from .load import load_data, get_series

    df = load_data()
    spot, logret = get_series(df, "USD")

    print("\n--- Stationnarité niveaux (cours spot) ---")
    test_stationarity(spot.loc[TRAIN_START:TRAIN_END], "TND_USD (niveaux)")

    print("\n--- Stationnarité log-rendements ---")
    test_stationarity(logret.loc[TRAIN_START:TRAIN_END], "LogRet_TND_USD")

    plot_acf_pacf(logret.loc[TRAIN_START:TRAIN_END], "USD")