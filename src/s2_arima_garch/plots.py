# =============================================================
#  src/s2_arima_garch/plots.py
#  Visualisations des prévisions
#
#  Test rapide : python -m src.s2_arima_garch.plots
# =============================================================

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from .config import FIGURES_DIR, TEST_START, TEST_END


def plot_forecast(actual: pd.Series, predictions: dict, currency: str, horizon: int):
    """Trace prévisions vs réel pour tous les modèles fournis."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual.index, actual.values, label="Réel", color="black", linewidth=1.5)

    colors = {"Naïf": "gray", "ARIMA": "blue"}
    for model_name, pred in predictions.items():
        common = actual.index.intersection(pred.index)
        ax.plot(common, pred.loc[common].values, label=model_name,
                linestyle="--", color=colors.get(model_name, "red"), linewidth=1.2)

    ax.set_title(
        f"Prévisions TND/{currency} — Horizon J+{horizon} "
        f"| Test {TEST_START} → {TEST_END}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(f"TND/{currency}")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    fname = os.path.join(FIGURES_DIR, f"forecast_{currency.lower()}_J{horizon}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 {fname}")


# -------------------------------------------------------------
#  Test rapide
# -------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    idx = pd.date_range("2023-01-01", periods=50, freq="B")
    actual = pd.Series(3.0 + np.cumsum(np.random.normal(0, 0.01, 50)), index=idx)
    naive  = actual.shift(1)
    arima  = actual + np.random.normal(0, 0.005, 50)

    plot_forecast(actual, {"Naïf": naive, "ARIMA": arima}, "TEST", 1)