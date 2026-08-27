# =============================================================
#  src/s2_arima_garch/metrics.py
#  Calcul des métriques RMSE, MAE, MAPE
#
#  Test rapide : python -m src.s2_arima_garch.metrics
# =============================================================

import numpy as np
import pandas as pd


def compute_metrics(actual: pd.Series, predicted: pd.Series,
                    model_name: str, currency: str, horizon: int) -> dict:
    """
    Calcule RMSE, MAE, MAPE entre valeurs réelles et prévisions.
    Aligne automatiquement les deux séries sur leur intersection.
    """
    common_idx = actual.index.intersection(predicted.index)
    a = actual.loc[common_idx].dropna()
    p = predicted.loc[common_idx].dropna()

    common_idx = a.index.intersection(p.index)
    a, p = a.loc[common_idx], p.loc[common_idx]

    if len(a) == 0:
        print(f"   ⚠️  compute_metrics({model_name}, {currency}, "
              f"J+{horizon}) : aucune observation commune entre "
              f"actual et predicted — métriques renvoyées à NaN")
        return {
            "Modèle": model_name, "Devise": f"TND/{currency}",
            "Horizon": f"J+{horizon}",
            "RMSE": float("nan"), "MAE": float("nan"),
            "MAPE (%)": float("nan"),
        }

    rmse = np.sqrt(np.mean((a - p) ** 2))
    mae  = np.mean(np.abs(a - p))
    mape = np.mean(np.abs((a - p) / a)) * 100

    return {
        "Modèle": model_name, "Devise": f"TND/{currency}", "Horizon": f"J+{horizon}",
        "RMSE": round(rmse, 6), "MAE": round(mae, 6), "MAPE (%)": round(mape, 4),
    }


# -------------------------------------------------------------
#  Test rapide (données fictives)
# -------------------------------------------------------------
if __name__ == "__main__":
    idx = pd.date_range("2023-01-01", periods=10, freq="D")
    actual = pd.Series(np.linspace(3.0, 3.1, 10), index=idx)
    pred   = actual + np.random.normal(0, 0.01, 10)

    m = compute_metrics(actual, pred, "Test", "USD", 1)
    print(m)