# =============================================================
#  src/s3_lstm_xgboost/metrics.py
#  Métriques d'évaluation — RMSE, MAE, MAPE
#  Auteur : Youssef Neji | MINDS ENIT | 2025
# =============================================================

import numpy as np
import pandas as pd


def compute_metrics(actual: pd.Series, predicted: pd.Series,
                    model_name: str, currency: str,
                    horizon: int) -> dict:
    """
    Calcule RMSE, MAE et MAPE entre valeurs réelles et prévisions.
    Aligne automatiquement les deux séries sur leur intersection.
    """
    common_idx = actual.index.intersection(predicted.index)
    a = actual.loc[common_idx].dropna()
    p = predicted.loc[common_idx].dropna()
    common_idx = a.index.intersection(p.index)
    a, p = a.loc[common_idx], p.loc[common_idx]

    if len(a) == 0:
        print(f"   ⚠️  Aucune observation commune pour {model_name}")
        return {
            "Modèle": model_name, "Devise": f"TND/{currency}",
            "Horizon": f"J+{horizon}",
            "RMSE": None, "MAE": None, "MAPE (%)": None,
            "N_obs": 0
        }

    rmse = np.sqrt(np.mean((a - p) ** 2))
    mae  = np.mean(np.abs(a - p))
    mape = np.mean(np.abs((a - p) / a)) * 100

    return {
        "Modèle":   model_name,
        "Devise":   f"TND/{currency}",
        "Horizon":  f"J+{horizon}",
        "RMSE":     round(rmse, 6),
        "MAE":      round(mae, 6),
        "MAPE (%)": round(mape, 4),
        "N_obs":    len(a)
    }