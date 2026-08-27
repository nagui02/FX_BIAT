# =============================================================
#  src/inference.py
#  Prédiction instantanée à partir des modèles finaux sauvegardés
#  Sélection du modèle dynamique (metrics_all.csv) — plus de
#  dictionnaire codé en dur pointant vers un modèle exclu.
#
#  Usage :
#    from src.inference import predict_for_date
#    result = predict_for_date("USD", "2026-09-15")
# =============================================================

import os
import pickle
import numpy as np
import pandas as pd

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR    = os.path.join(BASE_DIR, "..", "data", "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

# Modèles supportés pour l'inférence instantanée. LSTM nécessite
# une séquence de 30 jours (pas juste un vecteur de features) —
# non supporté ici, repli automatique sur ARIMA si LSTM est
# désigné comme meilleur modèle par les métriques.
INSTANT_SUPPORTED = {"ARIMA", "MLP", "Naïf"}


def _nearest_horizon(n_steps: int) -> int:
    if n_steps <= 1:
        return 1
    if n_steps <= 10:
        return 7
    return 30


def get_best_model_dynamic(currency: str, horizon: int) -> str:
    """
    Détermine le meilleur modèle depuis metrics_all.csv (RMSE
    minimal) — source de vérité unique, cohérente avec le
    dashboard et S5 (compare.py). Repli sur ARIMA si le fichier
    est absent ou si aucune ligne ne correspond.
    """
    path = os.path.join(PROCESSED_DIR, "metrics_all.csv")
    if not os.path.exists(path):
        return "ARIMA"
    try:
        df = pd.read_csv(path)
        sub = df[
            (df["Devise"] == f"TND/{currency}") &
            (df["Horizon"] == f"J+{horizon}")
        ].dropna(subset=["RMSE"])
        if sub.empty:
            return "ARIMA"
        best = sub.loc[sub["RMSE"].idxmin(), "Modèle"]
    except Exception:
        return "ARIMA"

    if best not in INSTANT_SUPPORTED:
        print(f"[inference] {best} désigné meilleur modèle mais "
              f"non supporté pour l'inférence instantanée "
              f"(nécessite une séquence) — repli sur ARIMA")
        return "ARIMA"
    return best


def _load_arima(currency: str):
    import statsmodels.api as sm
    path = os.path.join(
        MODELS_DIR, f"arima_final_{currency.lower()}.pkl"
    )
    if not os.path.exists(path):
        return None, None
    model = sm.load(path)
    with open(path + ".meta", "rb") as f:
        meta = pickle.load(f)
    return model, meta


def _load_mlp(currency: str, horizon: int):
    """Charge le bundle MLP (state_dict + scalers + métadonnées)."""
    import torch
    from src.s3_lstm_xgboost.mlp import MLPModel

    path = os.path.join(
        MODELS_DIR, f"mlp_final_{currency.lower()}_J{horizon}.pt"
    )
    meta_path = path + ".meta"
    if not os.path.exists(path) or not os.path.exists(meta_path):
        return None, None

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    model = MLPModel(
        input_size=len(meta["feat_cols"]),
        hidden_sizes=meta["hidden_sizes"],
        dropout=meta["dropout"]
    )
    model.load_state_dict(torch.load(path))
    model.eval()
    return model, meta


def get_last_available_date() -> pd.Timestamp:
    p = os.path.join(PROCESSED_DIR, "dataset_final.csv")
    df = pd.read_csv(p, usecols=["Date"], parse_dates=["Date"])
    return df["Date"].iloc[-1]


def business_days_between(start: pd.Timestamp,
                          end: pd.Timestamp) -> int:
    if end <= start:
        return 0
    return len(pd.bdate_range(start, end)) - 1


def predict_for_date(currency: str, target_date: str,
                     model_override: str = None) -> dict:
    """
    Prédiction instantanée pour une date cible.

    Le modèle est choisi DYNAMIQUEMENT depuis metrics_all.csv
    (sauf si model_override est fourni), avec repli automatique
    sur ARIMA si le meilleur modèle désigné (ex: LSTM) n'est pas
    supporté pour l'inférence instantanée.
    """
    last_date = get_last_available_date()
    target    = pd.Timestamp(target_date)
    n_steps   = business_days_between(last_date, target)

    if n_steps <= 0:
        return {
            "error": (
                f"target_date ({target_date}) doit être "
                f"après la dernière donnée disponible "
                f"({last_date.date()})"
            )
        }

    horizon    = _nearest_horizon(n_steps)
    model_name = (model_override or
                  get_best_model_dynamic(currency, horizon))

    warning = None
    if n_steps > 30:
        warning = (
            f"⚠️ {n_steps} jours dans le futur — au-delà de "
            f"J+30, la fiabilité du modèle n'est plus validée."
        )

    if model_name == "Naïf":
        p = os.path.join(PROCESSED_DIR, "dataset_final.csv")
        df = pd.read_csv(p, parse_dates=["Date"], index_col="Date")
        last_value = float(df[f"TND_{currency}"].iloc[-1])
        return {
            "currency": currency, "target_date": target_date,
            "model": "Naïf", "horizon_used": horizon,
            "n_steps": n_steps, "value": round(last_value, 5),
            "last_known_date": str(last_date.date()),
            "last_known_value": last_value,
            "warning": warning,
        }

    if model_name == "MLP":
        model, meta = _load_mlp(currency, horizon)
        if model is None:
            return {"error": (
                "Modèle MLP non trouvé. Lancez : "
                "python -m src.train_final_models"
            )}
        import torch
        with torch.no_grad():
            pred_sc = model(
                torch.FloatTensor(meta["last_row_sc"])
            ).numpy()
        pred_ret = meta["scaler_y"].inverse_transform(
            pred_sc.reshape(-1, 1)
        ).flatten()[0]
        value = meta["last_value"] * np.exp(pred_ret)
        return {
            "currency": currency, "target_date": target_date,
            "model": "MLP", "horizon_used": horizon,
            "n_steps": n_steps, "value": round(float(value), 5),
            "last_known_date": meta["last_date"],
            "last_known_value": meta["last_value"],
            "warning": warning,
        }

    # ARIMA (défaut / repli)
    model, meta = _load_arima(currency)
    if model is None:
        return {"error": (
            "Aucun modèle ARIMA sauvegardé. Lancez : "
            "python -m src.train_final_models"
        )}
    fc = model.forecast(steps=n_steps)
    value = float(fc.iloc[-1] if hasattr(fc, "iloc") else fc[-1])
    return {
        "currency": currency, "target_date": target_date,
        "model": "ARIMA", "horizon_used": horizon,
        "n_steps": n_steps, "value": round(value, 5),
        "last_known_date": meta["last_date"],
        "last_known_value": meta["last_value"],
        "warning": warning,
    }


if __name__ == "__main__":
    import sys
    cur  = sys.argv[1] if len(sys.argv) > 1 else "USD"
    date = sys.argv[2] if len(sys.argv) > 2 else "2026-09-01"
    result = predict_for_date(cur, date)
    print(result)