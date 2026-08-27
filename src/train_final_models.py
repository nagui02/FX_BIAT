# =============================================================
#  src/train_final_models.py
#  Entraîne UNE FOIS les modèles finaux sur toutes les données
#  disponibles et les sauvegarde sur disque (.pkl / .pt).
#
#  À lancer après update_data.py (nouvelles données dispo) :
#    python -m src.train_final_models
#
#  Ces modèles sauvegardés sont ensuite chargés instantanément
#  par src/inference.py pour la prédiction à la demande.
# =============================================================

import os
import sys
import pickle
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "..", "data", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE_DIR, ".."))


def load_dataset() -> pd.DataFrame:
    p = os.path.join(PROCESSED_DIR, "dataset_final.csv")
    return pd.read_csv(p, parse_dates=["Date"], index_col="Date")


# =============================================================
#  ARIMA — sauvegarde via .save() natif statsmodels
# =============================================================

def train_and_save_arima(df: pd.DataFrame, currency: str):
    from statsmodels.tsa.arima.model import ARIMA
    from src.s2_arima_garch.arima import select_arima_order

    print(f"\n📈 ARIMA final TND/{currency}...")
    col    = f"TND_{currency}"
    series = df[col].dropna()

    order = select_arima_order(series, max_p=4, max_q=4)
    model = ARIMA(series.values, order=order).fit()

    path = os.path.join(
        MODELS_DIR, f"arima_final_{currency.lower()}.pkl"
    )
    model.save(path)

    meta = {
        "order":      order,
        "last_date":  series.index[-1].strftime("%Y-%m-%d"),
        "last_value": float(series.iloc[-1]),
        "n_obs":      len(series),
    }
    with open(path + ".meta", "wb") as f:
        pickle.dump(meta, f)

    print(f"   ✅ {path}  (ARIMA{order}, "
          f"dernière obs: {meta['last_date']})")


# =============================================================
#  ARIMAX — idem, + variables exogènes utilisées
# =============================================================

def train_and_save_arimax(df: pd.DataFrame, currency: str):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from src.s2_arima_garch.arimax import (
        select_arimax_order, get_exog_features,
        set_available_columns
    )

    print(f"\n🌍 ARIMAX final TND/{currency}...")
    col    = f"TND_{currency}"
    series = df[col].dropna()
    set_available_columns(df.columns.tolist())

    # Un modèle par horizon (variables exogènes différentes)
    for h in [1, 7, 30]:
        exog_cols = [c for c in get_exog_features(h, currency)
                     if c in df.columns]
        exog = df[exog_cols].ffill().bfill()

        order = select_arimax_order(
            series, exog, max_p=3, max_q=3
        )
        model = SARIMAX(
            series.values, exog=exog.values, order=order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        path = os.path.join(
            MODELS_DIR,
            f"arimax_final_{currency.lower()}_J{h}.pkl"
        )
        model.save(path)

        meta = {
            "order":       order,
            "exog_cols":   exog_cols,
            "last_exog":   exog.iloc[-1].values.tolist(),
            "last_date":   series.index[-1].strftime("%Y-%m-%d"),
            "last_value":  float(series.iloc[-1]),
        }
        with open(path + ".meta", "wb") as f:
            pickle.dump(meta, f)

        print(f"   ✅ {path}  (ARIMAX{order}, J+{h})")


# =============================================================
#  MLP — sauvegarde via joblib
# =============================================================

def train_and_save_mlp(df: pd.DataFrame, currency: str):
    """
    FIX : réutilise désormais train_final_mlp() de mlp.py — même
    architecture (MLPModel PyTorch), même cible (rendement log
    reconstruit), même sélection de variables macro par horizon
    que le modèle réellement validé par Diebold-Mariano. Élimine
    la divergence précédemment identifiée avec l'ancienne version
    (sklearn MLPRegressor, cible = niveau brut, variables macro
    figées sur J+1 pour tous les horizons).
    """
    import torch
    from src.s3_lstm_xgboost.mlp import train_final_mlp

    print(f"\n⚡ MLP final TND/{currency}...")

    for h in [1, 7, 30]:
        bundle = train_final_mlp(df, currency, h, end_date=None)

        path = os.path.join(
            MODELS_DIR, f"mlp_final_{currency.lower()}_J{h}.pt"
        )
        torch.save(bundle["model_state"], path)

        meta = {k: v for k, v in bundle.items()
                if k != "model_state"}
        with open(path + ".meta", "wb") as f:
            pickle.dump(meta, f)

        print(f"   ✅ {path}  (J+{h}, "
              f"dernière obs: {bundle['last_date']})")


# =============================================================
#  PIPELINE PRINCIPAL
# =============================================================

def run_all():
    print("=" * 60)
    print("  ENTRAÎNEMENT MODÈLES FINAUX (pour inférence rapide)")
    print("=" * 60)

    df = load_dataset()
    print(f"Dataset : {df.shape} | "
          f"jusqu'à {df.index[-1].date()}")

    for currency in ["USD", "EUR"]:
        train_and_save_arima(df, currency)
        train_and_save_arimax(df, currency)
        train_and_save_mlp(df, currency)
        # LSTM : voir note ci-dessous

    print("\n✅ Tous les modèles finaux sont sauvegardés "
          f"dans {MODELS_DIR}")
    print("   → Utilisables instantanément via src/inference.py")


if __name__ == "__main__":
    run_all()