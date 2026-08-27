# =============================================================
#  src/consolidate.py
#  Consolidation metrics (3→1) et predictions (3→1 par devise)
#  Idempotent : peut être appelé après chaque run sans risque.
#
#  ARIMAX EXCLU DE LA CONSOLIDATION (Vague 2 + sélection de
#  variables) : validation croisée temporelle stricte a montré
#  qu'aucune configuration exogène ne bat ARIMA seul sous
#  contrainte réaliste (pas de connaissance du futur). ARIMAX
#  reste disponible en fichiers bruts (arimax_predictions_*.csv,
#  metrics_arimax.csv, arimax_variable_selection.csv) comme
#  preuve documentée de l'exploration, mais n'entre plus dans
#  la sélection automatique du "meilleur modèle".
#
#  Usage direct : python -m src.consolidate
#  Usage import  : from src.consolidate import consolidate_all
# =============================================================

import os
import pandas as pd

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")


# =============================================================
#  METRICS — 3 fichiers → metrics_all.csv (ARIMAX exclu)
# =============================================================

def consolidate_metrics() -> pd.DataFrame:
    """
    Fusionne metrics_S2 (Naïf + ARIMA) + metrics_S3_lstm (LSTM)
    + metrics_mlp (MLP) en un seul fichier de production.

    metrics_arimax.csv volontairement EXCLU — voir note en tête
    de fichier.
    """
    sources = [
        ("metrics_S2.csv",      ["Naïf", "ARIMA"]),
        ("metrics_S3_lstm.csv", ["LSTM"]),
        ("metrics_mlp.csv",     ["MLP"]),
    ]

    frames = []
    for fname, keep_models in sources:
        path = os.path.join(PROCESSED_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df[df["Modèle"].isin(keep_models)]
        cols = ["Modèle", "Devise", "Horizon",
                "RMSE", "MAE", "MAPE (%)"]
        df = df[[c for c in cols if c in df.columns]]
        frames.append(df)

    if not frames:
        print("   ⚠️  Aucun fichier metrics source trouvé")
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(
        subset=["Modèle", "Devise", "Horizon"], keep="last"
    )

    order_map = {"Naïf": 0, "ARIMA": 1, "LSTM": 2, "MLP": 3}
    merged["_ord"] = merged["Modèle"].map(order_map).fillna(9)
    merged = merged.sort_values(
        ["Devise", "Horizon", "_ord"]
    ).drop(columns="_ord")

    out = os.path.join(PROCESSED_DIR, "metrics_all.csv")
    merged.to_csv(out, index=False)
    print(f"   ✅ metrics_all.csv : {len(merged)} lignes "
          f"({merged['Modèle'].nunique()} modèles — "
          f"ARIMAX exclu de la production)")
    return merged


# =============================================================
#  PREDICTIONS — 3 fichiers/devise → 1 fichier/devise
#  (ARIMAX exclu)
# =============================================================

def consolidate_predictions(currency: str) -> pd.DataFrame:
    """
    Fusionne arima_ + lstm_ + mlp_predictions_{cur}.csv en un
    seul predictions_{cur}.csv de production, aligné sur Date.

    arimax_predictions_{cur}.csv volontairement EXCLU.
    """
    cl = currency.lower()
    sources = [
        f"arima_predictions_{cl}.csv",
        f"lstm_predictions_{cl}.csv",
        f"mlp_predictions_{cl}.csv",
    ]

    frames = []
    for fname in sources:
        path = os.path.join(PROCESSED_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "Date"
        frames.append(df)

    if not frames:
        print(f"   ⚠️  Aucune prédiction trouvée pour {currency}")
        return pd.DataFrame()

    merged = pd.concat(frames, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]

    out = os.path.join(PROCESSED_DIR, f"predictions_{cl}.csv")
    merged.to_csv(out)
    print(f"   ✅ predictions_{cl}.csv : {merged.shape[0]} lignes "
          f"× {merged.shape[1]} colonnes (ARIMAX exclu)")
    return merged


# =============================================================
#  POINT D'ENTRÉE UNIQUE — appelé après chaque run
# =============================================================

def consolidate_all():
    print("\n🔗 Consolidation metrics + predictions "
          "(ARIMAX exclu de la production)...")
    consolidate_metrics()
    for currency in ["USD", "EUR"]:
        consolidate_predictions(currency)
    print("✅ Consolidation terminée\n")


if __name__ == "__main__":
    consolidate_all()