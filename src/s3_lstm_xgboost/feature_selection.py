# =============================================================
#  src/s3_lstm_xgboost/feature_selection.py
#  Sélection data-driven des variables exogènes pour LSTM/MLP —
#  DÉCOUPLÉE d'ARIMAX (modèle exclu de la production).
#
#  Deux temps :
#   1. Classement exploratoire (corrélation + importance Random
#      Forest) de TOUTES les variables disponibles, sur le
#      SOUS-TRAIN (2012-2021) uniquement.
#   2. Validation empirique : quelques jeux de variables candidats
#      (dont le top-K data-driven) sont réellement entraînés sur un
#      MLP (proxy rapide — LSTM trop coûteux pour un balayage de
#      candidats) et comparés par RMSE sur la fenêtre de validation
#      interne (2022-2023). Le vrai test (2024-2026) n'est JAMAIS
#      utilisé ici. Réutilise MLPModel/train_mlp/predict_mlp de
#      mlp.py telles quelles — pas de MLP ad hoc dupliqué.
#
#  ⚠️ Le gagnant n'est PAS adopté automatiquement. Pour l'appliquer :
#   1. Ouvrir exog_features.py
#   2. Mettre à jour EXOG_FEATURES_S3[devise][horizon]
#   3. Relancer run_s3.py (mode complet) pour la validation finale
#      sur le vrai test (2024-2026) — même principe que pour ARIMAX
#      en son temps : jamais d'adoption sans validation croisée.
#
#  Auteur : Youssef Neji | MINDS ENIT | 2026
#  Usage : python -m src.s3_lstm_xgboost.feature_selection
# =============================================================

import warnings
import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

from .config import PROCESSED_DIR, TRAIN_START, HORIZONS, SEED
from .load import load_data, build_features
from .mlp import MLPModel, prepare_mlp_data, train_mlp, predict_mlp, get_mlp_config
from .metrics import compute_metrics
from .exog_features import EXOG_FEATURES_S3

# Même sous-train que arimax_variable_selection.py (S2) — ne
# JAMAIS étendre jusqu'à 2022+ ici.
SUBTRAIN_END      = "2021-12-31"
VALIDATION_START  = "2022-01-01"
VALIDATION_END    = "2023-12-31"

EXCLUDED_COLS = {
    "TND_USD", "TND_EUR", "LogRet_TND_USD", "LogRet_TND_EUR"
}
TOP_K = 5


# =============================================================
#  1. CLASSEMENT EXPLORATOIRE
# =============================================================

def _adf_stationary(series: pd.Series):
    s = series.dropna()
    if len(s) < 30:
        return None
    try:
        return adfuller(s, autolag="AIC")[1] < 0.05
    except Exception:
        return None


def forward_return(spot: pd.Series, horizon: int) -> pd.Series:
    """r_h[t] = ln(S[t+h]/S[t]) — la cible réellement prédite."""
    return np.log(spot.shift(-horizon) / spot)


def rank_features_for_config(
    df: pd.DataFrame, currency: str, horizon: int
) -> pd.DataFrame:
    """Corrélation + importance RF de toutes les variables
    candidates, sous-train uniquement (2012-2021)."""
    col_spot = f"TND_{currency}"
    sub = df.loc[TRAIN_START:SUBTRAIN_END]
    target = forward_return(sub[col_spot], horizon)

    candidates = [c for c in df.columns if c not in EXCLUDED_COLS]
    rows, X_cols = [], []

    for col in candidates:
        x = sub[col]
        y = target.dropna()
        common = x.dropna().index.intersection(y.index)
        x_c, y_c = x.loc[common], y.loc[common]
        if len(x_c) < 60:
            rows.append({"variable": col, "n_obs": len(x_c),
                         "correlation": None, "adf_stationnaire": None})
            continue
        corr = x_c.corr(y_c)
        rows.append({
            "variable": col, "n_obs": len(x_c),
            "correlation": round(corr, 4) if not pd.isna(corr) else None,
            "adf_stationnaire": _adf_stationary(x_c),
        })
        X_cols.append(col)

    rank_df = pd.DataFrame(rows)
    rank_df["abs_correlation"] = rank_df["correlation"].abs()

    X_full = sub[X_cols].ffill()
    common = X_full.dropna().index.intersection(target.dropna().index)
    X_fit, y_fit = X_full.loc[common], target.loc[common]

    importance_map = {}
    if len(X_fit) >= 100:
        rf = RandomForestRegressor(n_estimators=300, max_depth=5,
                                   random_state=42, n_jobs=-1)
        rf.fit(X_fit.values, y_fit.values)
        importance_map = dict(zip(X_cols, rf.feature_importances_))

    rank_df["rf_importance"] = rank_df["variable"].map(importance_map)
    rank_df["rf_importance"] = rank_df["rf_importance"].round(4)
    rank_df = rank_df.sort_values(
        "rf_importance", ascending=False, na_position="last"
    )
    rank_df.insert(0, "currency", currency)
    rank_df.insert(1, "horizon", horizon)
    return rank_df.reset_index(drop=True)


def suggest_top_k(ranked: pd.DataFrame, k: int = TOP_K) -> list:
    if ranked["rf_importance"].notna().any():
        top = ranked.sort_values("rf_importance", ascending=False).head(k)
    else:
        top = ranked.sort_values("abs_correlation", ascending=False).head(k)
    return top["variable"].tolist()


# =============================================================
#  2. VALIDATION EMPIRIQUE — MLP rapide, RMSE sur 2022-2023
# =============================================================

def _internal_split(feat: pd.DataFrame, currency: str) -> tuple:
    """
    Split interne sous-train/validation (2012-2021 / 2022-2023),
    DISTINCT de split_features() (qui utilise les bornes de
    production TRAIN_END=2023/TEST_START=2024) — ne touche jamais
    le vrai test 2024-2026.
    """
    col_spot = f"TND_{currency}"
    train = feat.loc[TRAIN_START:SUBTRAIN_END]
    val   = feat.loc[VALIDATION_START:VALIDATION_END]
    feature_cols = [c for c in feat.columns
                    if c not in [col_spot, f"LogRet_TND_{currency}"]]
    return (train[feature_cols], train[col_spot],
            val[feature_cols], val[col_spot], feature_cols)


def evaluate_candidate(
    df: pd.DataFrame, currency: str, horizon: int,
    macro_cols: list
) -> float:
    """
    Entraîne un MLP (mode rapide, 5 epochs) sur le sous-train
    avec le jeu de variables `macro_cols`, évalue le RMSE sur la
    validation interne. Réutilise l'architecture et la procédure
    EXACTES de mlp.py.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    feat = build_features(df, currency, horizon, macro_override=macro_cols)
    X_train, y_train, X_val, y_val, feat_cols = _internal_split(feat, currency)

    if len(X_train) < 200 or len(X_val) < 30:
        print(f"      ⚠️  Pas assez de données (train={len(X_train)}, "
              f"val={len(X_val)}) — candidat ignoré")
        return float("nan")

    level_full = pd.concat([y_train, y_val]).sort_index()
    mlp_cfg = get_mlp_config(horizon)

    train_loader, scaler_X, scaler_y, X_val_sc, target_dates, S_ref = \
        prepare_mlp_data(X_train, X_val, level_full, horizon,
                         mlp_cfg["batch_size"])

    model = MLPModel(input_size=len(feat_cols),
                     hidden_sizes=mlp_cfg["hidden_sizes"],
                     dropout=mlp_cfg["dropout"])
    train_mlp(model, train_loader, mlp_cfg, quick=True)

    pred = predict_mlp(model, X_val_sc, target_dates, S_ref,
                       scaler_y, horizon)
    actual = level_full.loc[target_dates]
    m = compute_metrics(actual, pred, "MLP-candidat", currency, horizon)
    return m["RMSE"]


def compare_candidates(
    df: pd.DataFrame, currency: str, horizon: int,
    data_driven_top_k: list
) -> pd.DataFrame:
    current = EXOG_FEATURES_S3[currency][horizon]

    candidates = {
        "Actuel (production)": current,
        "Data-driven (top-K)": data_driven_top_k,
        "Aucun (lags/MA/GARCH seuls)": [],
        "Toutes disponibles": [
            c for c in df.columns if c not in EXCLUDED_COLS
        ],
    }

    rows = []
    for label, cols in candidates.items():
        print(f"      🔎 {label} ({len(cols)} variables)...")
        rmse = evaluate_candidate(df, currency, horizon, cols)
        rows.append({
            "currency": currency, "horizon": horizon,
            "candidat": label, "n_variables": len(cols),
            "variables": cols, "RMSE_validation": rmse,
        })
        print(f"         RMSE validation : {rmse}")

    return pd.DataFrame(rows)


# =============================================================
#  3. ORCHESTRATION
# =============================================================

def run_all_feature_selection():
    print("=" * 70)
    print("  SÉLECTION DATA-DRIVEN DES FEATURES — LSTM/MLP (S3)")
    print(f"  Sous-train : {TRAIN_START} → {SUBTRAIN_END}")
    print(f"  Validation interne : {VALIDATION_START} → {VALIDATION_END}")
    print("  (ARIMAX non utilisé ici — modèle exclu de la production)")
    print("=" * 70)

    df = load_data()
    all_rankings, all_comparisons = [], []

    for currency in ["USD", "EUR"]:
        for horizon in HORIZONS:
            print(f"\n{'='*60}\n  TND/{currency} — J+{horizon}\n{'='*60}")

            print("\n   --- Classement exploratoire ---")
            ranked = rank_features_for_config(df, currency, horizon)
            print(ranked.head(8).to_string(index=False))
            top_k = suggest_top_k(ranked)
            print(f"   → Top-{TOP_K} proposé : {top_k}")
            all_rankings.append(ranked)

            print("\n   --- Validation empirique (RMSE MLP) ---")
            comp = compare_candidates(df, currency, horizon, top_k)
            all_comparisons.append(comp)

            valid = comp.dropna(subset=["RMSE_validation"])
            if not valid.empty:
                best = valid.loc[valid["RMSE_validation"].idxmin()]
                print(f"\n   🏆 Meilleur candidat : {best['candidat']} "
                      f"(RMSE={best['RMSE_validation']})")

    rank_df = pd.concat(all_rankings, ignore_index=True)
    comp_df = pd.concat(all_comparisons, ignore_index=True)

    rank_df.to_csv(f"{PROCESSED_DIR}/s3_feature_ranking.csv", index=False)
    comp_df.to_csv(f"{PROCESSED_DIR}/s3_feature_candidates.csv", index=False)
    print(f"\n💾 s3_feature_ranking.csv")
    print(f"💾 s3_feature_candidates.csv")

    print("\n" + "=" * 70)
    print("  RÉSUMÉ — meilleur candidat par devise/horizon")
    print("=" * 70)
    valid_comp = comp_df.dropna(subset=["RMSE_validation"])
    if not valid_comp.empty:
        best_rows = valid_comp.loc[
            valid_comp.groupby(["currency", "horizon"])["RMSE_validation"].idxmin()
        ]
        print(best_rows[
            ["currency", "horizon", "candidat", "n_variables", "RMSE_validation"]
        ].to_string(index=False))

    print("\n⚠️  Aucune adoption automatique. Pour appliquer un résultat :")
    print("   1. Ouvrir exog_features.py")
    print("   2. Mettre à jour EXOG_FEATURES_S3[devise][horizon] avec")
    print("      la liste 'variables' du candidat gagnant")
    print("   3. Relancer run_s3.py (mode complet) pour la validation")
    print("      finale sur le vrai test (2024-2026)")

    return rank_df, comp_df


if __name__ == "__main__":
    run_all_feature_selection()