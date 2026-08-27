# =============================================================
#  src/s2_arima_garch/arimax_variable_selection.py
#  Sélection des variables exogènes ARIMAX par validation croisée
#  temporelle — JAMAIS sur le test (2024-2026).
#
#  Principe : découpe le TRAIN (2004-2023) en sous-train +
#  fenêtre de validation interne (2022-2023). On compare des
#  jeux de variables candidats UNIQUEMENT sur cette validation.
#  Le vrai test ne sera regardé qu'une fois, à la toute fin,
#  avec le jeu de variables déjà figé.
#
#  Auteur : Youssef Neji | MINDS ENIT | 2026
#  Usage : python -m src.s2_arima_garch.arimax_variable_selection
# =============================================================

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .config import PROCESSED_DIR, TRAIN_START, TRAIN_END, HORIZONS
from .load import load_data
from .arimax import forecast_arimax, set_available_columns
from .metrics import compute_metrics

# ── Découpage interne, à distance du vrai test ────────────────
SUBTRAIN_END      = "2021-12-31"
VALIDATION_START  = "2022-01-01"
VALIDATION_END    = TRAIN_END  # "2023-12-31"

# ── Ordre fixé par devise (pas de grille AIC par candidat —
#    isole l'effet du choix de variables, contrôle l'ordre) ────
FIXED_ORDER = {"USD": (1, 1, 1), "EUR": (2, 1, 1)}

# ── Refit — mêmes valeurs qu'en production, pour rester
#    représentatif du comportement réel ─────────────────────────
REFIT_BY_HORIZON = {1: 1, 7: 2, 30: 7}

# ── Variables BCT complémentaires (ajout 2026) ────────────────
# Features dérivées stationnaires uniquement — cf. data.py::
# load_bct_extra() et src/s3_lstm_xgboost/load.py::BCT_EXTRA_FEATURES.
# Décalage de publication anti look-ahead bias déjà géré en amont
# dans data.py, donc aucune précaution supplémentaire nécessaire
# ici — ces colonnes se comportent comme n'importe quelle autre
# variable exogène du point de vue de ce script.
BCT_EXTRA_VARS = ["reserves_mom_pct", "bct_market_share_pct"]

# ── Candidats — hypothèse-driven, pas une recherche exhaustive ──
CANDIDATE_SETS = {
    "USD": {
        1:  {"Actuel (VIX/EURUSD/Brent/DXY)":
                 ["VIX", "EURUSD", "Brent_USD", "DXY"],
             "Minimal (EURUSD seul)": ["EURUSD"],
             "Lent (taux+macro)":
                 ["TauxFed_USD", "TauxMMBCT", "IPC_USA", "US10Y"],
             "Toutes rapides":
                 ["VIX", "Brent_USD", "EURUSD", "DXY", "Gold_USD"],
             "Toutes (rapides+lentes)":
                 ["VIX", "Brent_USD", "EURUSD", "DXY", "Gold_USD",
                  "TauxFed_USD", "TauxMMBCT", "IPC_USA", "US10Y"],
             "Actuel + BCT":
                 ["VIX", "EURUSD", "Brent_USD", "DXY"] + BCT_EXTRA_VARS,
             "BCT seul (nouveauté)": list(BCT_EXTRA_VARS),
             "Aucun (ARIMA pur)": []},
        7:  {"Actuel (VIX/EURUSD/Brent/DXY)":
                 ["VIX", "EURUSD", "Brent_USD", "DXY"],
             "Minimal (EURUSD seul)": ["EURUSD"],
             "Lent (taux+macro)":
                 ["TauxFed_USD", "TauxMMBCT", "IPC_USA", "US10Y"],
             "Toutes rapides":
                 ["VIX", "Brent_USD", "EURUSD", "DXY", "Gold_USD"],
             "Toutes (rapides+lentes)":
                 ["VIX", "Brent_USD", "EURUSD", "DXY", "Gold_USD",
                  "TauxFed_USD", "TauxMMBCT", "IPC_USA", "US10Y"],
             "Actuel + BCT":
                 ["VIX", "EURUSD", "Brent_USD", "DXY"] + BCT_EXTRA_VARS,
             "BCT seul (nouveauté)": list(BCT_EXTRA_VARS),
             "Aucun (ARIMA pur)": []},
        30: {"Actuel (Taux/IPC/BCT/US10Y)":
                 ["TauxFed_USD", "IPC_USA", "TauxMMBCT", "US10Y"],
             "Minimal (EURUSD seul)": ["EURUSD"],
             "Lent (taux+macro)":
                 ["TauxFed_USD", "TauxMMBCT", "IPC_USA", "US10Y"],
             "Toutes rapides":
                 ["VIX", "Brent_USD", "EURUSD", "DXY", "Gold_USD"],
             "Toutes (rapides+lentes)":
                 ["VIX", "Brent_USD", "EURUSD", "DXY", "Gold_USD",
                  "TauxFed_USD", "TauxMMBCT", "IPC_USA", "US10Y"],
             "Actuel + BCT":
                 ["TauxFed_USD", "IPC_USA", "TauxMMBCT", "US10Y"] + BCT_EXTRA_VARS,
             "BCT seul (nouveauté)": list(BCT_EXTRA_VARS),
             "Aucun (ARIMA pur)": []},
    },
    "EUR": {
        1:  {"Actuel (EURUSD/VIX/BCE/EUR10Y)":
                 ["EURUSD", "VIX", "TauxBCE_EUR", "EUR10Y"],
             "Minimal (EURUSD seul)": ["EURUSD"],
             "Lent (taux+macro)":
                 ["TauxBCE_EUR", "TauxMMBCT", "EUR10Y"],
             "Toutes rapides":
                 ["VIX", "Brent_USD", "EURUSD", "DXY", "Gold_USD"],
             "Toutes (rapides+lentes)":
                 ["VIX", "Brent_USD", "EURUSD", "DXY", "Gold_USD",
                  "TauxBCE_EUR", "TauxMMBCT", "EUR10Y"],
             "Actuel + BCT":
                 ["EURUSD", "VIX", "TauxBCE_EUR", "EUR10Y"] + BCT_EXTRA_VARS,
             "BCT seul (nouveauté)": list(BCT_EXTRA_VARS),
             "Aucun (ARIMA pur)": []},
        7:  {"Actuel (EURUSD/VIX/BCE/EUR10Y)":
                 ["EURUSD", "VIX", "TauxBCE_EUR", "EUR10Y"],
             "Minimal (EURUSD seul)": ["EURUSD"],
             "Lent (taux+macro)":
                 ["TauxBCE_EUR", "TauxMMBCT", "EUR10Y"],
             "Toutes rapides":
                 ["VIX", "Brent_USD", "EURUSD", "DXY", "Gold_USD"],
             "Toutes (rapides+lentes)":
                 ["VIX", "Brent_USD", "EURUSD", "DXY", "Gold_USD",
                  "TauxBCE_EUR", "TauxMMBCT", "EUR10Y"],
             "Actuel + BCT":
                 ["EURUSD", "VIX", "TauxBCE_EUR", "EUR10Y"] + BCT_EXTRA_VARS,
             "BCT seul (nouveauté)": list(BCT_EXTRA_VARS),
             "Aucun (ARIMA pur)": []},
        30: {"Actuel (EURUSD/VIX/BCE/EUR10Y)":
                 ["EURUSD", "VIX", "TauxBCE_EUR", "EUR10Y"],
             "Minimal (EURUSD seul)": ["EURUSD"],
             "Lent (taux+macro)":
                 ["TauxBCE_EUR", "TauxMMBCT", "EUR10Y"],
             "Toutes rapides":
                 ["VIX", "Brent_USD", "EURUSD", "DXY", "Gold_USD"],
             "Toutes (rapides+lentes)":
                 ["VIX", "Brent_USD", "EURUSD", "DXY", "Gold_USD",
                  "TauxBCE_EUR", "TauxMMBCT", "EUR10Y"],
             "Actuel + BCT":
                 ["EURUSD", "VIX", "TauxBCE_EUR", "EUR10Y"] + BCT_EXTRA_VARS,
             "BCT seul (nouveauté)": list(BCT_EXTRA_VARS),
             "Aucun (ARIMA pur)": []},
    },
}


def run_selection_for_config(
    df: pd.DataFrame, currency: str, horizon: int
) -> pd.DataFrame:
    """
    Compare tous les candidats pour une devise/horizon donné,
    UNIQUEMENT sur la fenêtre de validation interne (2022-2023).
    """
    col_spot = f"TND_{currency}"
    spot_subtrain = df[col_spot].loc[TRAIN_START:SUBTRAIN_END]
    spot_val      = df[col_spot].loc[VALIDATION_START:VALIDATION_END]

    order = FIXED_ORDER[currency]
    refit = REFIT_BY_HORIZON.get(horizon, 1)

    candidates = CANDIDATE_SETS[currency][horizon]
    rows = []

    for label, exog_cols in candidates.items():
        print(f"\n   🔎 Candidat : {label} — {exog_cols or '(aucune)'}")

        if not exog_cols:
            # ARIMA pur — pas de SARIMAX, juste ARIMA classique
            from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
            history = list(spot_subtrain.values)
            preds   = {}
            dates   = list(spot_val.index)
            for i in range(len(spot_val)):
                target_idx = i + horizon - 1
                if i % refit == 0 or i == 0:
                    result = ARIMAModel(history, order=order).fit()
                else:
                    result = result.append(
                        [history[-1]], refit=False
                    )
                fc = result.forecast(steps=horizon)
                fc_val = fc[-1] if hasattr(fc, "__len__") else fc
                if target_idx < len(spot_val):
                    preds[dates[target_idx]] = fc_val
                history.append(spot_val.values[i])
            pred_series = pd.Series(preds).reindex(dates)
        else:
            available = [c for c in exog_cols if c in df.columns]
            if len(available) < len(exog_cols):
                missing = set(exog_cols) - set(available)
                print(f"      ⚠️  Colonnes manquantes ignorées : {missing}")

            exog_df    = df[available].ffill()
            exog_train = exog_df.loc[TRAIN_START:SUBTRAIN_END]
            exog_val   = exog_df.loc[VALIDATION_START:VALIDATION_END]

            pred_series, _ = forecast_arimax(
                spot_subtrain, spot_val,
                exog_train, exog_val,
                order, horizon, refit_every=refit,
                verbose_every=1000
            )

        m = compute_metrics(
            spot_val, pred_series, label, currency, horizon
        )
        rows.append({
            "currency": currency, "horizon": horizon,
            "candidat": label,
            "n_variables": len(exog_cols),
            "RMSE": m["RMSE"], "MAE": m["MAE"],
        })
        print(f"      RMSE validation : {m['RMSE']}")

    return pd.DataFrame(rows)


def run_all_selection() -> pd.DataFrame:
    print("=" * 70)
    print("  SÉLECTION VARIABLES EXOGÈNES ARIMAX — VALIDATION CROISÉE")
    print(f"  Sous-train : {TRAIN_START} → {SUBTRAIN_END}")
    print(f"  Validation interne : {VALIDATION_START} → {VALIDATION_END}")
    print(f"  (Le vrai test 2024-2026 n'est JAMAIS regardé dans ce script)")
    print("=" * 70)

    df = load_data()
    set_available_columns(df.columns.tolist())

    all_results = []
    for currency in ["USD", "EUR"]:
        for horizon in HORIZONS:
            print(f"\n{'='*60}")
            print(f"  TND/{currency} — J+{horizon}")
            print(f"{'='*60}")
            res = run_selection_for_config(df, currency, horizon)
            all_results.append(res)

    final_df = pd.concat(all_results, ignore_index=True)

    print("\n" + "=" * 70)
    print("  RÉSULTATS COMPLETS (validation interne uniquement)")
    print("=" * 70)
    print(final_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("  MEILLEUR CANDIDAT PAR DEVISE/HORIZON")
    print("=" * 70)
    best = (final_df.loc[
        final_df.groupby(["currency", "horizon"])["RMSE"].idxmin()
    ])
    print(best.to_string(index=False))

    out_path = os.path.join(
        PROCESSED_DIR, "arimax_variable_selection.csv"
    )
    final_df.to_csv(out_path, index=False)
    print(f"\n💾 {out_path}")

    print("\n⚠️  IMPORTANT : ces résultats viennent de la validation")
    print("   interne (2022-2023), PAS du vrai test (2024-2026).")
    print("   Si un candidat autre que 'Actuel' gagne — y compris")
    print("   'Actuel + BCT' ou 'BCT seul' — il faudra mettre à jour")
    print("   get_exog_features() dans arimax.py, puis relancer")
    print("   run_arimax.py UNE SEULE FOIS sur le vrai test pour")
    print("   valider le choix final.")

    return final_df


if __name__ == "__main__":
    run_all_selection()