# =============================================================
#  src/s2_arima_garch/run_s2.py
#  Orchestrateur final — lance tout le pipeline S2
#
#  Lancer depuis la racine du projet :
#    python -m src.s2_arima_garch.run_s2
#  Mode rapide pour tester la mécanique sans attendre :
#    python -m src.s2_arima_garch.run_s2 --quick
# =============================================================

import sys
import os
import pandas as pd

from .config import PROCESSED_DIR, TRAIN_START, TRAIN_END, TEST_START, TEST_END, HORIZONS
from .load import load_data, split_train_test, get_series
from .stationarity import test_stationarity, plot_acf_pacf
from .naive import benchmark_naive
from .arima import select_arima_order, forecast_arima
from .garch import fit_garch, forecast_garch_test_period, plot_garch_volatility, save_garch_params
from .metrics import compute_metrics
from .plots import plot_forecast


def run_currency(df: pd.DataFrame, currency: str, quick: bool = False) -> dict:
    print(f"\n{'='*60}\n  ARIMA + GARCH — TND/{currency}\n{'='*60}")

    spot, logret = get_series(df, currency)
    train, test  = split_train_test(spot)

    print("\n--- Stationnarité ---")
    test_stationarity(spot.loc[TRAIN_START:TRAIN_END], f"TND_{currency} (niveaux)")
    test_stationarity(logret.loc[TRAIN_START:TRAIN_END], f"LogRet_TND_{currency}")
    plot_acf_pacf(logret.loc[TRAIN_START:TRAIN_END], currency)

    print("\n--- Benchmark Naïf ---")
    naive_preds = {h: benchmark_naive(train, test, h) for h in HORIZONS}

    print("\n--- ARIMA ---")
    if quick:
        order     = (1, 1, 1)
        test_used = test.iloc[:30]
        refit     = 10
    else:
        order     = select_arima_order(train, max_p=6, max_q=6)
        test_used = test
        refit     = 1

    arima_preds = {}
    residuals   = None
    for h in HORIZONS:
        pred_h, res_h = forecast_arima(train, test_used, order, h, refit_every=refit)
        arima_preds[h] = pred_h
        if residuals is None:
            residuals = res_h  # résidus du modèle du 1er horizon, pour Ljung-Box

    print("\n--- Validation résidus ARIMA ---")
    from .stationarity import test_residuals
    residuals_ok = test_residuals(residuals, f"ARIMA{order} TND/{currency}")
    if not residuals_ok:
        print(f"   💡 Suggestion : relancer select_arima_order avec max_p=6, max_q=6")

    print("\n--- GARCH ---")
    sigma_train, result = fit_garch(logret, currency)
    sigma_test  = forecast_garch_test_period(result, logret, currency)
    sigma_full  = pd.concat([sigma_train, sigma_test])
    plot_garch_volatility(sigma_full, currency)

    print("\n--- Métriques ---")
    metrics = []
    for h in HORIZONS:
        m_naive = compute_metrics(test_used, naive_preds[h], "Naïf",  currency, h)
        m_arima = compute_metrics(test_used, arima_preds[h], "ARIMA", currency, h)
        metrics.extend([m_naive, m_arima])
        print(f"\n   J+{h} : Naïf RMSE={m_naive['RMSE']} | ARIMA RMSE={m_arima['RMSE']}")

    print("\n--- Figures ---")
    for h in HORIZONS:
        plot_forecast(test_used, {"Naïf": naive_preds[h], "ARIMA": arima_preds[h]}, currency, h)

    preds_dict = {"Actual": test_used}
    for h in HORIZONS:
        preds_dict[f"Naif_J{h}"]  = naive_preds[h]
        preds_dict[f"ARIMA_J{h}"] = arima_preds[h]
    preds_df = pd.DataFrame(preds_dict)
    preds_df.to_csv(os.path.join(PROCESSED_DIR, f"arima_predictions_{currency.lower()}.csv"))
    sigma_full.to_csv(os.path.join(PROCESSED_DIR, f"garch_sigma_{currency.lower()}.csv"))

    return {"metrics": metrics, "order": order, "garch_result": result}


def run_pipeline(quick: bool = False):
    mode_str = "RAPIDE (test mécanique)" if quick else "COMPLET"
    print("=" * 60)
    print(f"  PIPELINE S2 — ARIMA + GARCH — Mode {mode_str}")
    print("=" * 60)

    df = load_data()

    all_metrics   = []
    garch_results = {}
    for currency in ["USD", "EUR"]:
        result = run_currency(df, currency, quick=quick)
        all_metrics.extend(result["metrics"])
        garch_results[currency] = result["garch_result"]

    save_garch_params(garch_results)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(PROCESSED_DIR, "metrics_S2.csv"), index=False)

    print("\n" + "=" * 60)
    print("  TABLEAU COMPARATIF FINAL — S2")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print(f"\n✅ Pipeline S2 terminé ({mode_str})")

    from ..consolidate import consolidate_all
    consolidate_all()

    return metrics_df


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    run_pipeline(quick=quick)