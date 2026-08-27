# =============================================================
#  src/s2_arima_garch/run_arimax.py
#  Orchestrateur ARIMAX — USD + EUR, tous horizons de config.HORIZONS
#
#  python -m src.s2_arima_garch.run_arimax
#  python -m src.s2_arima_garch.run_arimax --quick
# =============================================================

import sys
import os
import pandas as pd

from .config import PROCESSED_DIR, HORIZONS, REFIT_BY_HORIZON
from .load import load_data
from .arimax import run_arimax_currency, set_available_columns


def run_pipeline(quick: bool = False) -> pd.DataFrame:
    mode_str = "RAPIDE (test mécanique)" if quick else "COMPLET"
    print("=" * 60)
    print(f"  PIPELINE ARIMAX — Mode {mode_str}")
    print(f"  Horizons : {HORIZONS}")
    print(f"  Refit par horizon :")
    for h in HORIZONS:
        print(f"    J+{h:<3} : refit_every={REFIT_BY_HORIZON.get(h, 1)}")
    print("=" * 60)

    df = load_data()

    # Indispensable : initialise les colonnes disponibles pour
    # que get_exog_features puisse vérifier les variables
    # optionnelles (DXY, US10Y, Gold_USD, EUR10Y).
    set_available_columns(df.columns.tolist())

    all_metrics = []
    for currency in ["USD", "EUR"]:
        result = run_arimax_currency(df, currency, quick=quick)
        all_metrics.extend(result["metrics"])

    metrics_df = pd.DataFrame(all_metrics)

    if not metrics_df.empty:
        order_map = {"ARIMAX": 0, "Naïf": 1, "ARIMA": 2}
        metrics_df["_ord"] = metrics_df["Modèle"].map(order_map)
        metrics_df = (
            metrics_df.sort_values(["Devise", "Horizon", "_ord"])
                      .drop(columns="_ord")
        )

        print("\n" + "=" * 60)
        print("  TABLEAU COMPARATIF FINAL — ARIMAX")
        print("=" * 60)
        print(metrics_df[
            ["Modèle", "Devise", "Horizon", "RMSE", "MAE", "MAPE (%)"]
        ].to_string(index=False))

    metrics_df.to_csv(
        os.path.join(PROCESSED_DIR, "metrics_arimax.csv"), index=False
    )
    print(f"\n   💾 metrics_arimax.csv sauvegardé")
    print(f"\n✅ Pipeline ARIMAX terminé ({mode_str})")

    from ..consolidate import consolidate_all
    consolidate_all()

    return metrics_df


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    run_pipeline(quick=quick)