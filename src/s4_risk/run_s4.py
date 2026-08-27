# =============================================================
#  src/s4_risk/run_s4.py
#  Orchestrateur S4 — Risque de change
#  Usage : python -m src.s4_risk.run_s4
# =============================================================

import os
import pandas as pd

from .config import PROCESSED_DIR, CLIENT_PROFILES, CONFIDENCE_LEVELS
from .var import run_all_profiles, portfolio_var_multi_currency
from .stress import (
    run_stress_grid, run_historical_scenarios, run_backtesting
)


def run_pipeline():
    print("=" * 60)
    print("  PIPELINE S4 — RISQUE DE CHANGE")
    print("=" * 60)

    print("\n📊 VaR + CVaR (3 méthodes × 2 niveaux de confiance)...")
    var_df = run_all_profiles()
    out = os.path.join(PROCESSED_DIR, "var_results.csv")
    var_df.to_csv(out, index=False)
    print(f"   ✅ {out} ({len(var_df)} lignes)")
    print(var_df.to_string(index=False))

    print("\n📊 VaR portefeuille Desk BIAT (multi-devises)...")
    for alpha in CONFIDENCE_LEVELS:
        r = portfolio_var_multi_currency("Desk BIAT", alpha)
        print(f"   Confiance {alpha*100:.0f}% : VaR diversifiée = "
              f"{r['VaR_diversifie']:,.0f} TND "
              f"(bénéfice diversification : {r['benefice_pct']:.1f}%)")

    print("\n⚠️  Stress testing (chocs 10%/20%/30%)...")
    stress_df = run_stress_grid()
    out2 = os.path.join(PROCESSED_DIR, "stress_results.csv")
    stress_df.to_csv(out2, index=False)
    print(f"   ✅ {out2} ({len(stress_df)} lignes)")

    print("\n📉 Scénarios historiques (2018, COVID, Ukraine)...")
    hist_df = run_historical_scenarios()
    out3 = os.path.join(PROCESSED_DIR, "historical_scenarios.csv")
    hist_df.to_csv(out3, index=False)
    print(f"   ✅ {out3} ({len(hist_df)} lignes)")
    print(hist_df.to_string(index=False))

    print("\n🔬 Backtesting VaR (test de Kupiec)...")
    backtest_df = run_backtesting()
    out4 = os.path.join(PROCESSED_DIR, "backtest_kupiec.csv")
    backtest_df.to_csv(out4, index=False)
    print(f"   ✅ {out4}")
    print(backtest_df.to_string(index=False))

    print("\n✅ Pipeline S4 terminé")


if __name__ == "__main__":
    run_pipeline()