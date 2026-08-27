# =============================================================
#  src/s3_lstm_xgboost/run_s3.py
#  Orchestrateur S3 — LSTM + MLP
#  Usage : python -m src.s3_lstm_xgboost.run_s3
#         python -m src.s3_lstm_xgboost.run_s3 --quick
# =============================================================

import sys

from .lstm import run_pipeline as run_lstm_pipeline
from .mlp import run_pipeline as run_mlp_pipeline


def run_pipeline(quick: bool = False):
    print("=" * 60)
    print("  PIPELINE S3 — LSTM + MLP")
    print("=" * 60)

    print("\n[1/2] LSTM...")
    run_lstm_pipeline(quick=quick)

    print("\n[2/2] MLP...")
    run_mlp_pipeline(quick=quick)

    print("\n✅ Pipeline S3 (LSTM + MLP) terminé")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    run_pipeline(quick=quick)