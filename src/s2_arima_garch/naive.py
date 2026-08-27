# =============================================================
#  src/s2_arima_garch/naive.py
#  Benchmark naïf (random walk)
#
#  Test rapide : python -m src.s2_arima_garch.naive
# =============================================================

import pandas as pd
from .config import TEST_START, TEST_END


def benchmark_naive(train: pd.Series, test: pd.Series, horizon: int) -> pd.Series:
    """
    Benchmark naïf : prévision = dernier cours connu à horizon donné.
    pred[t] = actual[t - horizon]

    C'est la référence minimale (random walk hypothesis).
    """
    full_series = pd.concat([train, test])
    preds = full_series.shift(horizon).loc[TEST_START:TEST_END]
    preds.name = f"Naif_J{horizon}"
    return preds


# -------------------------------------------------------------
#  Test rapide
# -------------------------------------------------------------
if __name__ == "__main__":
    from .load import load_data, get_series, split_train_test

    df = load_data()
    spot, _ = get_series(df, "USD")
    train, test = split_train_test(spot)

    naive_j1 = benchmark_naive(train, test, horizon=1)
    naive_j7 = benchmark_naive(train, test, horizon=7)
    naive_j30 = benchmark_naive(train, test, horizon=30)

    print(f"\nNaïf J+1 : {len(naive_j1)} prévisions")
    print(naive_j1.head(5))
    print(f"\nNaïf J+7 : {len(naive_j7)} prévisions")
    print(naive_j7.head(5))
    print(f"\nNaïf J+30 : {len(naive_j30)} prévisions")
    print(naive_j30.head(5))