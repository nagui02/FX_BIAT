# =============================================================
#  src/s2_arima_garch/load.py
#  Chargement des données + split train/test
#
#  Test rapide : python -m src.s2_arima_garch.load
# =============================================================

import pandas as pd
from .config import DATASET_PATH, TRAIN_START, TRAIN_END, TEST_START, TEST_END


def load_data() -> pd.DataFrame:
    """Charge le dataset final produit par data.py."""
    df = pd.read_csv(DATASET_PATH, parse_dates=["Date"], index_col="Date")
    print(f"✅ Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"   Période : {df.index[0].date()} → {df.index[-1].date()}")
    return df


def split_train_test(series: pd.Series) -> tuple:
    """Découpe une série en train (2004-2022) et test (2023-2025)."""
    train = series.loc[TRAIN_START:TRAIN_END]
    test  = series.loc[TEST_START:TEST_END]
    print(f"   Train : {len(train)} obs ({TRAIN_START} → {TRAIN_END})")
    print(f"   Test  : {len(test)} obs ({TEST_START} → {TEST_END})")
    return train, test


def get_series(df: pd.DataFrame, currency: str) -> tuple:
    """
    Extrait le cours spot et les log-rendements pour une devise.
    Args:
        currency : "USD" ou "EUR"
    Returns:
        (spot, logret)
    """
    spot   = df[f"TND_{currency}"]
    logret = df[f"LogRet_TND_{currency}"].dropna()
    return spot, logret


# -------------------------------------------------------------
#  Test rapide
# -------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    print("\nColonnes :", list(df.columns))

    spot, logret = get_series(df, "USD")
    print(f"\nTND_USD : {len(spot)} obs")
    print(f"LogRet_TND_USD : {len(logret)} obs")

    train, test = split_train_test(spot)
    print(f"\nAperçu train :\n{train.head(3)}")
    print(f"\nAperçu test :\n{test.head(3)}")