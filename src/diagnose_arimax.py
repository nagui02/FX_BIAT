# diagnose_arimax.py
import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/dataset_final.csv",
                 parse_dates=["Date"], index_col="Date")

cols_check = ["VIX", "EURUSD", "Brent_USD", "DXY"]
print("=== Qualité des variables exogènes ARIMAX J+1 USD ===\n")

for col in cols_check:
    s = df[col]
    n_nan = s.isna().sum()
    n_zero = (s == 0).sum()
    recent = s.tail(30)
    print(f"{col:12} | NaN: {n_nan:4} | "
          f"zéros: {n_zero:3} | "
          f"min={s.min():.4f} max={s.max():.4f} | "
          f"30 derniers jours: min={recent.min():.4f} "
          f"max={recent.max():.4f} std={recent.std():.4f}")

print("\n=== Dernières 15 valeurs DXY (suspect n°1) ===")
print(df["DXY"].tail(15))

print("\n=== Comparaison prédictions brutes vs consolidées ===")
raw = pd.read_csv("data/processed/arimax_predictions_usd.csv",
                  index_col=0, parse_dates=True)
cons = pd.read_csv("data/processed/predictions_usd.csv",
                   index_col=0, parse_dates=True)

col = "ARIMAX_J1"
if col in raw.columns and col in cons.columns:
    common = raw.index.intersection(cons.index)
    diff = (raw.loc[common, col] - cons.loc[common, col]).abs()
    print(f"Écart max brut vs consolidé : {diff.max():.5f}")
    print(f"Nb écarts > 0.001 : {(diff > 0.001).sum()} / {len(common)}")