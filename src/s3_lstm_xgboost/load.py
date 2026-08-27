# =============================================================
#  src/s3_lstm_xgboost/load.py
#  Chargement + Features Engineering — S3
#  Auteur : Youssef Neji | MINDS ENIT | 2025
#
#  Test rapide : python -m src.s3_lstm_xgboost.load
# =============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import os

from .config import (
    PROCESSED_DIR, TRAIN_START, TRAIN_END,
    TEST_START, TEST_END,
    LSTM_CONFIG_BASE, LAG_FEATURES
)

# Sélection de variables exogènes propre à S3 (LSTM/MLP),
# DÉCOUPLÉE d'ARIMAX (src/s2_arima_garch/arimax.py) — ARIMAX est
# un modèle exclu de la production (validation croisée + DM test),
# son historique de sélection de variables n'a plus vocation à
# servir de référence pour LSTM/MLP. Voir exog_features.py pour la
# sélection actuelle, et feature_selection.py pour la méthodologie
# de re-sélection data-driven (corrélation + importance RF,
# validée par RMSE MLP réel sur fenêtre de validation interne).
from .exog_features import get_exog_features_s3


# =============================================================
#  1. CHARGEMENT
# =============================================================

def load_data() -> pd.DataFrame:
    """Charge le dataset final produit par data.py."""
    path = os.path.join(PROCESSED_DIR, "dataset_final.csv")
    df   = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    print(f"✅ Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"   Période : {df.index[0].date()} → {df.index[-1].date()}")
    return df


# =============================================================
#  2. FEATURES ENGINEERING
# =============================================================

def build_features(df: pd.DataFrame,
                   currency: str,
                   horizon: int,
                   macro_override: list = None) -> pd.DataFrame:
    """
    Construit le dataset enrichi de features pour une devise ET
    un horizon donnés.

    Les variables macro sont sélectionnées par
    get_exog_features_s3(horizon, currency) — sélection propre à
    S3, indépendante d'ARIMAX (voir exog_features.py).

    Args:
        macro_override : si fourni, remplace la sélection par
            défaut par cette liste explicite de colonnes macro.
            Permet à feature_selection.py de tester des jeux de
            variables candidats sans dupliquer toute la logique
            de construction de features (lags, moyennes mobiles,
            volatilité, momentum, dummies de régime, sigma GARCH
            restent identiques quel que soit le candidat testé).
    """
    col_spot   = f"TND_{currency}"
    col_logret = f"LogRet_TND_{currency}"

    print(f"\n🔧 Features engineering — TND/{currency} — J+{horizon}...")

    feat = pd.DataFrame(index=df.index)

    # Cible
    feat[col_spot]   = df[col_spot]
    feat[col_logret] = df[col_logret]

    # Lags cours spot
    for lag in LAG_FEATURES:
        feat[f"{col_spot}_lag{lag}"] = df[col_spot].shift(lag)

    # Lags log-rendements
    for lag in [1, 2, 3, 5]:
        feat[f"{col_logret}_lag{lag}"] = df[col_logret].shift(lag)

    # Moyennes mobiles
    for window in [5, 10, 20]:
        feat[f"{col_spot}_ma{window}"] = (
            df[col_spot].rolling(window).mean()
        )

    # Volatilité glissante
    for window in [5, 10, 20]:
        feat[f"vol_{currency}_{window}d"] = (
            df[col_logret].rolling(window).std()
        )

    # Momentum
    for lag in [5, 10, 20]:
        feat[f"momentum_{currency}_{lag}d"] = (
            df[col_spot] / df[col_spot].shift(lag) - 1
        )

    # Variables macro — sélectionnées par devise ET par horizon
    # (get_exog_features_s3, propre à S3), sauf si macro_override
    # est fourni (utilisé par feature_selection.py pour tester des
    # candidats). Inclut désormais les variables BCT complémentaires
    # (reserves_mom_pct, bct_market_share_pct) sur un pied d'égalité
    # avec les autres macro — plus de bloc "toujours ajouté" séparé,
    # elles sont pleinement soumises à la sélection/validation comme
    # n'importe quelle autre variable.
    macro_cols_wanted = (
        macro_override if macro_override is not None
        else get_exog_features_s3(horizon, currency)
    )
    macro_cols = [c for c in macro_cols_wanted if c in df.columns]
    missing = [c for c in macro_cols_wanted if c not in df.columns]
    if missing:
        print(f"   ⚠️  Variables macro manquantes : {missing}")
    print(f"   Variables macro (TND/{currency}, J+{horizon}) : {macro_cols}")

    for col in macro_cols:
        feat[col] = df[col]
        feat[f"{col}_lag1"] = df[col].shift(1)

        # Variation 5 jours pour les variables journalières
        # (pas pour les taux/indices lents type TauxFed, IPC,
        # TauxMMBCT, ni pour les variables BCT complémentaires,
        # mensuelles par nature — leur variation à 5 jours est
        # peu informative)
        if col in ["Brent_USD", "VIX", "EURUSD",
                   "DXY", "Gold_USD", "US10Y", "EUR10Y"]:
            feat[f"{col}_ret5"] = df[col].pct_change(5)

    # IPC Tunisie si disponible (2015+)
    if ("IPC_Tunisie" in df.columns and
            TRAIN_START >= "2015-01-01"):
        feat["IPC_Tunisie"]      = df["IPC_Tunisie"]
        feat["IPC_Tunisie_lag1"] = df["IPC_Tunisie"].shift(1)

    # Sigma GARCH
    sigma_path = os.path.join(
        PROCESSED_DIR,
        f"garch_sigma_{currency.lower()}.csv"
    )
    if os.path.exists(sigma_path):
        sigma = pd.read_csv(
            sigma_path, index_col=0, parse_dates=True
        ).squeeze()
        sigma.name = f"sigma_garch_{currency}"
        feat = feat.join(sigma, how="left")

        # FIX : sigma_garch ne couvre que jusqu'à TEST_END
        # (borne du pipeline d'évaluation S2, indépendante des
        # mises à jour du dataset brut). Au-delà, on prolonge la
        # dernière volatilité connue plutôt que de perdre ces
        # lignes au dropna() — cohérent avec la persistance
        # extrême de GARCH (α+β≈0.996/0.978) déjà établie, et
        # avec la convention de persistance utilisée ailleurs
        # dans le projet (exogènes ARIMAX, cascade MLP).
        n_before_ffill = feat[f"sigma_garch_{currency}"].isna().sum()
        feat[f"sigma_garch_{currency}"] = (
            feat[f"sigma_garch_{currency}"].ffill()
        )
        n_filled = n_before_ffill - feat[f"sigma_garch_{currency}"].isna().sum()
        if n_filled > 0:
            print(f"   ⚠️  sigma_garch prolongée (persistance) "
                  f"sur {n_filled} jour(s) au-delà de la "
                  f"couverture GARCH (borne TEST_END)")
        print(f"   ✅ Sigma GARCH intégré")

    # Saisonnalité
    feat["month"]     = df.index.month
    feat["dayofweek"] = df.index.dayofweek
    feat["quarter"]   = df.index.quarter

    # Différentiels de taux — ajoutés UNIQUEMENT si le taux
    # correspondant fait partie des variables macro sélectionnées
    # pour cet horizon (cohérence horizon-devise-variable stricte ;
    # avant ce fix, ces différentiels étaient ajoutés indépendamment
    # de la sélection macro, réintroduisant Fed/BCE "par la bande"
    # même aux horizons où la sélection les exclut).
    if "TauxFed_USD" in macro_cols and "TauxMMBCT" in df.columns:
        feat["diff_Fed_BCT"] = df["TauxFed_USD"] - df["TauxMMBCT"]

    if "TauxBCE_EUR" in macro_cols and "TauxMMBCT" in df.columns:
        feat["diff_BCE_BCT"] = df["TauxBCE_EUR"] - df["TauxMMBCT"]

    # ── DUMMIES DE RÉGIME ──────────────────────────────────
    # Régime 1 : quasi-fixité BCT (2004–2011)
    feat["regime_fixe"] = 0
    # Régime 2 : transition post-révolution (2012–2015)
    feat["regime_transition"] = 0
    # Régime 3 : dépréciation accélérée (2016–présent)
    feat["regime_deprec"] = 0

    feat.loc["2004-01-01":"2011-12-31",
             "regime_fixe"] = 1
    feat.loc["2012-01-01":"2015-12-31",
             "regime_transition"] = 1
    feat.loc["2016-01-01":,
             "regime_deprec"] = 1

    print(f"   ✅ Dummies de régime ajoutés "
          f"(fixe/transition/deprec)")
    # ────────────────────────────────────────────────────────
    # Supprimer NaN
    n_before = len(feat)
    feat = feat.dropna()
    print(f"   Lignes après dropna : {len(feat)} "
          f"(supprimées : {n_before - len(feat)})")
    print(f"   Features totales : "
          f"{len([c for c in feat.columns if c not in [col_spot, col_logret]])}")

    return feat



def split_features(feat: pd.DataFrame, currency: str) -> tuple:
    """
    Découpe le dataset en train et test.
    Retourne (X_train, y_train, X_test, y_test, feature_names)
    """
    col_spot = f"TND_{currency}"

    train = feat.loc[TRAIN_START:TRAIN_END]
    test  = feat.loc[TEST_START:TEST_END]

    feature_cols = [c for c in feat.columns
                    if c not in [col_spot, f"LogRet_TND_{currency}"]]

    X_train = train[feature_cols]
    y_train = train[col_spot]
    X_test  = test[feature_cols]
    y_test  = test[col_spot]

    print(f"\n   Split train/test :")
    print(f"   Train : {len(X_train)} obs "
          f"({TRAIN_START} → {TRAIN_END})")
    print(f"   Test  : {len(X_test)} obs "
          f"({TEST_START} → {TEST_END})")
    print(f"   Features : {len(feature_cols)}")

    return X_train, y_train, X_test, y_test, feature_cols


# =============================================================
#  3. DATASET PYTORCH POUR LSTM
# =============================================================

class FXDataset(Dataset):
    """
    Dataset PyTorch pour séquences temporelles.

    Pour chaque index i, retourne :
      - X : séquence de shape (sequence_length, n_features)
      - y : valeur cible à l'horizon h (scalaire)

    Le LSTM apprend : étant donné les 30 derniers jours
    de cours + features → prédire le cours dans h jours.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 seq_len: int, horizon: int):
        self.X       = torch.FloatTensor(X)
        self.y       = torch.FloatTensor(y)
        self.seq_len = seq_len
        self.horizon = horizon
    def __len__(self):
        n = len(self.X) - self.seq_len - self.horizon + 1
        if n <= 0:
            raise ValueError(
                f"FXDataset : pas assez de données ({len(self.X)} obs) "
                f"pour seq_len={self.seq_len} + horizon={self.horizon}. "
                f"Il faut au moins {self.seq_len + self.horizon} observations."
            )
        return n

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        y_val = self.y[idx + self.seq_len + self.horizon - 1]
        return x_seq, y_val


def prepare_lstm_data(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test:  pd.DataFrame, y_test:  pd.Series,
    horizon: int
) -> tuple:
    """
    Prépare les données pour le LSTM.
    La cible est désormais le rendement log cumulé à horizon h :
        r_h[t] = ln(S[t] / S[t-h])
    et non plus le niveau brut — cohérent avec ARIMA/GARCH
    et nécessaire pour que la série cible soit stationnaire.
    """
    seq_len = LSTM_CONFIG_BASE["sequence_length"]
    batch   = LSTM_CONFIG_BASE["batch_size"]

    y_train_level = y_train.values
    y_test_level  = y_test.values

    y_train_ret = np.full(len(y_train_level), np.nan)
    y_train_ret[horizon:] = np.log(
        y_train_level[horizon:] / y_train_level[:-horizon]
    )
    y_test_ret = np.full(len(y_test_level), np.nan)
    y_test_ret[horizon:] = np.log(
        y_test_level[horizon:] / y_test_level[:-horizon]
    )

    scaler_X   = MinMaxScaler()
    X_train_sc = scaler_X.fit_transform(X_train.values)
    X_test_sc  = scaler_X.transform(X_test.values)

    scaler_y = MinMaxScaler()
    valid_train = ~np.isnan(y_train_ret)
    scaler_y.fit(y_train_ret[valid_train].reshape(-1, 1))

    y_train_sc = scaler_y.transform(
        np.nan_to_num(y_train_ret).reshape(-1, 1)
    ).flatten()
    y_test_sc = scaler_y.transform(
        np.nan_to_num(y_test_ret).reshape(-1, 1)
    ).flatten()

    train_ds = FXDataset(X_train_sc, y_train_sc, seq_len, horizon)
    test_ds  = FXDataset(X_test_sc,  y_test_sc,  seq_len, horizon)

    train_loader = DataLoader(train_ds, batch_size=batch,
                              shuffle=False, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch,
                              shuffle=False, drop_last=False)

    print(f"\n   DataLoader prêt (horizon=J+{horizon}) :")
    print(f"   Train batches : {len(train_loader)}")
    print(f"   Test  batches : {len(test_loader)}")

    return (train_loader, test_loader,
            scaler_X, scaler_y,
            X_test_sc, y_test_level)

# =============================================================
#  TEST RAPIDE
# =============================================================

if __name__ == "__main__":
    df   = load_data()
    feat = build_features(df, "USD", horizon=1)

    print("\n--- Aperçu features ---")
    print(feat.head(3).to_string())
    print(f"\nShape features : {feat.shape}")

    X_train, y_train, X_test, y_test, feat_cols = \
        split_features(feat, "USD")

    print("\n--- Préparation LSTM J+1 ---")
    result = prepare_lstm_data(
        X_train, y_train, X_test, y_test, horizon=1
    )
    train_loader, test_loader, scaler_X, scaler_y, _, _ = result

    print("\n--- Vérification batch ---")
    for x_batch, y_batch in train_loader:
        print(f"   X batch shape : {x_batch.shape}")
        print(f"   y batch shape : {y_batch.shape}")
        break

    print("\n✅ load.py — OK")