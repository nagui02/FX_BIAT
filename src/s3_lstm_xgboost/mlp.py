# =============================================================
#  src/s3_lstm_xgboost/mlp.py
#  MLP (feedforward) — alternative à LSTM (Option E)
#  Auteur : Youssef Neji | MINDS ENIT | 2025-2026
#
#  Teste l'hypothèse : la structure séquentielle du LSTM apporte-
#  t-elle une vraie valeur ajoutée, sachant que les features
#  contiennent déjà des lags/moyennes mobiles/momentum engineered
#  (information temporelle explicite) ? Un simple MLP sur les
#  mêmes features, sans mémoire séquentielle, sert de test direct
#  de cette hypothèse.
#
#  Même cible que LSTM : rendement log cumulé à horizon h,
#  reconstruction du niveau via S_pred = S_t * exp(r_pred).
#  Même principe de non-lookahead (aucune fuite d'information
#  future) — S_t est le niveau connu AUJOURD'HUI, pas le futur.
#
#  python -m src.s3_lstm_xgboost.mlp --quick
#  python -m src.s3_lstm_xgboost.mlp
# =============================================================

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

from .config import (
    PROCESSED_DIR, FIGURES_DIR,
    TEST_START, TEST_END, HORIZONS, SEED
)
from .load import load_data, build_features, split_features
from .metrics import compute_metrics


# =============================================================
#  1. CONFIGURATION MLP PAR HORIZON
# =============================================================
MLP_CONFIG_BASE = {
    "batch_size":    32,
    "learning_rate": 1e-3,
    "epochs":        100,
    "patience":      10,
}

# Capacité réduite aux horizons longs, même logique que pour LSTM
MLP_CONFIG_BY_HORIZON = {
    1:  {"hidden_sizes": [64, 32], "dropout": 0.20, "weight_decay": 1e-5},
    7:  {"hidden_sizes": [32, 16], "dropout": 0.30, "weight_decay": 1e-4},
    30: {"hidden_sizes": [16],     "dropout": 0.35, "weight_decay": 1e-4},
}


def get_mlp_config(horizon: int) -> dict:
    cfg = dict(MLP_CONFIG_BASE)
    cfg.update(MLP_CONFIG_BY_HORIZON.get(horizon, MLP_CONFIG_BY_HORIZON[1]))
    return cfg


# =============================================================
#  2. ARCHITECTURE MLP
# =============================================================
class MLPModel(nn.Module):
    """
    Réseau feedforward simple : Linear -> ReLU -> Dropout, empilé
    selon hidden_sizes, puis couche de sortie scalaire.

    Contrairement au LSTM, ne reçoit PAS de séquence de 30 jours :
    un seul vecteur de features à la date t (incluant déjà lags,
    moyennes mobiles, momentum) prédit directement le rendement
    log cumulé à horizon h.
    """
    def __init__(self, input_size: int, hidden_sizes: list, dropout: float):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =============================================================
#  3. CIBLE TABULAIRE — rendement log cumulé à horizon h
# =============================================================
def build_horizon_target(level_full: pd.Series, horizon: int) -> pd.Series:
    """
    target_t = ln(S_{t+h} / S_t), aligné sur l'index t (aujourd'hui).

    Utilise la série de niveau complète (train+test concaténés)
    pour un décalage correct à la frontière train/test — sinon
    les dernières lignes du train perdraient leur cible à tort.
    NaN sur les dernières `horizon` dates (futur non observable).
    """
    future = level_full.shift(-horizon)
    return np.log(future / level_full)


# =============================================================
#  4. PRÉPARATION DES DONNÉES (tabulaire, sans séquence)
# =============================================================
def prepare_mlp_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    level_full: pd.Series, horizon: int, batch_size: int
) -> tuple:
    """
    X = features à la date t (déjà construites par
        load.build_features / split_features — identiques à LSTM)
    y = rendement log cumulé de t à t+h (build_horizon_target)

    Aucune séquence : chaque ligne est un exemple indépendant.

    FIX (Vague 3) : scaler_y était un MinMaxScaler, sensible aux
    valeurs extrêmes historiques (ex : rendement 30j USD lors de
    la crise Lehman, sept-nov 2008, à ~5σ de la distribution
    normale). Un seul jour extrême calibrait toute la plage de
    normalisation, compressant la dynamique "normale" 2024-2026
    dans une fraction de l'espace [0,1] — dégradation sévère
    observée spécifiquement sur USD J+30 (RMSE +60% vs Naïf).
    RobustScaler centre sur la médiane et normalise par l'IQR,
    insensible à ces outliers ponctuels.
    """
    y_full_ret = build_horizon_target(level_full, horizon)

    y_train_ret = y_full_ret.loc[X_train.index]
    y_test_ret  = y_full_ret.loc[X_test.index]

    # Retirer les lignes sans cible valide (dernières `horizon`
    # dates du train et du test, où le futur n'est pas disponible)
    train_mask = y_train_ret.notna()
    test_mask  = y_test_ret.notna()

    X_train_v, y_train_v = X_train.loc[train_mask], y_train_ret.loc[train_mask]
    X_test_v              = X_test.loc[test_mask]

    scaler_X   = MinMaxScaler()
    X_train_sc = scaler_X.fit_transform(X_train_v.values)
    X_test_sc  = scaler_X.transform(X_test_v.values)

    # FIX : RobustScaler au lieu de MinMaxScaler pour la cible —
    # voir docstring ci-dessus.
    scaler_y   = RobustScaler()
    y_train_sc = scaler_y.fit_transform(
        y_train_v.values.reshape(-1, 1)
    ).flatten()

    train_ds = TensorDataset(
        torch.FloatTensor(X_train_sc), torch.FloatTensor(y_train_sc)
    )
    # FIX (Vague 6, Finding 8) : shuffle=False — ce DataLoader
    # externe n'est jamais itéré directement. train_mlp() extrait
    # les tenseurs bruts via .dataset.tensors puis mélange
    # lui-même via tr_loader, correctement, plus loin. shuffle=True
    # ici était trompeur, sans effet réel sur l'entraînement.
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=False, drop_last=True)

    # S_ref : niveau connu à la date t (aujourd'hui), pour
    # reconstruire le niveau prédit après inverse-transform
    S_ref = level_full.loc[X_test_v.index]

    # Dates CIBLES (t+h) — la prévision porte sur t+h, pas sur t.
    # Même convention que Naïf/ARIMA/LSTM : l'index d'une série de
    # prévisions est la date PRÉDITE, pas la date des features
    # d'entrée. Dérivées par décalage positionnel dans level_full
    # (cohérent avec build_horizon_target qui utilise .shift(-horizon)).
    pos = level_full.index.get_indexer(X_test_v.index)
    target_dates = level_full.index[pos + horizon]

    return (train_loader, scaler_X, scaler_y, X_test_sc,
            target_dates, S_ref)


# =============================================================
#  5. ENTRAÎNEMENT
# =============================================================
def train_mlp(model: MLPModel, train_loader: DataLoader,
              mlp_cfg: dict, quick: bool = False) -> dict:
    """
    Entraîne le MLP avec un split interne train/validation
    chronologique (80/20, sans shuffle sur la validation pour
    éviter toute fuite temporelle), early stopping et
    ReduceLROnPlateau — même logique que pour le LSTM.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device : {device}")
    model = model.to(device)

    epochs   = 5 if quick else mlp_cfg["epochs"]
    patience = 3 if quick else mlp_cfg["patience"]

    optimizer = Adam(model.parameters(), lr=mlp_cfg["learning_rate"],
                     weight_decay=mlp_cfg["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  factor=0.5, patience=5)
    criterion = nn.MSELoss()

    full_X, full_y = train_loader.dataset.tensors
    n     = len(full_X)
    n_val = max(1, int(0.2 * n))
    n_tr  = n - n_val

    X_tr, y_tr   = full_X[:n_tr], full_y[:n_tr]
    X_val, y_val = full_X[n_tr:], full_y[n_tr:]

    tr_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=train_loader.batch_size,
        shuffle=True, drop_last=True
    )

    best_val, best_state, patience_counter = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = criterion(
                model(X_val.to(device)), y_val.to(device)
            ).item()

        train_loss = float(np.mean(train_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"   Epoch {epoch+1:>3}/{epochs} | Train={train_loss:.6f} | "
                  f"Val={val_loss:.6f} | LR={lr:.6f} | "
                  f"Patience={patience_counter}/{patience}")

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   ⏹️  Early stopping à l'epoch {epoch+1} "
                      f"(best val_loss={best_val:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"   ✅ Meilleur modèle restauré (val_loss={best_val:.6f})")

    return history


# =============================================================
#  6. PRÉVISION + RECONSTRUCTION DU NIVEAU
# =============================================================
def predict_mlp(model: MLPModel, X_test_sc: np.ndarray,
                target_dates: pd.DatetimeIndex, S_ref: pd.Series,
                scaler_y, horizon: int) -> pd.Series:
    """
    Prédit le rendement log cumulé puis reconstruit le niveau :
    S_pred[t+h] = S_t * exp(r_pred[t]), où S_t (S_ref) est le
    niveau observé à la date t — aucune fuite d'information.

    La série retournée est indexée sur t+h (la date PRÉDITE),
    pas sur t — cohérent avec la convention Naïf/ARIMA/LSTM.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    with torch.no_grad():
        x = torch.FloatTensor(X_test_sc).to(device)
        pred_scaled = model(x).cpu().numpy().reshape(-1, 1)

    pred_ret     = scaler_y.inverse_transform(pred_scaled).flatten()
    preds_level  = S_ref.values * np.exp(pred_ret)

    pred_series = pd.Series(preds_level, index=target_dates,
                            name=f"MLP_J{horizon}")
    print(f"   ✅ {len(pred_series)} prévisions MLP J+{horizon} générées")
    print(f"   Période : {pred_series.index[0].date()} → "
          f"{pred_series.index[-1].date()}")
    return pred_series


# =============================================================
#  7. VISUALISATION
# =============================================================
def plot_mlp_forecast(actual, predictions, currency, horizon):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual.index, actual.values, label="Réel",
            color="black", linewidth=1.5)
    colors = {"Naïf": "gray", "ARIMA": "blue", "MLP": "green"}
    for name, pred in predictions.items():
        common = actual.index.intersection(pred.index)
        ax.plot(common, pred.loc[common].values, label=name,
                linestyle="--", color=colors.get(name, "red"), linewidth=1.2)
    ax.set_title(f"Prévisions TND/{currency} — Horizon J+{horizon} "
                 f"| Test {TEST_START} → {TEST_END}", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"TND/{currency}")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fname = os.path.join(FIGURES_DIR, f"mlp_forecast_{currency.lower()}_J{horizon}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 {fname}")


def plot_learning_curve_mlp(history, currency, horizon):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="Train")
    ax.plot(history["val_loss"], label="Validation")
    ax.set_title(f"Courbe d'apprentissage MLP TND/{currency} J+{horizon}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (rendement normalisé)")
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(FIGURES_DIR, f"mlp_learning_{currency.lower()}_J{horizon}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 {fname}")


# =============================================================
#  8. PIPELINE PAR DEVISE
# =============================================================
def run_mlp_currency(df: pd.DataFrame, currency: str, quick: bool = False) -> dict:
    print(f"\n{'='*60}\n  MLP — TND/{currency}\n{'='*60}")

    # FIX (Vague 6, Finding 3) : seed pour reproductibilité.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    arima_path = os.path.join(
        PROCESSED_DIR, f"arima_predictions_{currency.lower()}.csv"
    )
    arima_preds = None
    if os.path.exists(arima_path):
        arima_preds = pd.read_csv(arima_path, index_col=0, parse_dates=True)
        print(f"   ✅ Prévisions ARIMA/Naïf chargées depuis S2")
    else:
        print(f"   ⚠️  {arima_path} introuvable — comparaison ARIMA/Naïf ignorée")

    metrics     = []
    predictions = {}

    for horizon in HORIZONS:
        print(f"\n--- Horizon J+{horizon} ---")

        # Features + split RECALCULÉS pour CET horizon — même
        # sélection de variables macro que pour ARIMAX/LSTM
        # (get_exog_features par devise et par horizon)
        feat = build_features(df, currency, horizon)
        X_train, y_train, X_test, y_test, feat_cols = \
            split_features(feat, currency)
        print(f"   Features : {len(feat_cols)}")

        level_full = pd.concat([y_train, y_test]).sort_index()

        mlp_cfg = get_mlp_config(horizon)
        print(f"   Config J+{horizon} : hidden_sizes={mlp_cfg['hidden_sizes']} | "
              f"dropout={mlp_cfg['dropout']} | weight_decay={mlp_cfg['weight_decay']}")

        (train_loader, scaler_X, scaler_y,
         X_test_sc, target_dates, S_ref) = prepare_mlp_data(
            X_train, X_test, level_full, horizon, mlp_cfg["batch_size"]
        )

        model = MLPModel(input_size=len(feat_cols),
                         hidden_sizes=mlp_cfg["hidden_sizes"],
                         dropout=mlp_cfg["dropout"])
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Paramètres MLP : {n_params:,}")

        history = train_mlp(model, train_loader, mlp_cfg, quick=quick)
        if not quick:
            plot_learning_curve_mlp(history, currency, horizon)

        pred_mlp = predict_mlp(model, X_test_sc, target_dates, S_ref,
                               scaler_y, horizon)
        predictions[f"MLP_J{horizon}"] = pred_mlp

        # Comparaison contre la vraie valeur à la date CIBLE (t+h),
        # pas contre y_test.loc[test_index] qui serait la valeur
        # d'aujourd'hui (t) — c'était le bug.
        actual_test = level_full.loc[target_dates]
        m_mlp = compute_metrics(actual_test, pred_mlp, "MLP", currency, horizon)
        metrics.append(m_mlp)
        print(f"\n   MLP J+{horizon} : RMSE={m_mlp['RMSE']} | "
              f"MAE={m_mlp['MAE']} | MAPE={m_mlp['MAPE (%)']}%")

        if arima_preds is not None:
            naive_col, arima_col = f"Naif_J{horizon}", f"ARIMA_J{horizon}"
            if naive_col in arima_preds.columns and arima_col in arima_preds.columns:
                naive_s = arima_preds[naive_col]
                arima_s = arima_preds[arima_col]
                m_naive = compute_metrics(actual_test, naive_s, "Naïf", currency, horizon)
                m_arima = compute_metrics(actual_test, arima_s, "ARIMA", currency, horizon)
                metrics.extend([m_naive, m_arima])
                print(f"   Naïf  J+{horizon} : RMSE={m_naive['RMSE']}")
                print(f"   ARIMA J+{horizon} : RMSE={m_arima['RMSE']}")
                gain = (m_arima["RMSE"] - m_mlp["RMSE"]) / m_arima["RMSE"] * 100
                print(f"   📊 Gain MLP vs ARIMA : {gain:+.1f}%")
                if not quick:
                    plot_mlp_forecast(
                        actual_test,
                        {"Naïf": naive_s, "ARIMA": arima_s, "MLP": pred_mlp},
                        currency, horizon
                    )

    pred_df  = pd.DataFrame(predictions)
    out_path = os.path.join(PROCESSED_DIR, f"mlp_predictions_{currency.lower()}.csv")
    pred_df.to_csv(out_path)
    print(f"\n   💾 {out_path}")

    return {"metrics": metrics}


def run_pipeline(quick: bool = False):
    mode = "RAPIDE" if quick else "COMPLET"
    print("=" * 60)
    print(f"  PIPELINE MLP (Option E) — Mode {mode}")
    print("=" * 60)

    df = load_data()
    all_metrics = []
    for currency in ["USD", "EUR"]:
        result = run_mlp_currency(df, currency, quick=quick)
        all_metrics.extend(result["metrics"])

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(PROCESSED_DIR, "metrics_mlp.csv"), index=False)

    print("\n" + "=" * 60)
    print("  TABLEAU COMPARATIF FINAL — MLP")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print(f"\n✅ Pipeline MLP terminé ({mode})")
    return metrics_df


# =============================================================
#  9. ENTRAÎNEMENT FINAL (production) — réutilise l'architecture
#     et la procédure DÉJÀ VALIDÉES ci-dessus, sans les dupliquer
#     ni les réécrire.
#
#     Utilisé par train_final_models.py (inférence instantanée)
#     et predict_august.py (prévisions Août 2026) — élimine la
#     divergence précédemment identifiée entre le modèle
#     benchmarké (ci-dessus) et le modèle déployé.
# =============================================================

def prepare_mlp_data_final(
    X_all: pd.DataFrame, level_full: pd.Series,
    horizon: int, batch_size: int
) -> tuple:
    """
    Version 'production' de prepare_mlp_data : pas de split
    train/test — toutes les données disponibles jusqu'à la
    coupure servent à l'entraînement. train_mlp() se charge déjà
    de découper une validation interne (80/20 chronologique) à
    partir de ce qui lui est transmis — réutilisé tel quel.

    FIX (Vague 3) : RobustScaler au lieu de MinMaxScaler pour la
    cible, même correction que prepare_mlp_data — voir sa
    docstring pour le diagnostic complet (sensibilité aux
    rendements extrêmes de crise, ex. USD sept-nov 2008).
    """
    y_full_ret = build_horizon_target(level_full, horizon)
    y_ret = y_full_ret.loc[X_all.index]

    mask = y_ret.notna()
    X_v, y_v = X_all.loc[mask], y_ret.loc[mask]

    scaler_X = MinMaxScaler()
    X_sc = scaler_X.fit_transform(X_v.values)

    # FIX : RobustScaler au lieu de MinMaxScaler pour la cible.
    scaler_y = RobustScaler()
    y_sc = scaler_y.fit_transform(
        y_v.values.reshape(-1, 1)
    ).flatten()

    train_ds = TensorDataset(
        torch.FloatTensor(X_sc), torch.FloatTensor(y_sc)
    )
    # FIX (Vague 6, Finding 8) : shuffle=False — voir note dans
    # prepare_mlp_data, même raison exactement.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=False, drop_last=True
    )

    return train_loader, scaler_X, scaler_y


def train_final_mlp(
    df: pd.DataFrame, currency: str,
    horizon: int, end_date: str = None
) -> dict:
    """
    Entraîne un MLP de production — MÊME architecture (MLPModel),
    MÊME procédure (train_mlp), MÊME cible (rendement log
    reconstruit via S_ref × exp(pred)), MÊME sélection de
    variables macro par horizon que le modèle benchmarké par
    Diebold-Mariano — mais sur toutes les données disponibles
    jusqu'à `end_date` (None = jusqu'à la dernière date dispo),
    sans test réservé puisqu'il s'agit d'un modèle de déploiement,
    pas d'évaluation.

    Returns:
        dict prêt à sauvegarder : state_dict du modèle, scalers,
        colonnes de features, dernière ligne de features (pour
        inférence immédiate à J+7/J+30), dernière date/valeur
        connues.
    """
    col        = f"TND_{currency}"
    col_logret = f"LogRet_TND_{currency}"

    feat = build_features(df, currency, horizon)
    if end_date is not None:
        feat = feat.loc[:end_date]

    feat_cols = [c for c in feat.columns
                 if c not in [col, col_logret]]
    X_all      = feat[feat_cols]
    level_full = feat[col]

    mlp_cfg = get_mlp_config(horizon)

    train_loader, scaler_X, scaler_y = prepare_mlp_data_final(
        X_all, level_full, horizon, mlp_cfg["batch_size"]
    )

    model = MLPModel(
        input_size=len(feat_cols),
        hidden_sizes=mlp_cfg["hidden_sizes"],
        dropout=mlp_cfg["dropout"]
    )

    print(f"   🔄 Entraînement MLP final TND/{currency} "
          f"J+{horizon} "
          f"(jusqu'à {feat.index[-1].date()})...")
    train_mlp(model, train_loader, mlp_cfg, quick=False)

    last_row    = X_all.iloc[-1:].values
    last_row_sc = scaler_X.transform(last_row)

    return {
        "model_state":  model.state_dict(),
        "scaler_X":     scaler_X,
        "scaler_y":     scaler_y,
        "feat_cols":    feat_cols,
        "hidden_sizes": mlp_cfg["hidden_sizes"],
        "dropout":      mlp_cfg["dropout"],
        "last_row_sc":  last_row_sc,
        "last_date":    feat.index[-1].strftime("%Y-%m-%d"),
        "last_value":   float(level_full.iloc[-1]),
    }


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    run_pipeline(quick=quick)
    from src.consolidate import consolidate_all
    consolidate_all()