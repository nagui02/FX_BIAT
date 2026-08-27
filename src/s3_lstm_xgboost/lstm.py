# =============================================================
#  src/s3_lstm_xgboost/lstm.py
#  Modèle LSTM PyTorch — Prédiction cours de change
#  Auteur : Youssef Neji | MINDS ENIT | 2025
#
#  Test rapide : python -m src.s3_lstm_xgboost.lstm --quick
#  Test complet : python -m src.s3_lstm_xgboost.lstm
# =============================================================

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config import (
    LSTM_CONFIG_BASE, get_lstm_config,
    PROCESSED_DIR, FIGURES_DIR,
    MODELS_DIR, TRAIN_START, TRAIN_END,
    TEST_START, TEST_END, HORIZONS, SEED
)
from .load import (
    load_data, build_features, split_features, prepare_lstm_data
)


# =============================================================
#  1. ARCHITECTURE LSTM
# =============================================================

# =============================================================
#  Architecture LSTM avec mécanisme d'attention
# =============================================================

class LSTMAttentionModel(nn.Module):
    """
    LSTM avec mécanisme d'attention multi-têtes simplifié.

    Architecture :
      1. Couche(s) LSTM → séquence d'états cachés h_1,...,h_T
      2. Couche d'attention → score d'importance α_i pour chaque pas
      3. Contexte = somme pondérée : c = Σ α_i * h_i
      4. Couche FC finale → prédiction scalaire

    L'attention résout le problème du LSTM standard qui
    traite tous les pas de temps également. Pour J+1,
    le modèle devrait apprendre à mettre plus de poids
    sur les derniers jours (hier > il y a 30 jours).

    Paramètres supplémentaires vs LSTM standard :
      hidden_size × 1 (couche d'attention) = 64 paramètres
      → coût négligeable vs les 61 761 paramètres totaux
    """

    def __init__(self, input_size: int,
                 hidden_size: int = 64,
                 num_layers:  int = 2,
                 dropout:     float = 0.2,
                 attention:   bool = True):
        super(LSTMAttentionModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.use_attention = attention

        # Couche LSTM
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True
        )

        # Couche d'attention (un score par pas de temps)
        if attention:
            self.attention_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1)
            )

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x : (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out : (batch, seq_len, hidden_size)

        if self.use_attention:
            # Score d'attention pour chaque pas de temps
            # scores : (batch, seq_len, 1)
            scores  = self.attention_layer(lstm_out)

            # Softmax sur la dimension temporelle
            # weights : (batch, seq_len, 1)
            weights = torch.softmax(scores, dim=1)

            # Contexte = somme pondérée des états cachés
            # context : (batch, hidden_size)
            context = (weights * lstm_out).sum(dim=1)

            out = self.dropout(context)

        else:
            # Sans attention : utiliser seulement le dernier état
            out = self.dropout(lstm_out[:, -1, :])

        return self.fc(out).squeeze(-1)

    def get_attention_weights(self, x):
        """
        Retourne les poids d'attention pour interprétation.
        Utile pour visualiser quels jours le modèle regarde.
        """
        self.eval()
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            scores      = self.attention_layer(lstm_out)
            weights     = torch.softmax(scores, dim=1)
        return weights.squeeze(-1).cpu().numpy()


# Alias pour compatibilité avec le reste du code
LSTMModel = LSTMAttentionModel

# =============================================================
#  2. ENTRAÎNEMENT
# =============================================================

def train_lstm(
    model:        LSTMModel,
    train_loader,
    test_loader,
    currency:     str,
    horizon:      int,
    lstm_cfg:     dict,
    epochs:       int = None,
    patience:     int = None,
    quick:        bool = False
) -> dict:
    """
    Entraîne le modèle LSTM avec early stopping.

    Optimiseur  : Adam avec weight_decay (régularisation L2,
                  valeur lue depuis lstm_cfg["weight_decay"] —
                  plus forte aux horizons longs pour compenser
                  la capacité réduite du modèle, voir config.py)
    Loss        : MSE (Mean Squared Error) sur valeurs normalisées
    Scheduler   : ReduceLROnPlateau (réduit lr si val_loss stagne)
    Early stop  : arrêt si val_loss ne s'améliore pas pendant
                  `patience` epochs

    Args:
        lstm_cfg : dict retourné par config.get_lstm_config(horizon),
                   contient hidden_size, num_layers, dropout,
                   weight_decay, learning_rate, epochs, patience.

    Returns:
        dict avec historique des losses train et validation
    """
    if quick:
        epochs  = 5
        patience = 3
    else:
        epochs  = epochs  or lstm_cfg["epochs"]
        patience = patience or lstm_cfg["patience"]

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"   Device : {device}")

    model = model.to(device)
    optimizer = Adam(
        model.parameters(),
        lr=lstm_cfg["learning_rate"],
        weight_decay=lstm_cfg.get("weight_decay", 0.0)
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=5
    )
    criterion = nn.MSELoss()

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0

    history = {"train_loss": [], "val_loss": []}

    print(f"\n   🔄 Entraînement LSTM "
          f"TND/{currency} J+{horizon} "
          f"({epochs} epochs max, patience={patience})...")

    for epoch in range(1, epochs + 1):

        # --- Train ---
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                preds   = model(x_batch)
                loss    = criterion(preds, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"   Epoch {epoch:>3}/{epochs} | "
                  f"Train={train_loss:.6f} | "
                  f"Val={val_loss:.6f} | "
                  f"LR={lr:.6f} | "
                  f"Patience={patience_count}/{patience}")

        if patience_count >= patience:
            print(f"\n   ⏹️  Early stopping à l'epoch {epoch} "
                  f"(best val_loss={best_val_loss:.6f})")
            break

    # Restaurer le meilleur état
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"   ✅ Meilleur modèle restauré "
              f"(val_loss={best_val_loss:.6f})")

    return history


# =============================================================
#  3. PRÉVISIONS
# =============================================================

def predict_lstm(
    model:        LSTMModel,
    X_test_sc:    np.ndarray,
    y_test_level: np.ndarray,
    y_test_index: pd.DatetimeIndex,
    scaler_y,
    seq_len:      int,
    horizon:      int
) -> pd.Series:
    """
    Génère les prévisions LSTM sur la période de test.
    Le modèle prédit un rendement log ; on reconstruit
    le niveau via S_pred[t] = S[t-h] * exp(r_pred).
    """
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model.eval()
    model.to(device)

    n = len(X_test_sc)
    n_preds = n - seq_len - horizon + 1

    # FIX (Vague 6, Finding 4) : prédiction batchée en un seul
    # passage forward, au lieu d'une boucle séquence par séquence
    # — même résultat, beaucoup plus rapide.
    predictions = []
    if n_preds > 0:
        sequences = np.stack([
            X_test_sc[i : i + seq_len] for i in range(n_preds)
        ])
        with torch.no_grad():
            x_batch = torch.FloatTensor(sequences).to(device)
            preds   = model(x_batch).cpu().numpy()
        predictions = list(preds)

    # Dénormalisation → rendement log prédit
    preds_array = np.array(predictions).reshape(-1, 1)
    pred_ret = scaler_y.inverse_transform(preds_array).flatten()

    # Reconstruction du niveau : S_ref = niveau à la fin de la
    # séquence d'entrée (dernier jour connu, i.e. t = i + seq_len - 1)
    S_ref = y_test_level[seq_len - 1 : seq_len - 1 + len(pred_ret)]
    preds_level = S_ref * np.exp(pred_ret)

    start_idx  = seq_len + horizon - 1
    pred_index = y_test_index[start_idx : start_idx + len(preds_level)]

    pred_series = pd.Series(
        preds_level[:len(pred_index)],
        index=pred_index,
        name=f"LSTM_J{horizon}"
    )

    print(f"   ✅ {len(pred_series)} prévisions LSTM J+{horizon} générées")
    print(f"   Période : {pred_series.index[0].date()} "
          f"→ {pred_series.index[-1].date()}")
    return pred_series

# =============================================================
#  4. COURBES D'APPRENTISSAGE
# =============================================================

def plot_learning_curves(history: dict, currency: str,
                         horizon: int):
    """Trace les courbes de perte train/validation."""
    fig, ax = plt.subplots(figsize=(10, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"],
            label="Train loss", color="#1E88E5", linewidth=1.5)
    ax.plot(epochs, history["val_loss"],
            label="Val loss",   color="#FF6B35",
            linewidth=1.5, linestyle="--")

    best_epoch = np.argmin(history["val_loss"]) + 1
    best_loss  = min(history["val_loss"])
    ax.axvline(best_epoch, color="green", linestyle=":",
               linewidth=1.0, alpha=0.7)
    ax.annotate(f"Best epoch {best_epoch}\n"
                f"val={best_loss:.6f}",
                xy=(best_epoch, best_loss),
                xytext=(10, 20), textcoords="offset points",
                fontsize=8, color="green",
                arrowprops=dict(arrowstyle="->",
                                color="green", lw=0.8))

    ax.set_title(
        f"Courbes d'apprentissage LSTM — "
        f"TND/{currency} J+{horizon}",
        fontweight="bold"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalisé)")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()

    fname = os.path.join(
        FIGURES_DIR,
        f"lstm_learning_{currency.lower()}_J{horizon}.png"
    )
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 {fname}")


def plot_lstm_forecast(actual: pd.Series,
                       pred_lstm: pd.Series,
                       pred_naive: pd.Series,
                       pred_arima: pd.Series,
                       currency: str, horizon: int):
    """
    Compare les prévisions LSTM vs Naïf vs ARIMA
    sur la période de test.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    common = actual.index.intersection(pred_lstm.index)

    ax.plot(actual.index, actual.values,
            label="Réel", color="black",
            linewidth=1.5, zorder=4)
    ax.plot(common, pred_naive.loc[common].values,
            label="Naïf", color="gray",
            linewidth=1.0, linestyle=":", zorder=2)
    ax.plot(common, pred_arima.loc[common].values,
            label="ARIMA", color="#FF6B35",
            linewidth=1.0, linestyle="--", zorder=3)
    ax.plot(common, pred_lstm.loc[common].values,
            label="LSTM", color="#1E88E5",
            linewidth=1.5, zorder=5)

    ax.set_title(
        f"Prévisions TND/{currency} — Horizon J+{horizon} "
        f"| Test {TEST_START} → {TEST_END}",
        fontweight="bold"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(f"TND/{currency}")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.xticks(rotation=45)
    plt.tight_layout()

    fname = os.path.join(
        FIGURES_DIR,
        f"lstm_forecast_{currency.lower()}_J{horizon}.png"
    )
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 {fname}")


# =============================================================
#  5. PIPELINE PAR DEVISE
# =============================================================

def run_lstm_currency(df: pd.DataFrame, currency: str,
                      quick: bool = False) -> dict:
    from .metrics import compute_metrics

    print(f"\n{'='*60}")
    print(f"  LSTM — TND/{currency}")
    print(f"{'='*60}")

    # FIX (Vague 6, Finding 3) : seed pour reproductibilité —
    # l'initialisation des poids LSTM/attention reste sinon
    # aléatoire d'un run à l'autre.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    arima_path = os.path.join(
        PROCESSED_DIR,
        f"arima_predictions_{currency.lower()}.csv"
    )
    arima_preds = None
    if os.path.exists(arima_path):
        arima_preds = pd.read_csv(arima_path, index_col=0, parse_dates=True)
        arima_preds.index.name = "Date"
        print(f"   ✅ Prévisions ARIMA/Naïf chargées depuis S2")
    else:
        print(f"   ⚠️  Prévisions ARIMA non trouvées — "
              f"lancer S2 d'abord")

    results = {"metrics": [], "predictions": {}}

    # FIX : découpage interne train → sous-train + validation,
    # le TEST (2024-2026) n'est JAMAIS vu pendant l'entraînement
    # ni pour l'early stopping — seulement à la toute fin, pour
    # les prévisions et les métriques finales.
    INTERNAL_VAL_START = "2022-01-01"

    for horizon in HORIZONS:
        print(f"\n--- Horizon J+{horizon} ---")

        feat = build_features(df, currency, horizon)
        X_train, y_train, X_test, y_test, feat_cols = \
            split_features(feat, currency)

        # Sous-découpage du train uniquement
        X_subtrain = X_train.loc[:INTERNAL_VAL_START].iloc[:-1]
        y_subtrain = y_train.loc[:INTERNAL_VAL_START].iloc[:-1]
        X_val      = X_train.loc[INTERNAL_VAL_START:]
        y_val      = y_train.loc[INTERNAL_VAL_START:]

        print(f"   Sous-train : {len(X_subtrain)} obs | "
              f"Validation interne : {len(X_val)} obs | "
              f"Test (jamais vu avant la fin) : {len(X_test)} obs")

        lstm_cfg = get_lstm_config(horizon)
        print(f"   Config J+{horizon} : hidden_size={lstm_cfg['hidden_size']} | "
              f"num_layers={lstm_cfg['num_layers']} | "
              f"dropout={lstm_cfg['dropout']} | "
              f"weight_decay={lstm_cfg['weight_decay']}")

        # DataLoader : train sur sous-train, "test_loader" ici
        # est en réalité la VALIDATION interne (nom conservé
        # pour compatibilité avec train_lstm, mais ce n'est plus
        # le test final)
        (train_loader, val_loader,
         scaler_X, scaler_y,
         _, _) = prepare_lstm_data(
            X_subtrain, y_subtrain, X_val, y_val, horizon
        )

        model = LSTMModel(
            input_size  = len(feat_cols),
            hidden_size = lstm_cfg["hidden_size"],
            num_layers  = lstm_cfg["num_layers"],
            dropout     = lstm_cfg["dropout"]
        )
        n_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
        print(f"   Paramètres LSTM : {n_params:,}")

        # Entraînement — early stopping sur la VALIDATION interne,
        # jamais sur le test
        history = train_lstm(
            model, train_loader, val_loader,
            currency, horizon, lstm_cfg, quick=quick
        )

        # Prévisions sur le VRAI test (2024-2026), jamais vu
        # avant maintenant — nécessite de re-normaliser X_test
        # avec le scaler_X ajusté sur le sous-train
        X_test_sc_final = scaler_X.transform(X_test.values)
        y_test_arr = y_test.values

        pred_lstm = predict_lstm(
            model, X_test_sc_final, y_test_arr,
            X_test.index,
            scaler_y, LSTM_CONFIG_BASE["sequence_length"], horizon
        )
        results["predictions"][f"LSTM_J{horizon}"] = pred_lstm

        actual = pd.Series(
            y_test_arr,
            index=X_test.index,
            name="Actual"
        )

        m_lstm = compute_metrics(
            actual, pred_lstm, "LSTM", currency, horizon
        )
        results["metrics"].append(m_lstm)
        print(f"\n   LSTM  J+{horizon} : "
              f"RMSE={m_lstm['RMSE']} | "
              f"MAE={m_lstm['MAE']} | "
              f"MAPE={m_lstm['MAPE (%)']}%")

        if arima_preds is not None:
            naive_col = f"Naif_J{horizon}"
            arima_col = f"ARIMA_J{horizon}"

            if naive_col in arima_preds.columns and \
               arima_col in arima_preds.columns:
                naive_s = arima_preds[naive_col]
                arima_s = arima_preds[arima_col]

                m_naive = compute_metrics(
                    actual, naive_s, "Naïf", currency, horizon
                )
                m_arima = compute_metrics(
                    actual, arima_s, "ARIMA", currency, horizon
                )
                results["metrics"].extend([m_naive, m_arima])

                print(f"   Naïf  J+{horizon} : "
                      f"RMSE={m_naive['RMSE']}")
                print(f"   ARIMA J+{horizon} : "
                      f"RMSE={m_arima['RMSE']}")

                gain = (m_arima["RMSE"] - m_lstm["RMSE"]) \
                        / m_arima["RMSE"] * 100
                print(f"   📊 Gain LSTM vs ARIMA : "
                      f"{gain:+.1f}% RMSE")

                plot_lstm_forecast(
                    actual, pred_lstm, naive_s, arima_s,
                    currency, horizon
                )

        plot_learning_curves(history, currency, horizon)

        model_path = os.path.join(
            MODELS_DIR,
            f"lstm_{currency.lower()}_J{horizon}.pt"
        )
        # FIX (Vague 6, Finding 2) : state_dict CPU explicite,
        # portable indépendamment du device d'entraînement.
        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state, model_path)
        print(f"   💾 Modèle sauvegardé → {model_path}")

    pred_df = pd.DataFrame(results["predictions"])
    pred_df.to_csv(os.path.join(
        PROCESSED_DIR,
        f"lstm_predictions_{currency.lower()}.csv"
    ))
    print(f"\n   💾 lstm_predictions_{currency.lower()}.csv")

    return results


def run_pipeline(quick: bool = False) -> pd.DataFrame:
    """
    Point d'entrée standard pour l'orchestrateur S3 (run_s3.py) —
    même rôle que mlp.py::run_pipeline(). Lance LSTM pour TND/USD
    puis TND/EUR, sauvegarde metrics_S3_lstm.csv, retourne le
    DataFrame de métriques combiné.

    Ne consolide PAS automatiquement (metrics_all.csv/predictions_*)
    — la consolidation reste un appel explicite séparé, cohérent
    avec run_s3.py qui n'appelle pas consolidate_all() lui-même.
    """
    mode = "RAPIDE (5 epochs)" if quick else "COMPLET"
    print("=" * 60)
    print(f"  PIPELINE LSTM — Mode {mode}")
    print("=" * 60)

    df = load_data()

    results_usd = run_lstm_currency(df, "USD", quick=quick)
    results_eur = run_lstm_currency(df, "EUR", quick=quick)

    all_metrics = results_usd["metrics"] + results_eur["metrics"]
    metrics_df  = pd.DataFrame(all_metrics)

    metrics_df.to_csv(
        os.path.join(PROCESSED_DIR, "metrics_S3_lstm.csv"),
        index=False
    )
    print(f"\n   💾 metrics_S3_lstm.csv sauvegardé")
    print("\n✅ Pipeline LSTM terminé")

    return metrics_df


# =============================================================
#  TEST RAPIDE / COMPLET
# =============================================================

if __name__ == "__main__":
    quick = "--quick" in sys.argv

    metrics_df = run_pipeline(quick=quick)

    print("\n" + "=" * 60)
    print("  TABLEAU MÉTRIQUES FINAL — LSTM TND/USD + TND/EUR")
    print("=" * 60)
    if not metrics_df.empty:
        print(metrics_df.to_string(index=False))

    print("\n✅ LSTM complet — TND/USD + TND/EUR terminé")
    from src.consolidate import consolidate_all
    consolidate_all()