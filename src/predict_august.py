# =============================================================
#  src/predict_august.py
#  Prévisions Août 2026 — Retrain final 2004→31/07/2026
#  Auteur : Youssef Neji | MINDS ENIT | 2026
#
#  Usage : python -m src.predict_august
# =============================================================

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

BASE_DIR      = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "data", "models")
LIVE_DIR      = os.path.join(BASE_DIR, "data", "live")
os.makedirs(LIVE_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)

# Jours ouvrés août 2026 — 10 premiers
AUG_BDAYS  = pd.bdate_range(
    "2026-08-01", "2026-08-31"
)[:10]
CURRENCIES = ["USD", "EUR"]
HORIZONS   = [1, 7, 30]

# ← Déclaré UNE SEULE FOIS ici, mis à jour dans load
RETRAIN_END = "2026-07-31"


# =============================================================
#  LOAD
# =============================================================



def load_full_dataset() -> pd.DataFrame:
    # RETRAIN_END reste fixé à "2026-07-31"
    # On ne le met PAS à jour automatiquement
    # car on veut entraîner JUSQU'AU 31/07/2026
    # et prédire AOÛT 2026 (données non vues)
    p  = os.path.join(PROCESSED_DIR, "dataset_final.csv")
    df = pd.read_csv(p, parse_dates=["Date"],
                     index_col="Date")
    print(f"✅ Dataset chargé : {df.shape}")
    print(f"   Période       : {df.index[0].date()} "
          f"→ {df.index[-1].date()}")
    print(f"   RETRAIN_END   : {RETRAIN_END}")
    return df
# =============================================================
#  PRÉVISIONS ARIMA — RETRAIN FINAL
# =============================================================

def predict_august_arima(
    df: pd.DataFrame, currency: str
) -> dict:
    from statsmodels.tsa.arima.model import (
        ARIMA as ARIMAModel
    )
    from src.s2_arima_garch.arima import (
        select_arima_order
    )

    print(f"\n📈 ARIMA TND/{currency} — Retrain final...")

    col_spot = f"TND_{currency}"
    series   = df[col_spot].loc[:RETRAIN_END].dropna()

    print(f"   Train : {series.index[0].date()} → "
          f"{series.index[-1].date()} "
          f"({len(series)} obs)")

    order     = select_arima_order(
        series, max_p=4, max_q=4
    )
    model_fit = ARIMAModel(
        series.values, order=order
    ).fit()
    print(f"   Ordre : ARIMA{order}")

    last_value = float(series.iloc[-1])
    results = {
        "model":      f"ARIMA{order}",
        "last_date":  series.index[-1].strftime(
            "%Y-%m-%d"
        ),
        "last_value": last_value,
        "J1_daily":   {},
        "J7":         None,
        "J30":        None,
    }

    # FIX (Vague 4, Finding 12) : un seul fit initial, puis
    # .append(refit=False) par jour au lieu d'un refit MLE complet
    # à chaque itération — même principe que arima.py/arimax.py.
    print(f"   J+1 quotidien ({len(AUG_BDAYS)}j)...")
    result = model_fit  # déjà ajusté sur `series` plus haut
    for target_date in AUG_BDAYS:
        fc  = result.forecast(steps=1)
        val = float(fc.iloc[0] if hasattr(fc, "iloc") else fc[0])
        results["J1_daily"][
            target_date.strftime("%Y-%m-%d")
        ] = {"prediction": round(val, 5), "actual": None}
        result = result.append([val], refit=False)
    print(f"   ✅ J+1 : {len(results['J1_daily'])} prévisions")

    # J+7
    fc7 = model_fit.forecast(steps=7)
    results["J7"] = round(
        float(fc7.iloc[-1]
              if hasattr(fc7, "iloc") else fc7[-1]),
        5
    )

    # J+30
    fc30 = model_fit.forecast(steps=30)
    results["J30"] = round(
        float(fc30.iloc[-1]
              if hasattr(fc30, "iloc") else fc30[-1]),
        5
    )

    print(f"   J+7  : {results['J7']:.5f}")
    print(f"   J+30 : {results['J30']:.5f}")
    return results


# =============================================================
#  PRÉVISIONS ARIMAX — RETRAIN FINAL
# =============================================================

def predict_august_arimax(
    df: pd.DataFrame, currency: str
) -> dict:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from src.s2_arima_garch.arimax import (
        select_arimax_order,
        get_exog_features,
        set_available_columns,
    )

    print(
        f"\n🌍 ARIMAX TND/{currency} — Retrain final..."
    )

    col_spot = f"TND_{currency}"
    series = df[col_spot].loc[:RETRAIN_END].dropna()

    set_available_columns(df.columns.tolist())

    last_value = float(series.iloc[-1])
    results = {
        "model":      "ARIMAX",
        "last_date":  series.index[-1].strftime(
            "%Y-%m-%d"
        ),
        "last_value": last_value,
        "J1_daily":   {},
        "J7":         None,
        "J30":        None,
    }

    def build_exog(horizon):
        cols  = get_exog_features(horizon, currency)
        avail = [c for c in cols if c in df.columns]
        if not avail:
            return pd.DataFrame(index=df.index)
        return df[avail].loc[:RETRAIN_END].ffill().bfill()
    # ── J+1 quotidien ──────────────────────────────────
    print(f"   J+1 quotidien ({len(AUG_BDAYS)}j)...")
    exog_df1    = build_exog(1)
    exog_train1 = exog_df1.loc[:RETRAIN_END]

    order1 = select_arimax_order(
        series, exog_train1, max_p=3, max_q=3
    )
    print(f"   Ordre J+1 : ARIMAX{order1}")

    history_y    = list(series.values)
    history_exog = exog_train1.values.tolist()
    last_exog    = exog_train1.iloc[-1].values.copy()

    for target_date in AUG_BDAYS:
        try:
            m = SARIMAX(
                np.array(history_y),
                exog=np.array(history_exog),
                order=order1,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            fc  = m.forecast(
                steps=1,
                exog=last_exog.reshape(1, -1)
            )
            val = float(
                fc.iloc[0] if hasattr(fc, "iloc")
                else fc[0]
            )
        except Exception:
            val = history_y[-1]

        results["J1_daily"][
            target_date.strftime("%Y-%m-%d")
        ] = {
            "prediction": round(val, 5),
            "actual":     None
        }
        history_y.append(val)
        history_exog.append(last_exog.tolist())

    print(f"   ✅ J+1 : "
          f"{len(results['J1_daily'])} prévisions")

    # ── J+7 et J+30 ────────────────────────────────────
    for h, key in [(7, "J7"), (30, "J30")]:
        try:
            exog_h    = build_exog(h)
            exog_tr_h = exog_h.loc[:RETRAIN_END]
            order_h   = select_arimax_order(
                series, exog_tr_h,
                max_p=3, max_q=3
            )
            last_exog_h  = exog_tr_h.iloc[-1].values
            future_exog  = np.tile(
                last_exog_h, (h, 1)
            )
            m_h = SARIMAX(
                series.values,
                exog=exog_tr_h.values,
                order=order_h,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            fc_h = m_h.forecast(
                steps=h, exog=future_exog
            )
            results[key] = round(
                float(
                    fc_h.iloc[-1]
                    if hasattr(fc_h, "iloc")
                    else fc_h[-1]
                ), 5
            )
        except Exception as e:
            print(f"   ⚠️  J+{h} : {e}")
            results[key] = None
        print(f"   {key} : {results[key]}")

    return results


# =============================================================
#  PRÉVISIONS LSTM — RETRAIN FINAL (corrigé)
# =============================================================

def predict_august_lstm(
    df: pd.DataFrame, currency: str
) -> dict:
    import torch
    from src.s3_lstm_xgboost.load import (
        build_features, prepare_lstm_data
    )
    from src.s3_lstm_xgboost.lstm import (
        LSTMAttentionModel, train_lstm,
        get_lstm_config
    )
    from src.s3_lstm_xgboost.config import (
        LSTM_CONFIG_BY_HORIZON, LSTM_CONFIG_BASE
    )
    import src.s3_lstm_xgboost.config as cfg

    print(
        f"\n🤖 LSTM TND/{currency} — Retrain final..."
    )

    col_spot   = f"TND_{currency}"
    col_logret = f"LogRet_TND_{currency}"

    # Override config
    orig_start      = cfg.TRAIN_START
    orig_end        = cfg.TRAIN_END
    cfg.TRAIN_START = "2004-01-01"
    cfg.TRAIN_END   = RETRAIN_END

    feat      = build_features(df, currency, horizon=1)
    feat_all  = feat.loc["2004-01-01":RETRAIN_END]
    feat_cols = [c for c in feat_all.columns
                 if c not in [col_spot, col_logret]]

    X_all = feat_all[feat_cols]
    y_all = feat_all[col_spot]

    last_value = float(y_all.iloc[-1])
    results = {
        "model":      "LSTM+Attention",
        "last_date":  feat_all.index[-1].strftime(
            "%Y-%m-%d"
        ),
        "last_value": last_value,
        "J1_daily":   {},
        "J7":         None,
        "J30":        None,
    }

    seq_len = LSTM_CONFIG_BASE["sequence_length"]
    device  = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    for horizon in [1, 7, 30]:
        print(f"   Entraînement J+{horizon}...")
        try:
            (train_loader, val_loader,
             scaler_X, scaler_y,
             X_val_sc, _) = prepare_lstm_data(
                X_all, y_all,
                X_all.iloc[-seq_len*3:],
                y_all.iloc[-seq_len*3:],
                horizon
            )

            lstm_cfg = get_lstm_config(horizon)
            model    = LSTMAttentionModel(
                input_size  = len(feat_cols),
                hidden_size = lstm_cfg["hidden_size"],
                num_layers  = lstm_cfg["num_layers"],
                dropout     = lstm_cfg["dropout"]
            )

            train_lstm(
                model, train_loader, val_loader,
                currency, horizon, lstm_cfg,
                quick=False
            )
            model.eval()
            model.to(device)

            if horizon == 1:
                # ── J+1 quotidien en cascade ────────────
                # Valeur de référence = dernier cours réel
                last_spot   = float(y_all.iloc[-1])
                current_seq = (
                    X_all.iloc[-seq_len:].values.copy()
                )

                for target_date in AUG_BDAYS:
                    X_sc = scaler_X.transform(
                        current_seq[-seq_len:]
                    )
                    x_t = torch.FloatTensor(
                        X_sc
                    ).unsqueeze(0).to(device)

                    with torch.no_grad():
                        p_sc = model(x_t).cpu().item()

                    # Dénormaliser
                    try:
                        pred_raw = float(
                            scaler_y.inverse_transform(
                                [[p_sc]]
                            )[0][0]
                        )
                        # Si log-rendement → reconstruire
                        if abs(pred_raw) < 0.1:
                            val = last_spot * (
                                1 + pred_raw
                            )
                        else:
                            # Déjà un niveau
                            val = pred_raw
                    except Exception:
                        val = last_spot

                    # Vérification cohérence ±5%
                    if not (last_spot * 0.95
                            < val
                            < last_spot * 1.05):
                        val = last_spot

                    results["J1_daily"][
                        target_date.strftime(
                            "%Y-%m-%d"
                        )
                    ] = {
                        "prediction": round(val, 5),
                        "actual":     None
                    }

                    # Décaler la fenêtre
                    last_spot  = val
                    new_row    = current_seq[-1].copy()
                    new_row[0] = val
                    current_seq = np.vstack(
                        [current_seq[1:], new_row]
                    )

                print(f"   ✅ J+1 : "
                      f"{len(results['J1_daily'])} "
                      f"prévisions")

            else:
                # ── J+7 et J+30 depuis dernier point ───
                X_last = scaler_X.transform(
                    X_all.iloc[-seq_len:].values
                )
                x_t = torch.FloatTensor(
                    X_last
                ).unsqueeze(0).to(device)

                with torch.no_grad():
                    p_sc = model(x_t).cpu().item()

                try:
                    pred_raw = float(
                        scaler_y.inverse_transform(
                            [[p_sc]]
                        )[0][0]
                    )
                    last_spot = float(y_all.iloc[-1])
                    if abs(pred_raw) < 0.1:
                        pred_val = last_spot * (
                            1 + pred_raw
                        )
                    else:
                        pred_val = pred_raw

                    # Vérification cohérence
                    if not (last_spot * 0.90
                            < pred_val
                            < last_spot * 1.10):
                        pred_val = last_spot
                except Exception:
                    pred_val = float(y_all.iloc[-1])

                key = f"J{horizon}"
                results[key] = round(pred_val, 5)
                print(f"   {key} : {results[key]:.5f}")

        except Exception as e:
            print(f"   ⚠️  J+{horizon} échoué : {e}")

    cfg.TRAIN_START = orig_start
    cfg.TRAIN_END   = orig_end
    return results


# =============================================================
#  PRÉVISIONS MLP — RETRAIN FINAL
# =============================================================

def predict_august_mlp(
    df: pd.DataFrame, currency: str
) -> dict:
    """
    FIX : réutilise train_final_mlp() de mlp.py — même
    architecture et même cible (rendement log reconstruit) que
    le modèle validé par Diebold-Mariano. Élimine la divergence
    précédemment identifiée (niveau brut vs rendement log,
    variables macro figées sur J+1 pour tous les horizons).

    Cascade J+1 : recalcule les features chaque jour depuis la
    série étendue (même stratégie que le fix précédent dans ce
    fichier), prédit le rendement log, reconstruit le niveau,
    avance la fenêtre — pas de figeage.
    """
    import torch
    from src.s3_lstm_xgboost.mlp import train_final_mlp, MLPModel
    from src.s3_lstm_xgboost.load import build_features

    print(f"\n⚡ MLP TND/{currency} — Retrain final "
          f"(cohérent avec mlp.py)...")

    col        = f"TND_{currency}"
    col_logret = f"LogRet_TND_{currency}"

    results = {
        "model":      "MLP",
        "last_date":  "",
        "last_value": 0.0,
        "J1_daily":   {},
        "J7":         None,
        "J30":        None,
    }

    try:
        for horizon in [1, 7, 30]:
            print(f"   Entraînement J+{horizon}...")
            bundle = train_final_mlp(
                df, currency, horizon, end_date=RETRAIN_END
            )

            model = MLPModel(
                input_size=len(bundle["feat_cols"]),
                hidden_sizes=bundle["hidden_sizes"],
                dropout=bundle["dropout"]
            )
            model.load_state_dict(bundle["model_state"])
            model.eval()

            scaler_X  = bundle["scaler_X"]
            scaler_y  = bundle["scaler_y"]
            feat_cols = bundle["feat_cols"]
            last_spot = bundle["last_value"]

            results["last_date"]  = bundle["last_date"]
            results["last_value"] = last_spot

            if horizon == 1:
                extended_df = df.loc[:RETRAIN_END].copy()
                cur_spot    = last_spot

                for target_date in AUG_BDAYS:
                    feat_today = build_features(
                        extended_df, currency, horizon=1
                    )
                    x_today    = feat_today[feat_cols].iloc[-1:].values
                    x_today_sc = scaler_X.transform(x_today)

                    with torch.no_grad():
                        pred_sc = model(
                            torch.FloatTensor(x_today_sc)
                        ).numpy()

                    pred_ret = scaler_y.inverse_transform(
                        pred_sc.reshape(-1, 1)
                    ).flatten()[0]
                    val = cur_spot * np.exp(pred_ret)

                    # Garde-fou de cohérence ±5% — même logique
                    # que pour ARIMA/ARIMAX
                    if not (cur_spot*0.95 < val < cur_spot*1.05):
                        val = cur_spot

                    results["J1_daily"][
                        target_date.strftime("%Y-%m-%d")
                    ] = {
                        "prediction": round(val, 5),
                        "actual": None
                    }

                    new_row = extended_df.iloc[-1:].copy()
                    new_row.index = [target_date]
                    new_row[col] = val
                    prev_val = extended_df[col].iloc[-1]
                    if col_logret in new_row.columns:
                        new_row[col_logret] = (
                            np.log(val/prev_val)
                            if prev_val > 0 else 0.0
                        )
                    extended_df = extended_df[
                        ~extended_df.index.isin([target_date])
                    ]
                    extended_df = pd.concat(
                        [extended_df, new_row]
                    ).sort_index()

                    cur_spot = val

                print(f"   ✅ J+1 : "
                      f"{len(results['J1_daily'])} prévisions "
                      f"(rendement log reconstruit, "
                      f"cohérent avec mlp.py)")

            else:
                x_last_sc = bundle["last_row_sc"]
                with torch.no_grad():
                    pred_sc = model(
                        torch.FloatTensor(x_last_sc)
                    ).numpy()
                pred_ret = scaler_y.inverse_transform(
                    pred_sc.reshape(-1, 1)
                ).flatten()[0]
                val = last_spot * np.exp(pred_ret)

                if not (last_spot*0.90 < val < last_spot*1.10):
                    val = last_spot

                key = f"J{horizon}"
                results[key] = round(val, 5)
                print(f"   {key} : {results[key]:.5f}")

    except Exception as e:
        print(f"   ❌ MLP échoué : {e}")

    return results


# =============================================================
#  SAUVEGARDE JSON
# =============================================================

def _merge_preserving_actuals(new_forecasts: dict,
                              old_path: str) -> tuple:
    """
    Fusionne les nouvelles prévisions avec le fichier existant en
    PRÉSERVANT les valeurs 'actual' déjà injectées par
    update_august_actuals.py — sans ça, chaque relance de
    predict_august.py efface silencieusement la validation en
    cours.
    """
    if not os.path.exists(old_path):
        return new_forecasts, "pending", {}
    try:
        with open(old_path) as f:
            old_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"   ⚠️  Impossible de lire l'ancien fichier "
              f"({e}) — écrasement complet")
        return new_forecasts, "pending", {}

    old_forecasts = old_data.get("forecasts", {})
    for currency, models in new_forecasts.items():
        for model_name, res in models.items():
            if not isinstance(res, dict):
                continue
            old_res = old_forecasts.get(currency, {}).get(model_name, {})
            old_j1d = (old_res.get("J1_daily", {})
                      if isinstance(old_res, dict) else {})
            for date_str, entry in res.get("J1_daily", {}).items():
                old_entry = old_j1d.get(date_str)
                if old_entry and old_entry.get("actual") is not None:
                    entry["actual"] = old_entry["actual"]

    extra_fields = {
        k: old_data[k] for k in
        ("validation_results", "last_validation")
        if k in old_data
    }
    validation_status = old_data.get("validation_status", "pending")
    return new_forecasts, validation_status, extra_fields


def save_predictions(all_results: dict):
    path = os.path.join(LIVE_DIR, "predictions_aout2026.json")
    merged_forecasts, validation_status, extra_fields = (
        _merge_preserving_actuals(all_results, path)
    )

    output = {
        "generated_on":    datetime.now().isoformat(),
        "retrain_period":  f"2004-01-01 → {RETRAIN_END}",
        "forecast_period": {
            "start":  AUG_BDAYS[0].strftime("%Y-%m-%d"),
            "end":    AUG_BDAYS[-1].strftime("%Y-%m-%d"),
            "n_days": len(AUG_BDAYS),
            "days":   [d.strftime("%Y-%m-%d") for d in AUG_BDAYS]
        },
        "models_used": {
            "J+1_daily": "ARIMA · ARIMAX · LSTM · MLP",
            "J+7":       "ARIMA · ARIMAX · LSTM · MLP",
            "J+30":      "ARIMA · ARIMAX · LSTM · MLP"
        },
        "forecasts":         merged_forecasts,
        "validation_status": validation_status,
        "validation_note": (
            "Comparer avec vraies valeurs BCT en septembre 2026"
        ),
        **extra_fields,
    }

    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(output, f, indent=2)
    os.replace(tmp_path, path)
    print(f"\n💾 {path} (fusionné, actuals préservés)")

# =============================================================
#  PIPELINE PRINCIPAL
# =============================================================

def run_all_predictions():
    print("=" * 60)
    print("  PRÉVISIONS AOÛT 2026")
    print(f"  Jours ouvrés  : "
          f"{len(AUG_BDAYS)} (10 premiers)")
    print(f"  Currencies    : {CURRENCIES}")
    print(f"  Modèles       : "
          f"ARIMA · ARIMAX · LSTM · MLP")
    print("=" * 60)

    df = load_full_dataset()

    print(f"\n   Retrain final : 2004-01-01 "
          f"→ {RETRAIN_END}")

    all_results = {}
    for currency in CURRENCIES:
        print(f"\n{'='*60}")
        print(f"  TND/{currency}")
        print(f"{'='*60}")
        cur = {}

        for name, func in [
            ("ARIMA",  predict_august_arima),
            ("ARIMAX", predict_august_arimax),
            ("LSTM",   predict_august_lstm),
            ("MLP",    predict_august_mlp),
        ]:
            try:
                cur[name] = func(df, currency)
            except Exception as e:
                print(f"   ❌ {name} : {e}")
                cur[name] = {"error": str(e)}

        all_results[currency] = cur

    save_predictions(all_results)
    print("\n✅ Prévisions Août 2026 générées")


if __name__ == "__main__":
    run_all_predictions()