# update_august_actuals.py
# Lance : python update_august_actuals.py
# Lit BCT_taux_change.csv et met à jour predictions_aout2026.json

import json
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIVE_DIR = os.path.join(BASE_DIR, "data", "live")
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")


def update_actuals():
    # ── Charger les vraies valeurs BCT ───────────────────
    bct_path = os.path.join(RAW_DIR, "BCT_taux_change.csv")
    if not os.path.exists(bct_path):
        print(f"❌ {bct_path} non trouvé")
        return

    bct = pd.read_csv(bct_path, index_col=0,
                      parse_dates=True)
    bct.index.name = "Date"

    # Filtrer août 2026
    aug_actuals = bct.loc["2026-08-01":"2026-08-31"]
    print(f"✅ Valeurs BCT Août 2026 trouvées : "
          f"{len(aug_actuals)} jours")
    print(aug_actuals.to_string())

    # ── Charger predictions_aout2026.json ────────────────
    pred_path = os.path.join(
        LIVE_DIR, "predictions_aout2026.json"
    )
    if not os.path.exists(pred_path):
        print(f"\n❌ {pred_path} non trouvé")
        print("   Lancez : python -m src.predict_august")
        return

    with open(pred_path) as f:
        data = json.load(f)

    # ── Injecter les vraies valeurs ───────────────────────
    n_updated = 0
    for date_idx, row in aug_actuals.iterrows():
        date_str = date_idx.strftime("%Y-%m-%d")

        for currency, col in [
            ("USD", "TND_USD"),
            ("EUR", "TND_EUR")
        ]:
            actual_val = round(float(row[col]), 5)
            for model_res in \
                    data["forecasts"].get(
                        currency, {}
                    ).values():
                if not isinstance(model_res, dict):
                    continue
                j1d = model_res.get("J1_daily", {})
                if date_str in j1d:
                    j1d[date_str]["actual"] = actual_val
                    n_updated += 1

    print(f"\n✅ {n_updated} entrées mises à jour")

    # ── Calculer les métriques de validation ─────────────
    results = compute_validation_metrics(data)
    data["validation_results"] = results
    data["validation_status"]  = "in_progress"
    data["last_validation"]    = str(
        aug_actuals.index[-1].date()
    )

    # ── Sauvegarder ──────────────────────────────────────
    with open(pred_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ {pred_path} mis à jour")

    # ── Afficher le tableau ───────────────────────────────
    print_validation_table(results)


def compute_validation_metrics(data: dict) -> dict:
    results = {}
    for currency in ["USD", "EUR"]:
        results[currency] = {}
        for model_name, model_res in \
                data["forecasts"].get(
                    currency, {}
                ).items():
            if not isinstance(model_res, dict):
                continue

            j1d     = model_res.get("J1_daily", {})
            preds   = []
            actuals = []
            dates   = []

            for d, vals in sorted(j1d.items()):
                p = vals.get("prediction")
                a = vals.get("actual")
                if p is not None and a is not None:
                    preds.append(p)
                    actuals.append(a)
                    dates.append(d)

            if len(preds) < 2:
                results[currency][model_name] = {
                    "n_obs":  len(preds),
                    "status": "insufficient data"
                }
                continue

            p_arr = np.array(preds)
            a_arr = np.array(actuals)

            rmse = float(
                np.sqrt(np.mean((p_arr - a_arr)**2))
            )
            mae  = float(np.mean(np.abs(p_arr - a_arr)))
            mape = float(
                np.mean(
                    np.abs((p_arr - a_arr) / a_arr)
                ) * 100
            )

            # Directional accuracy
            d_a  = np.sign(np.diff(a_arr))
            d_p  = np.sign(np.diff(p_arr))
            mask = d_a != 0
            da   = (
                float(
                    np.mean(d_a[mask] == d_p[mask]) * 100
                ) if mask.sum() > 0 else None
            )

            # Erreurs par date
            daily_errors = {
                d: round(abs(float(p)-float(a)), 5)
                for d, p, a in zip(dates, preds, actuals)
            }

            results[currency][model_name] = {
                "n_obs":        len(preds),
                "dates":        dates,
                "RMSE":         round(rmse, 6),
                "MAE":          round(mae, 6),
                "MAPE_%":       round(mape, 4),
                "Dir_Acc_%":    (round(da, 1)
                                 if da is not None
                                 else None),
                "daily_errors": daily_errors,
            }

    return results


def print_validation_table(results: dict):
    print("\n" + "=" * 65)
    print("  VALIDATION AOÛT 2026 — Prévisions vs Réel BCT")
    print("=" * 65)

    for currency, models in results.items():
        print(f"\n  TND/{currency} :")
        print(f"  {'Modèle':<10} {'N':>4} "
              f"{'RMSE':>9} {'MAE':>9} "
              f"{'MAPE%':>7} {'Dir.Acc%':>9}")
        print(f"  {'-'*55}")

        valid = {
            k: v for k, v in models.items()
            if "RMSE" in v
        }
        for mod, m in sorted(
            valid.items(),
            key=lambda x: x[1].get("RMSE", 999)
        ):
            da_str = (f"{m['Dir_Acc_%']:.1f}%"
                      if m.get("Dir_Acc_%") is not None
                      else "—")
            print(
                f"  {mod:<10} "
                f"{m['n_obs']:>4} "
                f"{m['RMSE']:>9.5f} "
                f"{m['MAE']:>9.5f} "
                f"{m['MAPE_%']:>7.3f} "
                f"{da_str:>9}"
            )

        # Meilleur modèle
        if valid:
            best = min(valid,
                       key=lambda k: valid[k]["RMSE"])
            print(f"\n  🏆 Meilleur : {best} "
                  f"(RMSE={valid[best]['RMSE']:.5f})")

    print("\n" + "=" * 65)
    print("  ERREURS QUOTIDIENNES")
    print("=" * 65)
    for currency, models in results.items():
        print(f"\n  TND/{currency} :")
        valid = {k: v for k, v in models.items()
                 if "daily_errors" in v}
        if not valid:
            continue
        # Toutes les dates
        all_dates = sorted(set(
            d for m in valid.values()
            for d in m.get("daily_errors", {}).keys()
        ))
        # En-tête
        header = f"  {'Date':<12}"
        for mod in valid:
            header += f" {mod:>9}"
        print(header)
        print(f"  {'-'*55}")
        for d in all_dates:
            row = f"  {d:<12}"
            for mod, m in valid.items():
                err = m["daily_errors"].get(d)
                row += (f" {err:>9.5f}"
                        if err is not None else
                        f" {'—':>9}")
            print(row)


if __name__ == "__main__":
    update_actuals()