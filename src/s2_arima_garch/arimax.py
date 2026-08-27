# =============================================================
#  src/s2_arima_garch/arimax.py
#  Modèle ARIMAX — ARIMA avec variables exogènes macro
#  Auteur : Youssef Neji | MINDS ENIT | 2025-2026
#
#  Sélection des variables exogènes PAR DEVISE et PAR HORIZON,
#  refit variable par horizon (REFIT_BY_HORIZON dans config.py).
#
#  Test rapide : python -m src.s2_arima_garch.arimax --quick
#  Test complet : python -m src.s2_arima_garch.arimax
# =============================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

from .config import (
    PROCESSED_DIR, FIGURES_DIR,
    TRAIN_START, TRAIN_END,
    TEST_START, TEST_END,
    HORIZONS, REFIT_BY_HORIZON,
)
from .metrics import compute_metrics


# =============================================================
#  1. VARIABLES EXOGÈNES PAR DEVISE ET PAR HORIZON
# =============================================================

# Colonnes réellement disponibles dans le dataset (rempli au
# runtime par set_available_columns, avant toute sélection).
_AVAILABLE_COLS = []


def set_available_columns(df_columns: list):
    """
    Initialise la liste des colonnes disponibles dans le dataset.
    Doit être appelée une fois, au début du pipeline, avant tout
    appel à get_exog_features.
    """
    global _AVAILABLE_COLS
    _AVAILABLE_COLS = list(df_columns)


def get_exog_features(horizon: int, currency: str) -> list:
    """
    Sélection des variables exogènes selon l'horizon ET la devise.

    Principe de sélection :
    ─────────────────────────────────────────────────
    TND/USD :
      J+1  → VIX (aversion au risque), EURUSD (cross rate),
             Brent (canal pétrolier), DXY si disponible
             (variables à réaction quotidienne)
      J+7  → VIX, EURUSD, Brent, DXY si disponible
             (identique à J+1 — les variables rapides restent
             pertinentes jusqu'à une semaine)
      J+30 → UNIQUEMENT variables lentes (TauxFed, IPC_USA,
             TauxMMBCT, US10Y) — Brent/VIX/EURUSD/DXY sont
             totalement exclues à cet horizon, aucun
             recouvrement avec J+1/J+7.
             Justification : sous l'hypothèse de persistance
             (pas de triche — voir forecast_arimax), la valeur
             exogène du jour est supposée constante sur tout
             l'horizon de prévision. Cette approximation est
             raisonnable pour des taux/indices qui bougent peu
             d'un jour à l'autre, mais devient mauvaise sur 30
             jours pour des séries volatiles comme le Brent ou
             le VIX.
             Note empirique (voir rapport, Chapitre 3) : ce
             changement vers des variables lentes a en réalité
             DÉGRADÉ le RMSE ARIMAX à J+30 USD (0.01167 → 0.01656)
             par rapport à la version avec variables rapides —
             conservé néanmoins comme spécification finale et
             documenté comme résultat négatif informatif.

    TND/EUR :
      Même jeu de 4 variables à TOUS les horizons —
      EURUSD (cross rate mécanique), VIX (aversion au risque),
      TauxBCE_EUR (effet direct), EUR10Y si disponible.
      Aucune différenciation par horizon : ce jeu mixte
      (une variable rapide + trois relativement stables)
      s'est avéré performant sur les trois horizons sans
      ajustement nécessaire.

    IMPORTANT : passer `currency` ("USD" ou "EUR"), jamais
    df.columns — c'est la cause du bug ValueError rencontré
    précédemment (`if currency == "USD"` sur un array pandas).
    """
    def _opt(cols):
        return [c for c in cols if c in _AVAILABLE_COLS]

    if currency == "USD":
        if horizon == 1:
            return ["VIX", "EURUSD", "Brent_USD"] + _opt(["DXY"])
        elif horizon == 7:
            return (["VIX", "EURUSD", "Brent_USD"]
                    + _opt(["DXY"]))
        else:  # J+30
            return (["TauxFed_USD", "IPC_USA", "TauxMMBCT"]
                    + _opt(["US10Y"]))

    elif currency == "EUR":
        if horizon == 1:
            return ["EURUSD", "VIX", "TauxBCE_EUR"] + _opt(["EUR10Y"])
        elif horizon == 7:
            return (["EURUSD", "TauxBCE_EUR", "VIX"]
                    + _opt(["EUR10Y"]))
        else:  # J+30
            return (["EURUSD", "TauxBCE_EUR", "VIX"]
                    + _opt(["EUR10Y"]))

    else:
        raise ValueError(
            f"currency doit être 'USD' ou 'EUR', reçu : {currency!r}"
        )


# =============================================================
#  2. SÉLECTION D'ORDRE ARIMAX PAR AIC
# =============================================================

def select_arimax_order(train_y: pd.Series, train_exog: pd.DataFrame,
                        max_p: int = 3, max_q: int = 3) -> tuple:
    """
    Sélectionne (p, d=1, q) par minimisation de l'AIC.
    Grille réduite (max 3) par rapport à ARIMA pur (max 4) car
    SARIMAX est nettement plus coûteux et plus sujet au
    surapprentissage avec des variables exogènes en plus.
    """
    print(f"   🔍 Sélection ARIMAX (grille p=[0-{max_p}], d=1, "
          f"q=[0-{max_q}]) par AIC...")

    best_aic, best_order = np.inf, (1, 1, 1)

    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            try:
                model = SARIMAX(
                    train_y, exog=train_exog,
                    order=(p, 1, q),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False)
                if result.aic < best_aic:
                    best_aic, best_order = result.aic, (p, 1, q)
            except Exception:
                continue

    print(f"   ✅ Meilleur ordre : ARIMAX{best_order} | AIC={best_aic:.2f}")
    return best_order


# =============================================================
#  3. PRÉVISIONS ARIMAX EN ROLLING WINDOW
# =============================================================

def forecast_arimax(
    train_y:      pd.Series,
    test_y:       pd.Series,
    train_exog:   pd.DataFrame,
    test_exog:    pd.DataFrame,
    order:        tuple,
    horizon:      int,
    refit_every:  int = None,
    verbose_every: int = 100
) -> tuple:
    """
    Prévisions ARIMAX en rolling window avec variables exogènes.

    FIX (Vague 2) — trois corrections appliquées simultanément :

    1. GEL PARTIEL ENTRE REFITS : entre deux refits, l'ancien code
       ne mettait jamais à jour l'état interne du modèle avec les
       observations réellement survenues — seule `exog_future`
       changeait, pas la composante AR/MA. Fix : `.append(...,
       refit=False)` avance l'état d'un pas à chaque itération
       sans refit complet.

    2. MAUVAIS HORODATAGE : `fc[-1]` (dernier pas d'une prévision
       à horizon `horizon`) cible la position `i+horizon-1`, pas
       `i` — décalage silencieux de `horizon-1` jours à J+7/J+30.

    3. FUITE EXOGÈNE (oracle → réaliste) : l'ancien code utilisait
       `test_exog_vals[i]`, la vraie valeur exogène DU JOUR CIBLE
       (information non disponible en production). Fix : utilise
       `history_exog[-1]`, la dernière exogène RÉELLEMENT connue
       au moment de la prévision, répétée sur tout l'horizon —
       hypothèse de persistance, cohérente avec ce que la
       docstring de get_exog_features décrivait déjà (à tort,
       avant ce fix).

    ⚠️ Cette dernière correction change fondamentalement la
    nature du modèle : ARIMAX devient un vrai modèle de
    production (aucune connaissance du futur), et non plus une
    borne supérieure théorique ("oracle assumption" mentionnée
    dans le rapport). L'avantage à J+1 mis en avant jusqu'ici
    doit être revalidé — il peut se réduire significativement.

    Warm-start + garde-fou de convergence/déviation conservés
    du fix précédent (stabilité numérique du refit).
    """
    if refit_every is None:
        refit_every = REFIT_BY_HORIZON.get(horizon, 1)

    print(f"   🔄 Prévisions ARIMAX rolling (horizon=J+{horizon}, "
          f"refit_every={refit_every}) — mode RÉALISTE "
          f"(pas de valeurs futures)...")

    history_y    = list(train_y.values)
    history_exog = train_exog.values.tolist()

    dates          = list(test_y.index)
    test_y_vals    = test_y.values
    test_exog_vals = test_exog.values
    n_test         = len(test_y)

    predictions    = {}     # {date_cible: valeur}
    result         = None
    last_result    = None
    initial_result = None   # premier fit (train seul) — Ljung-Box
    n_rejected     = 0      # non-convergence / déviation aberrante
    n_failures     = 0      # exceptions, fallback valeur naïve

    for i in range(n_test):
        target_idx = i + horizon - 1
        try:
            if result is None or i % refit_every == 0:
                model = SARIMAX(
                    history_y, exog=np.array(history_exog),
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                if last_result is not None:
                    try:
                        new_result = model.fit(
                            start_params=last_result.params,
                            disp=False, maxiter=100
                        )
                    except Exception:
                        new_result = model.fit(disp=False)
                else:
                    new_result = model.fit(disp=False)

                converged = new_result.mle_retvals.get(
                    "converged", True
                )
                if converged or last_result is None:
                    result      = new_result
                    last_result = new_result
                else:
                    n_rejected += 1
                    result = last_result

                if initial_result is None:
                    initial_result = result  # premier fit == train seul
            else:
                # FIX 1 : avance l'état d'un pas sans refit complet,
                # en incorporant la dernière observation réelle.
                result = result.append(
                    [history_y[-1]],
                    exog=[history_exog[-1]],
                    refit=False
                )
                last_result = result

            # FIX 3 : dernière exogène RÉELLEMENT connue (pas
            # test_exog_vals[i], qui regardait le futur), répétée
            # sur l'horizon — hypothèse de persistance.
            last_known_exog = np.array(
                history_exog[-1]
            ).reshape(1, -1)
            exog_future = np.repeat(
                last_known_exog, horizon, axis=0
            )

            fc     = result.forecast(steps=horizon, exog=exog_future)
            fc_val = fc[-1] if hasattr(fc, "__len__") else fc

            last_actual = history_y[-1]
            if abs(fc_val - last_actual) / last_actual > 0.05:
                n_rejected += 1
                fc_val = last_actual

            if target_idx < n_test:
                predictions[dates[target_idx]] = fc_val
        except Exception as e:
            n_failures += 1
            if target_idx < n_test:
                predictions[dates[target_idx]] = history_y[-1]

        history_y.append(test_y_vals[i])
        history_exog.append(test_exog_vals[i].tolist())

        if (i + 1) % verbose_every == 0:
            print(f"      {i+1}/{n_test} observations traitées...")

    if n_rejected > 0:
        print(f"   ⚠️  {n_rejected} refits rejetés/corrigés "
              f"(non-convergence ou déviation aberrante)")
    if n_failures > 0:
        pct = n_failures / n_test * 100
        print(f"   ⚠️  {n_failures}/{n_test} prévisions ({pct:.1f}%) "
              f"ont échoué et sont retombées sur la valeur naïve")

    pred_series = pd.Series(predictions).reindex(dates)
    pred_series.name = f"ARIMAX_J{horizon}"
    print(f"   ✅ {pred_series.notna().sum()}/{n_test} prévisions "
          f"générées (les {horizon-1} premières dates de test "
          f"n'ont pas de prévision J+{horizon} valide)")

    if initial_result is not None:
        resid_values = initial_result.resid
        residuals = pd.Series(
            resid_values[-len(train_y):], index=train_y.index
        )
    else:
        residuals = pd.Series(dtype=float)

    return pred_series, residuals


# =============================================================
#  4. VISUALISATION
# =============================================================

def plot_arimax_forecast(actual: pd.Series, arima_pred: pd.Series,
                         arimax_pred: pd.Series, naive_pred: pd.Series,
                         currency: str, horizon: int):
    """Trace Réel vs Naïf vs ARIMA vs ARIMAX pour un horizon donné."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual.index, actual.values, label="Réel",
            color="black", linewidth=1.5)

    common_n = actual.index.intersection(naive_pred.index)
    common_a = actual.index.intersection(arima_pred.index)
    common_x = actual.index.intersection(arimax_pred.index)

    ax.plot(common_n, naive_pred.loc[common_n].values,
            label="Naïf", linestyle="--", color="gray", linewidth=1.0)
    ax.plot(common_a, arima_pred.loc[common_a].values,
            label="ARIMA", linestyle="--", color="blue", linewidth=1.2)
    ax.plot(common_x, arimax_pred.loc[common_x].values,
            label="ARIMAX", linestyle="--", color="red", linewidth=1.2)

    ax.set_title(f"Prévisions TND/{currency} — Horizon J+{horizon} "
                 f"| Test {TEST_START} → {TEST_END}")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"TND/{currency}")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.xticks(rotation=45)
    plt.tight_layout()

    fname = os.path.join(
        FIGURES_DIR, f"arimax_forecast_{currency.lower()}_J{horizon}.png"
    )
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 {fname}")


# =============================================================
#  5. PIPELINE PAR DEVISE
# =============================================================

def run_arimax_currency(df: pd.DataFrame, currency: str,
                        quick: bool = False) -> dict:
    """
    Pipeline ARIMAX complet pour une devise.

    Étapes :
      1. Extraction train/test spot
      2. Pour chaque horizon (config.HORIZONS) :
         a. Sélection des variables exogènes (par devise ET horizon)
         b. Sélection de l'ordre par AIC (sauf mode quick)
         c. Prévisions rolling (refit variable par horizon)
         d. Métriques vs Naïf et ARIMA (chargés depuis S2)
         e. Figure comparative
      3. Sauvegarde CSV des prévisions
    """
    print(f"\n{'='*60}")
    print(f"  ARIMAX — TND/{currency}")
    print(f"{'='*60}")

    col_spot = f"TND_{currency}"

    spot_train = df[col_spot].loc[TRAIN_START:TRAIN_END]
    spot_test  = df[col_spot].loc[TEST_START:TEST_END]

    print(f"   Train : {len(spot_train)} obs "
          f"({TRAIN_START} → {TRAIN_END})")
    print(f"   Test  : {len(spot_test)} obs "
          f"({TEST_START} → {TEST_END})")

    # --- Charger prévisions ARIMA/Naïf depuis S2 ---
    arima_path = os.path.join(
        PROCESSED_DIR, f"arima_predictions_{currency.lower()}.csv"
    )
    arima_preds = None
    if os.path.exists(arima_path):
        arima_preds = pd.read_csv(arima_path, index_col=0, parse_dates=True)
        arima_preds.index.name = "Date"
        print(f"   ✅ Prévisions ARIMA/Naïf chargées")
    else:
        print(f"   ⚠️  {arima_path} introuvable — lancer S2 d'abord")

    results = {"metrics": [], "predictions": {}}

    for horizon in HORIZONS:
        print(f"\n--- Horizon J+{horizon} ---")

        # --- Sélection des variables exogènes (FIX du bug) ---
        # currency, PAS df.columns
        exog_cols = get_exog_features(horizon, currency)

        available = [c for c in exog_cols if c in df.columns]
        missing   = [c for c in exog_cols if c not in df.columns]
        if missing:
            print(f"   ⚠️  Variables manquantes : {missing}")
        exog_cols = available

        print(f"   Variables exogènes (J+{horizon}, {currency}) : "
              f"{exog_cols}")

        # FIX (Vague 4) : .bfill() sur train+test avant split propage une
        # valeur FUTURE vers l'arrière pour toute colonne démarrant après
        # TRAIN_START (DXY, EUR10Y) — fuite d'information distincte de
        # celle déjà corrigée en Vague 2 (last_known_exog). .ffill() seul
        # est cohérent avec le principe "aucune connaissance du futur"
        # déjà appliqué partout ailleurs dans ce module.
        exog_df = df[exog_cols].ffill()
        valid_start = exog_df.dropna().index.min()

        train_start_ts = pd.Timestamp(TRAIN_START)
        effective_start = max(train_start_ts, valid_start)

        exog_train = exog_df.loc[effective_start:TRAIN_END]
        exog_test  = exog_df.loc[TEST_START:TEST_END]

        spot_train_h = spot_train.loc[exog_train.index]

        n_test    = 30 if quick else len(spot_test)
        test_y    = spot_test.iloc[:n_test]
        test_exog = exog_test.iloc[:n_test]
        if quick:
            order = (1, 1, 1)
            print(f"   Mode rapide : ordre fixé à ARIMAX(1,1,1)")
        else:
            order = select_arimax_order(
                spot_train, exog_train, max_p=3, max_q=3
            )

        refit = 1 if quick else REFIT_BY_HORIZON.get(horizon, 1)
        print(f"   refit_every={refit} (horizon J+{horizon})")

        pred_arimax, residuals = forecast_arimax(
            spot_train, test_y, exog_train, test_exog,
            order, horizon, refit_every=refit, verbose_every=100
        )
        results["predictions"][f"ARIMAX_J{horizon}"] = pred_arimax

        m_arimax = compute_metrics(
            test_y, pred_arimax, "ARIMAX", currency, horizon
        )
        results["metrics"].append(m_arimax)

        print(f"\n   ARIMAX J+{horizon} : RMSE={m_arimax['RMSE']} | "
              f"MAE={m_arimax['MAE']} | MAPE={m_arimax['MAPE (%)']}%")

        if arima_preds is not None:
            naive_col = f"Naif_J{horizon}"
            arima_col = f"ARIMA_J{horizon}"

            if naive_col in arima_preds.columns and \
               arima_col in arima_preds.columns:
                naive_s = arima_preds[naive_col]
                arima_s = arima_preds[arima_col]

                m_naive = compute_metrics(test_y, naive_s, "Naïf",
                                          currency, horizon)
                m_arima = compute_metrics(test_y, arima_s, "ARIMA",
                                          currency, horizon)
                results["metrics"].extend([m_naive, m_arima])

                print(f"   Naïf  J+{horizon} : RMSE={m_naive['RMSE']}")
                print(f"   ARIMA J+{horizon} : RMSE={m_arima['RMSE']}")

                gain_arima = ((m_arima["RMSE"] - m_arimax["RMSE"])
                              / m_arima["RMSE"] * 100)
                gain_naive = ((m_naive["RMSE"] - m_arimax["RMSE"])
                              / m_naive["RMSE"] * 100)

                print(f"   📊 Gain ARIMAX vs ARIMA : {gain_arima:+.1f}%")
                print(f"   📊 Gain ARIMAX vs Naïf  : {gain_naive:+.1f}%")

                if not quick:
                    plot_arimax_forecast(
                        test_y, arima_s, pred_arimax, naive_s,
                        currency, horizon
                    )
            else:
                print(f"   ⚠️  Colonnes {naive_col}/{arima_col} "
                      f"introuvables dans {arima_path}")

    pred_df = pd.DataFrame(results["predictions"])
    out_path = os.path.join(
        PROCESSED_DIR, f"arimax_predictions_{currency.lower()}.csv"
    )
    pred_df.to_csv(out_path)
    print(f"\n   💾 {out_path}")

    return results


# =============================================================
#  TEST RAPIDE (exécution directe du module)
# =============================================================

if __name__ == "__main__":
    from .load import load_data

    quick_mode = "--quick" in sys.argv
    mode = "RAPIDE (test mécanique)" if quick_mode else "COMPLET"

    print("=" * 60)
    print(f"  ARIMAX — Mode {mode}")
    print(f"  Horizons : {HORIZONS}")
    print("=" * 60)

    df = load_data()
    set_available_columns(df.columns.tolist())

    results_usd = run_arimax_currency(df, "USD", quick=quick_mode)
    print(f"\nAperçu USD :\n"
          f"{list(results_usd['predictions'].values())[0].head(5)}")