# =============================================================
#  src/s2_arima_garch/arima.py
#  Sélection d'ordre ARIMA + prévisions rolling
#
#  Test rapide (RAPIDE - ordre fixe) :
#    python -m src.s2_arima_garch.arima --quick
#  Test complet (LENT - grille AIC + rolling complet) :
#    python -m src.s2_arima_garch.arima
# =============================================================

import sys
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


def select_arima_order(train: pd.Series, max_p: int = 6, max_q: int = 6) -> tuple:
    """
    Sélectionne (p, d=1, q) par minimisation de l'AIC.
    d=1 car les cours de change sont non-stationnaires (confirmé par ADF/KPSS).
    """
    print(f"   🔍 Sélection ARIMA (grille p=[0-{max_p}], d=1, q=[0-{max_q}]) par AIC...")

    best_aic, best_order = np.inf, (1, 1, 1)

    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            try:
                result = ARIMA(train, order=(p, 1, q)).fit()
                if result.aic < best_aic:
                    best_aic, best_order = result.aic, (p, 1, q)
            except Exception:
                continue

    print(f"   ✅ Meilleur ordre : ARIMA{best_order} | AIC={best_aic:.2f}")
    return best_order


def forecast_arima(train: pd.Series, test: pd.Series,
                   order: tuple, horizon: int,
                   refit_every: int = 1,
                   verbose_every: int = 100) -> tuple:
    """
    Prévisions ARIMA en rolling window.

    FIX (Vague 2) — deux bugs corrigés simultanément :

    1. GEL ENTRE REFITS : l'ancienne version réutilisait `result`
       tel quel les jours sans refit, donc `result.forecast()`
       retournait la MÊME valeur en boucle jusqu'au refit suivant.
       Fix : `.append([history[-1]], refit=False)` met à jour
       l'état filtré du modèle avec la nouvelle observation SANS
       ré-estimer les paramètres (peu coûteux), au lieu de ne rien
       faire du tout entre deux refits.

    2. MAUVAIS HORODATAGE : `fc[-1]` (dernier pas de la prévision
       à horizon `horizon`) correspond à la date `dates[i+horizon-1]`,
       pas à `dates[i]` — l'ancienne version les confondait, ce qui
       n'a aucun effet à J+1 (horizon-1=0) mais décale les
       prévisions de horizon-1 jours à J+7 et J+30.

    Conséquence du fix n°2 : les horizon-1 premières dates du test
    n'ont plus de prévision J+horizon valide (aucune origine
    antérieure au test ne permet de les calculer dans ce schéma
    rolling) — légère réduction de N à J+7/J+30, intentionnelle
    et documentée dans le message de fin de fonction.

    Returns:
        (pred_series, residuals_train)
        residuals_train : résidus du modèle ajusté sur le train
                          SEUL (avant tout refit sur le test),
                          utilisés pour le test de Ljung-Box —
                          voir note Finding 3 plus bas.
    """
    print(f"   🔄 Prévisions ARIMA rolling (horizon=J+{horizon}, refit_every={refit_every})...")

    history     = list(train.values)
    dates       = list(test.index)
    test_values = test.values
    n_test      = len(test)

    predictions    = {}   # {date_cible: valeur} — indexé par date réelle
    result         = None
    last_result    = None
    initial_result = None   # premier fit (train seul) — pour Ljung-Box
    n_failures     = 0
    last_error     = None

    for i in range(n_test):
        target_idx = i + horizon - 1   # position réellement visée par fc[-1]
        try:
            if result is None or i % refit_every == 0:
                result      = ARIMA(history, order=order).fit()
                last_result = result
                if initial_result is None:
                    initial_result = result   # premier fit == train seul
            else:
                result      = result.append([history[-1]], refit=False)
                last_result = result

            fc     = result.forecast(steps=horizon)
            fc_val = fc[-1] if hasattr(fc, "__len__") else fc.iloc[-1]

            if target_idx < n_test:
                predictions[dates[target_idx]] = fc_val
        except Exception as e:
            n_failures += 1
            last_error  = e
            if target_idx < n_test:
                predictions[dates[target_idx]] = history[-1]

        history.append(test_values[i])

        if (i + 1) % verbose_every == 0:
            print(f"      {i+1}/{n_test} observations traitées...")

    if n_failures > 0:
        pct = n_failures / n_test * 100
        print(f"   ⚠️  {n_failures}/{n_test} prévisions ({pct:.1f}%) "
              f"ont échoué et sont retombées sur la valeur naïve "
              f"(dernière erreur : {last_error})")

    pred_series = pd.Series(predictions).reindex(dates)
    pred_series.name = f"ARIMA_J{horizon}"
    print(f"   ✅ {pred_series.notna().sum()}/{n_test} prévisions générées "
          f"(les {horizon-1} premières dates de test n'ont pas de "
          f"prévision J+{horizon} valide — aucune origine antérieure "
          f"au test disponible dans ce schéma rolling)")

    # Résidus du PREMIER fit (train seul) — FIX Finding 3 :
    # l'ancienne version utilisait last_result (dernier refit, qui
    # pour J+1/refit_every=1 est ajusté sur quasi tout train+test),
    # ce qui contaminait le test de Ljung-Box avec des résidus de
    # test period étiquetés à tort avec train.index.
    if initial_result is not None:
        resid_values = initial_result.resid
        residuals = pd.Series(
            resid_values[-len(train):],
            index=train.index
        )
    else:
        residuals = pd.Series(dtype=float)

    return pred_series, residuals

# -------------------------------------------------------------
#  Test rapide
# -------------------------------------------------------------
if __name__ == "__main__":
    from .load import load_data, get_series, split_train_test

    quick_mode = "--quick" in sys.argv

    df = load_data()
    spot, _ = get_series(df, "USD")
    train, test = split_train_test(spot)

    if quick_mode:
        print("\n⚡ MODE RAPIDE — ordre fixe (1,1,1), 30 premiers jours de test, refit tous les 10 jours\n")
        order = (1, 1, 1)
        test_small = test.iloc[:30]
        preds, _ = forecast_arima(train, test_small, order, horizon=1, refit_every=10, verbose_every=10)
    else:
        print("\n🔍 MODE COMPLET — sélection AIC + rolling complet (peut être long)\n")
        order = select_arima_order(train, max_p=6, max_q=6)
        preds, _ = forecast_arima(train, test, order, horizon=1, refit_every=1)

    print(f"\nAperçu prévisions :\n{preds.head(10)}\n")