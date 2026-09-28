# =============================================================
#  src/s2_arima_garch/garch.py
#  Modèle GARCH(1,1) sur les log-rendements
#
#  Test rapide : python -m src.s2_arima_garch.garch
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arch import arch_model
import os

from .config import (
    TRAIN_START, TRAIN_END, TEST_START, TEST_END,
    FIGURES_DIR, PROCESSED_DIR
)


def fit_garch(log_returns: pd.Series, currency: str,
             dist: str = "t") -> tuple:
    """
    Ajuste GARCH(1,1) sur les log-rendements (période train uniquement).

    Modèle : σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

    FIX (Vague 4) : dist="t" par défaut au lieu de "normal" —
    cohérent avec la kurtosis excédentaire élevée du TND déjà
    établie (Chapitre 2 : 3,83 pour USD et 5,96 pour EUR) et avec les innovations Student-t
    déjà utilisées pour la VaR Monte Carlo (S4). Les innovations
    gaussiennes sous-estiment structurellement le risque de queue.

    ⚠️ Change omega/alpha/beta et sigma_t — tout ce qui consomme
    garch_sigma_{cur}.csv en aval (S3 features, S4 VaR, S5 options/
    Markowitz) doit être régénéré après ce fix.

    Returns:
        (sigma_train, result)
    """
    print(f"\n   📊 Ajustement GARCH(1,1) — TND/{currency}...")

    train_returns  = log_returns.loc[TRAIN_START:TRAIN_END].dropna()
    returns_scaled = train_returns * 100

    model  = arch_model(returns_scaled, vol="Garch", p=1, q=1,
                        mean="Constant", dist=dist)
    result = model.fit(disp="off")

    if result.convergence_flag != 0:
        print(f"   ⚠️  GARCH(1,1) TND/{currency} : NON-CONVERGENCE "
              f"(convergence_flag={result.convergence_flag}) — "
              f"paramètres potentiellement non fiables, à examiner "
              f"avant utilisation en aval (VaR, etc.)")

    omega = result.params["omega"]
    alpha = result.params["alpha[1]"]
    beta  = result.params["beta[1]"]

    print(f"   ω={omega:.6f} | α={alpha:.4f} | β={beta:.4f} | α+β={alpha+beta:.4f}")
    print(f"   Distribution des innovations : {dist}"
          + (f" (nu={result.params.get('nu', float('nan')):.2f})"
             if dist == "t" else ""))
    if alpha + beta >= 1:
        print(f"   ⚠️  α+β ≥ 1 : volatilité non-stationnaire")
    else:
        print(f"   ✅ Stationnarité GARCH vérifiée")

    sigma_train = result.conditional_volatility / 100
    sigma_train.index = train_returns.index
    sigma_train.name  = f"sigma_{currency}"

    print(f"   Volatilité annualisée moy : {sigma_train.mean() * np.sqrt(252) * 100:.2f}%")

    return sigma_train, result


def forecast_garch_test_period(result, log_returns: pd.Series, currency: str) -> pd.Series:
    """
    Reconstruit sigma_t sur la période de test par récursion
    GARCH(1,1) à 1 pas, jour par jour, à partir des rendements
    RÉELLEMENT réalisés — pas un forecast multi-step statique.

    FIX : l'ancienne implémentation appelait
    result.forecast(horizon=len(test_returns)) une seule fois à la
    fin du train. Un forecast multi-step de ce type converge
    mathématiquement vers la variance inconditionnelle de long
    terme ω/(1-α-β) à mesure que l'horizon grandit — sur une
    période de test de 675 observations, sigma_t perdait donc
    presque toute réactivité aux chocs réels (courbe plate après
    quelques mois, alors que le train montre des pics nets sur
    chaque choc). Confirmé visuellement sur
    garch_volatility_both.png avant correction.

    Ici, sigma_t est recalculé récursivement :

        ε_{t-1} = r_{t-1} - μ
        σ²_t    = ω + α·ε²_{t-1} + β·σ²_{t-1}

    en utilisant à chaque pas le rendement RÉEL r_{t-1} (déjà
    observé au moment de calculer σ_t), et non une espérance
    modélisée. Aucun look-ahead bias : σ_t n'utilise jamais r_t
    lui-même, seulement l'information disponible jusqu'à t-1 —
    même logique que le rolling window déjà utilisé pour ARIMA
    (Section 3.3.3 du rapport). Les paramètres ω/α/β restent ceux
    estimés sur la période d'entraînement uniquement (pas de
    ré-estimation pendant le test), cohérent avec un déploiement
    opérationnel réaliste.

    La récursion redémarre depuis la dernière variance
    conditionnelle et le dernier résidu au carré estimés en fin
    d'entraînement (pas depuis la variance inconditionnelle), pour
    assurer une transition continue entre train et test.
    """
    test_returns        = log_returns.loc[TEST_START:TEST_END].dropna()
    test_returns_scaled = test_returns * 100  # même échelle que le fit

    omega = result.params["omega"]
    alpha = result.params["alpha[1]"]
    beta  = result.params["beta[1]"]
    mu    = result.params.get("mu", 0.0)

    prev_sigma2 = result.conditional_volatility.iloc[-1] ** 2
    prev_resid2 = result.resid.iloc[-1] ** 2

    sigma2_scaled = np.empty(len(test_returns_scaled))
    for i, r_t in enumerate(test_returns_scaled.values):
        cur_sigma2       = omega + alpha * prev_resid2 + beta * prev_sigma2
        sigma2_scaled[i] = cur_sigma2
        resid_t          = r_t - mu
        prev_sigma2, prev_resid2 = cur_sigma2, resid_t ** 2

    sigma_test = np.sqrt(sigma2_scaled) / 100  # retour à l'échelle originale

    sigma_test_series = pd.Series(
        sigma_test,
        index=test_returns.index,
        name=f"sigma_{currency}"
    )
    return sigma_test_series


def save_garch_params(results: dict, processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """
    Sauvegarde ω/α/β/(α+β) pour chaque devise dans garch_params.csv
    — consommé par report/generate_all_figures.py (fig_garch_var)
    pour afficher les vraies valeurs dans les titres au lieu de
    'α+β=?'.

    Args:
        results : dict {"USD": result_usd, "EUR": result_eur} ---
            objets retournés par fit_garch() (le 2e élément du tuple).

    À appeler depuis run_s2.py juste après avoir ajusté GARCH pour
    les deux devises, par exemple :

        _, result_usd = fit_garch(logret_usd, "USD")
        _, result_eur = fit_garch(logret_eur, "EUR")
        save_garch_params({"USD": result_usd, "EUR": result_eur})
    """
    rows = []
    for cur, result in results.items():
        alpha = result.params["alpha[1]"]
        beta  = result.params["beta[1]"]
        rows.append({
            "Devise": f"TND/{cur}",
            "omega": result.params["omega"],
            "alpha": alpha,
            "beta": beta,
            "alpha_plus_beta": alpha + beta,
        })
    df = pd.DataFrame(rows)
    path = os.path.join(processed_dir, "garch_params.csv")
    df.to_csv(path, index=False)
    print(f"   💾 {path}")
    return df


def plot_garch_volatility(sigma: pd.Series, currency: str):
    """Trace la volatilité conditionnelle GARCH annualisée."""
    sigma_annual = sigma * np.sqrt(252) * 100

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(sigma_annual.index, sigma_annual.values, color="darkred", linewidth=0.8)
    ax.set_title(f"Volatilité conditionnelle GARCH(1,1) — TND/{currency} (annualisée %)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatilité annualisée (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    fname = os.path.join(FIGURES_DIR, f"garch_volatility_{currency.lower()}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 Figure sauvegardée → {fname}")


# -------------------------------------------------------------
#  Test rapide
# -------------------------------------------------------------
if __name__ == "__main__":
    from .load import load_data, get_series

    df = load_data()
    results = {}

    for currency in ["USD", "EUR"]:
        _, logret = get_series(df, currency)

        sigma_train, result = fit_garch(logret, currency)
        sigma_test = forecast_garch_test_period(result, logret, currency)
        results[currency] = result

        sigma_full = pd.concat([sigma_train, sigma_test])
        plot_garch_volatility(sigma_full, currency)

        out_path = os.path.join(PROCESSED_DIR, f"garch_sigma_{currency.lower()}.csv")
        sigma_full.to_csv(out_path)
        print(f"   💾 {out_path}")

        print(f"\nAperçu sigma_t (train, {currency}) :\n{sigma_train.head(3)}")
        print(f"\nAperçu sigma_t (test, {currency}) :\n{sigma_test.head(3)}")
        print(f"\nVolatilité annualisée sur le test : "
              f"min={sigma_test.min()*np.sqrt(252)*100:.2f}% | "
              f"max={sigma_test.max()*np.sqrt(252)*100:.2f}% "
              f"(doit varier — une courbe quasi plate signale une "
              f"régression vers l'ancien comportement)")

    save_garch_params(results)