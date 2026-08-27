# =============================================================
#  src/s5_hedging/compare.py
#  Comparaison chiffrée Forward vs Option, par profil
#  Auteur : Youssef Neji | MINDS ENIT | 2026
# =============================================================

import pandas as pd

from .config import CLIENT_PROFILES, PROCESSED_DIR
from .forward import run_all_forwards, load_latest_rates
from .options import run_all_options

def breakeven_analysis(row: dict) -> dict:
    """
    FIX : l'ancienne version ne modélisait que la branche "non
    exercée" du payoff de l'option — elle pouvait donner un seuil
    hors du domaine de validité si le payoff plafonné dominait
    déjà partout.

    CALL (importateur) : coût_option(S_T) = prime + min(S_T, K)
    PUT (exportateur)  : recette_option(S_T) = max(S_T, K) - prime
    """
    F, prime, spot = row["forward"], row["prime_unitaire"], row["spot"]
    K = spot  # strike ATM

    if row["type_option"] == "CALL":
        capped_cost = K + prime
        if capped_cost < F:
            return {
                "seuil_rentabilite_option": None,
                "mouvement_requis_pct": None,
                "condition_option_gagnante": (
                    f"Option toujours gagnante (coût plafonné "
                    f"{capped_cost:.5f} < Forward {F:.5f})"
                ),
            }
        s_star = F - prime
        mouvement_pct = (s_star - spot) / spot * 100
        condition = f"si cours final < {s_star:.5f}"
    else:  # PUT
        floor_revenue = K - prime
        if floor_revenue > F:
            return {
                "seuil_rentabilite_option": None,
                "mouvement_requis_pct": None,
                "condition_option_gagnante": (
                    f"Option toujours gagnante (recette plancher "
                    f"{floor_revenue:.5f} > Forward {F:.5f})"
                ),
            }
        s_star = F + prime
        mouvement_pct = (s_star - spot) / spot * 100
        condition = f"si cours final > {s_star:.5f}"

    return {
        "seuil_rentabilite_option": round(s_star, 5),
        "mouvement_requis_pct": round(mouvement_pct, 3),
        "condition_option_gagnante": condition,
    }


def load_forecast(currency: str, horizon: int) -> float | None:
    """
    Charge la prévision du meilleur modèle pour cet horizon,
    déterminé DYNAMIQUEMENT depuis metrics_all.csv (RMSE minimal)
    plutôt qu'un dictionnaire codé en dur qui pointait vers
    ARIMAX — un modèle explicitement écarté de la production
    après la validation croisée (Vague 2). Cette version reste
    automatiquement cohérente si le classement des modèles
    change à l'avenir, sans jamais avoir à toucher ce fichier.
    """
    metrics_path = f"{PROCESSED_DIR}/metrics_all.csv"
    try:
        metrics_df = pd.read_csv(metrics_path)
        sub = metrics_df[
            (metrics_df["Devise"] == f"TND/{currency}") &
            (metrics_df["Horizon"] == f"J+{horizon}")
        ].dropna(subset=["RMSE"])
        if sub.empty:
            return None
        best = sub.loc[sub["RMSE"].idxmin(), "Modèle"]
    except FileNotFoundError:
        return None

    pred_col_prefix = "Naif" if best == "Naïf" else best
    path = f"{PROCESSED_DIR}/predictions_{currency.lower()}.csv"
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        col = f"{pred_col_prefix}_J{horizon}"
        if col not in df.columns:
            return None
        s = df[col].dropna()
        return float(s.iloc[-1]) if len(s) > 0 else None
    except FileNotFoundError:
        return None


def recommend_instrument(row: dict) -> dict:
    forecast = load_forecast(row["devise"], row["horizon_j"])
    if forecast is None:
        return {"cours_predit": None, "recommandation_modele": "N/A"}

    s_star = row["seuil_rentabilite_option"]
    option_type = row["type_option"]

    # FIX : nouveau cas où breakeven_analysis retourne None
    # (l'option domine partout, pas de seuil réel)
    if s_star is None:
        return {
            "cours_predit": round(forecast, 5),
            "recommandation_modele": "Option"
        }

    if option_type == "CALL":
        option_gagne = forecast < s_star
    else:
        option_gagne = forecast > s_star

    return {
        "cours_predit": round(forecast, 5),
        "recommandation_modele": "Option" if option_gagne else "Forward",
    }


def build_comparison() -> pd.DataFrame:
    fwd_df = run_all_forwards()
    opt_df = run_all_options()

    merged = fwd_df.merge(
        opt_df[["profile", "devise", "horizon_j", "type_option",
               "prime_unitaire", "prime_totale_tnd",
               "prime_pct_expo", "delta"]],
        on=["profile", "devise", "horizon_j"]
    )

    merged["ecart_cout_tnd"] = (
        merged["cout_couverture_tnd"] - merged["prime_totale_tnd"]
    )
    merged["instrument_moins_cher_aujourdhui"] = merged["ecart_cout_tnd"].apply(
        lambda x: "Option" if x > 0 else "Forward"
    )

    breakevens = merged.apply(
        lambda row: breakeven_analysis(row.to_dict()), axis=1
    )
    merged = pd.concat([merged, pd.DataFrame(list(breakevens))], axis=1)

    recos = merged.apply(
        lambda row: recommend_instrument(row.to_dict()), axis=1
    )
    merged = pd.concat([merged, pd.DataFrame(list(recos))], axis=1)

    return merged



def summary_table() -> pd.DataFrame:
    df = build_comparison()
    cols = [
        "profile", "devise", "type_option", "horizon_j",
        "spot", "forward", "seuil_rentabilite_option",
        "cours_predit", "recommandation_modele"
    ]
    return df[cols].sort_values(["profile", "horizon_j"])

if __name__ == "__main__":
    print("=" * 70)
    print("  S5 — COUVERTURE : FORWARD vs OPTIONS")
    print("=" * 70)

    rates = load_latest_rates()
    print(f"\nCours de référence ({rates['date']}) :")
    print(f"  TND/USD = {rates['spot_usd']:.5f} | "
          f"TND/EUR = {rates['spot_eur']:.5f}")
    print(f"  Taux TND = {rates['r_tnd']*100:.2f}% | "
          f"Taux USD = {rates['r_usd']*100:.2f}% | "
          f"Taux EUR = {rates['r_eur']*100:.2f}%")

    print("\n📈 Forward — parité des taux couverts...")
    fwd = run_all_forwards()
    fwd.to_csv("data/processed/forward_results.csv", index=False)
    print(fwd.to_string(index=False))

    print("\n📊 Options — Garman-Kohlhagen...")
    opt = run_all_options()
    opt.to_csv("data/processed/options_results.csv", index=False)
    print(opt.to_string(index=False))

    print("\n⚖️  Comparaison Forward vs Options...")
    comp = summary_table()
    comp.to_csv("data/processed/hedging_comparison.csv", index=False)
    print(comp.to_string(index=False))

    print("\n✅ S5 (Forward + Options) terminé")