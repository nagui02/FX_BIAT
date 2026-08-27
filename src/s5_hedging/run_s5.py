# =============================================================
#  src/s5_hedging/run_s5.py
#  S5 — Point d'entrée unique : Forward + Options + Comparaison
#       (ALPHA SARL, BETA Export, M. GAMMA, Dette BIAT) +
#       Markowitz (Desk BIAT)
#  Auteur : Youssef Neji | MINDS ENIT | 2026
#
#  Usage : python -m src.s5_hedging.run_s5
#
#  Sorties (data/processed/) :
#    - forward_results.csv        (pricing Forward, 4 profils x 2 horizons)
#    - options_results.csv        (pricing Options GK, 4 profils x 2 horizons)
#    - hedging_comparison_full.csv (fusion Forward+Options+breakeven+reco,
#                                    toutes colonnes)
#    - hedging_comparison.csv     (vue condensée — mêmes colonnes que
#                                   compare.summary_table())
#    - markowitz_summary.csv      (position actuelle / min-var / max-Sharpe)
#    - markowitz_cml.csv          (Capital Market Line, TND vs frontière FX)
#    - markowitz_frontier.csv     (points de la frontière efficiente)
# =============================================================

import os
import sys
import pandas as pd

from .config import PROCESSED_DIR, CLIENT_PROFILES
from .forward import load_latest_rates, run_all_forwards
from .options import run_all_options
from .compare import build_comparison
from .markowitz import (
    analyze_desk_biat, summary_table as markowitz_summary_table,
    capital_market_line_analysis, print_cml_conclusion,
)

# Mêmes colonnes que compare.summary_table() — dupliquées ici pour
# éviter de relancer build_comparison() une 2e fois (le résultat est
# déjà en mémoire dans comp_full). Si compare.summary_table() change
# de colonnes, mettre à jour cette liste en conséquence.
COMPARISON_SUMMARY_COLS = [
    "profile", "devise", "type_option", "horizon_j",
    "spot", "forward", "seuil_rentabilite_option",
    "cours_predit", "recommandation_modele",
]

MONO_CURRENCY_PROFILES = [
    name for name, p in CLIENT_PROFILES.items() if p["currency"] != "MULTI"
]
MULTI_CURRENCY_PROFILES = [
    name for name, p in CLIENT_PROFILES.items() if p["currency"] == "MULTI"
]


def run_forward_options_comparison() -> dict:
    """Forward + Options + comparaison/recommandation pour les profils
    mono-devise (tous les profils sauf Desk BIAT)."""
    print("\n📈 Forward — parité des taux couverts...")
    fwd = run_all_forwards()
    fwd.to_csv(os.path.join(PROCESSED_DIR, "forward_results.csv"), index=False)
    print(fwd.to_string(index=False))

    print("\n📊 Options — Garman-Kohlhagen...")
    opt = run_all_options()
    opt.to_csv(os.path.join(PROCESSED_DIR, "options_results.csv"), index=False)
    print(opt.to_string(index=False))

    print("\n⚖️  Comparaison Forward vs Options + recommandation...")
    comp_full = build_comparison()
    comp_full.to_csv(
        os.path.join(PROCESSED_DIR, "hedging_comparison_full.csv"), index=False
    )

    comp = (
        comp_full[COMPARISON_SUMMARY_COLS]
        .sort_values(["profile", "horizon_j"])
    )
    comp.to_csv(os.path.join(PROCESSED_DIR, "hedging_comparison.csv"), index=False)
    print(comp.to_string(index=False))

    return {"forward": fwd, "options": opt, "comparison_full": comp_full,
            "comparison_summary": comp}


def run_markowitz_desk_biat() -> dict:
    """Optimisation de portefeuille EUR/USD pour Desk BIAT (seul
    profil MULTI — traité à part, pas par forward/options)."""
    print("\n" + "=" * 70)
    print("  MARKOWITZ — DESK BIAT (EUR/USD)")
    print("=" * 70)

    analysis = analyze_desk_biat()
    data = analysis["data"]

    print(f"\nDonnées : {data['n_obs']} obs | {data['period']}")
    print(f"Rendement annualisé  : EUR={data['mu'][0]*100:.3f}% | "
          f"USD={data['mu'][1]*100:.3f}%")
    print(f"Taux sans risque (BCT) : {data['rf']*100:.2f}%")

    summary = markowitz_summary_table(analysis)
    summary.to_csv(os.path.join(PROCESSED_DIR, "markowitz_summary.csv"), index=False)
    print("\n📊 Comparaison des portefeuilles :")
    print(summary.to_string(index=False))

    cml = capital_market_line_analysis(data, analysis["max_sharpe"])
    print_cml_conclusion(cml)
    pd.DataFrame([cml]).to_csv(
        os.path.join(PROCESSED_DIR, "markowitz_cml.csv"), index=False
    )

    analysis["frontier"].to_csv(
        os.path.join(PROCESSED_DIR, "markowitz_frontier.csv"), index=False
    )

    return {"analysis": analysis, "summary": summary, "cml": cml}


# --------------------------------------------------------------
#  VALIDATION — checks automatiques alignés sur la checklist S5
# --------------------------------------------------------------

def validate_results(fo_results: dict, mk_results: dict) -> bool:
    """
    Vérifie les critères de la checklist S5 :
      - Pricing Forward cohérent (pas de NaN, valeurs positives)
      - Pricing Options cohérent (primes > 0, delta dans le bon signe)
      - Breakeven calculé pour chaque ligne (valeur numérique ou
        message explicite si l'option domine partout)
      - Allocation Markowitz min-variance dans une plage plausible
        (référence documentée : ~70% EUR / ~30% USD)
      - Recommandation produite pour les 5 profils clients
    Retourne True si tout passe, affiche le détail dans tous les cas.
    """
    print("\n" + "=" * 70)
    print("  VALIDATION S5")
    print("=" * 70)

    checks = []

    fwd, opt = fo_results["forward"], fo_results["options"]
    comp = fo_results["comparison_full"]

    n_expected = len(MONO_CURRENCY_PROFILES) * 2  # profils x 2 horizons
    ok = len(fwd) == n_expected and fwd["forward"].notna().all() and (fwd["forward"] > 0).all()
    checks.append(("Forward cohérent (n={}, aucune valeur nulle/négative)".format(n_expected), ok))

    ok = (
        len(opt) == n_expected
        and opt["prime_unitaire"].notna().all()
        and (opt["prime_unitaire"] > 0).all()
        and opt.apply(
            lambda r: (0 <= r["delta"] <= 1) if r["type_option"] == "CALL"
            else (-1 <= r["delta"] <= 0), axis=1
        ).all()
    )
    checks.append(("Options cohérentes (primes > 0, delta dans [0,1]/[-1,0])", ok))

    ok = comp["seuil_rentabilite_option"].notna().sum() + \
        comp["condition_option_gagnante"].notna().sum() >= len(comp)
    checks.append(("Breakeven calculé pour chaque ligne (seuil ou cas dominant)", ok))

    ok = comp["recommandation_modele"].isin(["Forward", "Option", "N/A"]).all()
    n_na = (comp["recommandation_modele"] == "N/A").sum()
    label = "Recommandation produite pour les profils mono-devise"
    if n_na > 0:
        label += f" ⚠️  {n_na} ligne(s) en N/A — forecast manquant (metrics_all.csv / predictions_*.csv absents ou incomplets)"
    checks.append((label, ok))

    min_var_w = mk_results["analysis"]["min_variance"]["weights"]
    eur_pct = min_var_w[0] * 100
    ok = 40 <= eur_pct <= 100  # plage large : alerte seulement si très éloigné
    close_to_reference = 55 <= eur_pct <= 85  # référence documentée ~70/30
    label = f"Markowitz min-variance calculé (EUR={eur_pct:.1f}% / USD={100-eur_pct:.1f}%)"
    if not close_to_reference:
        label += " ⚠️  s'écarte de la référence documentée (~70% EUR / ~30% USD) — à vérifier si les données ont changé"
    checks.append((label, ok))

    profils_couverts = set(comp["profile"].unique()) | set(MULTI_CURRENCY_PROFILES)
    ok = profils_couverts == set(CLIENT_PROFILES.keys())
    checks.append((f"Les 5 profils clients ont une analyse (4 hedging + 1 Markowitz)", ok))

    all_pass = True
    for label, ok in checks:
        status = "✅" if ok else "❌"
        print(f"   {status} {label}")
        all_pass = all_pass and ok

    print("\n" + ("✅ S5 VALIDÉ" if all_pass else "❌ S5 — des points nécessitent une vérification manuelle"))
    return all_pass


# --------------------------------------------------------------
#  PIPELINE PRINCIPAL
# --------------------------------------------------------------

def run_s5() -> dict:
    print("=" * 70)
    print("  S5 — COUVERTURE : FORWARD / OPTIONS / MARKOWITZ")
    print("=" * 70)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    rates = load_latest_rates()
    print(f"\nCours de référence ({rates['date']}) :")
    print(f"  TND/USD = {rates['spot_usd']:.5f} | TND/EUR = {rates['spot_eur']:.5f}")
    print(f"  Taux TND = {rates['r_tnd']*100:.2f}% | "
          f"Taux USD = {rates['r_usd']*100:.2f}% | "
          f"Taux EUR = {rates['r_eur']*100:.2f}%")

    fo_results = run_forward_options_comparison()
    mk_results = run_markowitz_desk_biat()

    all_pass = validate_results(fo_results, mk_results)

    print("\n" + "=" * 70)
    print("  ✅ S5 TERMINÉ" if all_pass else "  ⚠️  S5 TERMINÉ AVEC AVERTISSEMENTS")
    print("=" * 70)

    return {**fo_results, **mk_results, "validation_passed": all_pass}


if __name__ == "__main__":
    results = run_s5()
    sys.exit(0 if results["validation_passed"] else 1)