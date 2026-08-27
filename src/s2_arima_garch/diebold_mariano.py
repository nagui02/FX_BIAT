# =============================================================
#  src/s2_arima_garch/diebold_mariano.py
#  Test de Diebold-Mariano (1995)
#  Comparaison formelle des modèles de prévision
#  Auteur : Youssef Neji | MINDS ENIT | 2025
#
#  Usage : python -m src.s2_arima_garch.diebold_mariano
# =============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

warnings.filterwarnings("ignore")

from .config import (
    PROCESSED_DIR, FIGURES_DIR,
    TRAIN_START, TRAIN_END,
    TEST_START, TEST_END,
    HORIZONS
)

os.makedirs(FIGURES_DIR, exist_ok=True)


# =============================================================
#  1. TEST DE DIEBOLD-MARIANO
# =============================================================

def diebold_mariano_test(
    actual:       pd.Series,
    pred1:        pd.Series,
    pred2:        pd.Series,
    model1_name:  str,
    model2_name:  str,
    horizon:      int = 1,
    loss:         str = "squared"
) -> dict:
    """
    Test de Diebold-Mariano (1995) — comparaison de précision
    prédictive entre deux modèles.

    Hypothèses :
      H0 : les deux modèles ont la même précision prédictive
           E[d_t] = 0  où  d_t = L(e1_t) - L(e2_t)
      H1 : le modèle 2 est plus précis que le modèle 1
           E[d_t] > 0

    Statistique DM (Harvey, Leybourne & Newbold, 1997
    correction pour petits échantillons) :
      d_t   = L(e1_t) - L(e2_t)
      DM    = d_bar / sqrt(LRV / n)
      DM*   = sqrt((n + 1 - 2h + h(h-1)/n) / n) * DM
      DM* ~ t(n-1) asymptotiquement

    Correction HAC (Newey-West) pour l'autocorrélation
    due aux prévisions multi-pas (horizon > 1).

    FIX (Vague 2, Finding 9) : l'ancienne version mélangeait deux
    conventions de dénominateur — np.var(d, ddof=1) divise par
    (n-1) pour gamma0, tandis que np.cov(...)[0,1] (ddof=1 par
    défaut) divise par (n-k-1) ET démoyenne chaque sous-tableau
    par SA PROPRE moyenne plutôt que par la moyenne de l'échantillon
    complet. Le Newey-West standard utilise un dénominateur unique
    (n) et un démoyennage unique par la moyenne globale — appliqué
    ci-dessous.

    Args:
        actual       : valeurs réelles (pd.Series avec DatetimeIndex)
        pred1        : prévisions modèle 1 — référence (benchmark)
        pred2        : prévisions modèle 2 — challenger
        model1_name  : nom du modèle 1
        model2_name  : nom du modèle 2
        horizon      : horizon de prévision h (pour correction HAC)
        loss         : "squared" (RMSE-based) ou "absolute" (MAE-based)

    Returns:
        dict avec statistique DM, p-value, conclusion
    """
    # Aligner les trois séries sur leur intersection
    common = (actual.index
              .intersection(pred1.index)
              .intersection(pred2.index))

    if len(common) < 10:
        return {
            "Modèle 1":     model1_name,
            "Modèle 2":     model2_name,
            "Horizon":      f"J+{horizon}",
            "Loss":         loss,
            "DM stat":      np.nan,
            "p-value":      np.nan,
            "N obs":        len(common),
            "Significatif": False,
            "Gagnant":      "Indéterminé",
            "Conclusion":   "Pas assez d'observations communes",
        }

    a  = actual.loc[common].values
    p1 = pred1.loc[common].values
    p2 = pred2.loc[common].values

    # Sécurité supplémentaire : certaines séries de prévisions
    # (LSTM/MLP notamment) sont sauvegardées dans un DataFrame
    # multi-horizons dont les colonnes ont des plages de dates
    # valides différentes (l'horizon le plus long démarre plus
    # tard à cause du warmup). pandas complète alors les dates
    # manquantes par NaN lors de la construction du DataFrame.
    # L'intersection d'index ci-dessus ne filtre que sur les
    # LABELS de date, pas sur la présence de NaN dans les valeurs
    # — sans ce filtre, un seul NaN dans le tableau ferait
    # basculer TOUT le calcul (moyenne, variance) en NaN.
    valid = ~(np.isnan(a) | np.isnan(p1) | np.isnan(p2))
    n_dropped = (~valid).sum()
    a, p1, p2 = a[valid], p1[valid], p2[valid]

    if len(a) < 10:
        return {
            "Modèle 1":     model1_name,
            "Modèle 2":     model2_name,
            "Horizon":      f"J+{horizon}",
            "Loss":         loss,
            "DM stat":      np.nan,
            "p-value":      np.nan,
            "N obs":        len(a),
            "Significatif": False,
            "Gagnant":      "Indéterminé",
            "Conclusion":   (f"Pas assez d'observations valides après "
                             f"exclusion des NaN ({n_dropped} exclues)"),
        }

    e1 = a - p1
    e2 = a - p2

    # Différentiel de perte
    if loss == "squared":
        d = e1**2 - e2**2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Loss inconnue : {loss}")

    n     = len(d)
    d_bar = d.mean()

    # FIX Finding 9 : démoyennage UNIQUE par la moyenne globale,
    # dénominateur UNIQUE (/n) pour gamma0 ET pour chaque
    # autocovariance — convention Newey-West standard, cohérente
    # du début à la fin (au lieu de mélanger ddof=1 sur gamma0
    # et ddof=1 par sous-tableau sur les covariances).
    d_dev = d - d_bar

    gamma0 = np.sum(d_dev ** 2) / n
    gamma  = gamma0

    for k in range(1, horizon):
        if k < n:
            cov_k  = np.sum(d_dev[k:] * d_dev[:-k]) / n
            weight = 1 - k / horizon
            gamma += 2 * weight * cov_k

    var_d   = max(gamma / n, 1e-12)
    dm_stat = d_bar / np.sqrt(var_d)

    # Correction Harvey-Leybourne-Newbold pour petits échantillons
    hln_factor = np.sqrt(
        (n + 1 - 2*horizon + horizon*(horizon-1)/n) / n
    )
    dm_star = hln_factor * dm_stat

    # p-value bilatérale — loi t(n-1)
    p_value = 2 * (1 - stats.t.cdf(abs(dm_star), df=n-1))

    # Conclusion
    if p_value > 0.10:
        sig_str    = "ns"
        conclusion = (f"No significant difference "
                      f"(p={p_value:.4f})")
        winner     = "Statistical tie"
    elif dm_star > 0:
        sig_str    = ("***" if p_value < 0.01 else
                      "**"  if p_value < 0.05 else "*")
        conclusion = (f"{model2_name} significantly better "
                      f"(p={p_value:.4f}) {sig_str}")
        winner     = model2_name
    else:
        sig_str    = ("***" if p_value < 0.01 else
                      "**"  if p_value < 0.05 else "*")
        conclusion = (f"{model1_name} significantly better "
                      f"(p={p_value:.4f}) {sig_str}")
        winner     = model1_name

    return {
        "Modèle 1":     model1_name,
        "Modèle 2":     model2_name,
        "Horizon":      f"J+{horizon}",
        "Loss":         loss,
        "DM stat":      round(dm_star, 4),
        "p-value":      round(p_value, 4),
        "N obs":        n,
        "Significatif": p_value < 0.05,
        "Gagnant":      winner,
        "Conclusion":   conclusion,
    }


# =============================================================
#  2. CHARGEMENT DES PRÉVISIONS
# =============================================================

def load_all_predictions(currency: str) -> dict:
    """
    Charge toutes les prévisions disponibles pour une devise.

    Sources :
      - arima_predictions_{currency}.csv  : Naïf + ARIMA
      - arimax_predictions_{currency}.csv : ARIMAX
      - lstm_predictions_{currency}.csv   : LSTM
      - mlp_predictions_{currency}.csv    : MLP

    Returns:
        dict {model_name: pd.Series} pour chaque horizon
    """
    preds = {}

    # ARIMA + Naïf
    arima_path = os.path.join(
        PROCESSED_DIR,
        f"arima_predictions_{currency.lower()}.csv"
    )
    if os.path.exists(arima_path):
        df = pd.read_csv(arima_path, index_col=0,
                         parse_dates=True)
        df.index.name = "Date"
        preds["arima_df"] = df
        print(f"   ✅ ARIMA/Naïf chargé : {df.shape}")
    else:
        print(f"   ❌ ARIMA manquant : {arima_path}")
        preds["arima_df"] = None

    # ARIMAX
    arimax_path = os.path.join(
        PROCESSED_DIR,
        f"arimax_predictions_{currency.lower()}.csv"
    )
    if os.path.exists(arimax_path):
        df = pd.read_csv(arimax_path, index_col=0,
                         parse_dates=True)
        df.index.name = "Date"
        preds["arimax_df"] = df
        print(f"   ✅ ARIMAX chargé : {df.shape}")
    else:
        print(f"   ⚠️  ARIMAX non trouvé")
        preds["arimax_df"] = None

    # LSTM
    lstm_path = os.path.join(
        PROCESSED_DIR,
        f"lstm_predictions_{currency.lower()}.csv"
    )
    if os.path.exists(lstm_path):
        df = pd.read_csv(lstm_path, index_col=0,
                         parse_dates=True)
        df.index.name = "Date"
        preds["lstm_df"] = df
        print(f"   ✅ LSTM chargé : {df.shape}")
    else:
        print(f"   ⚠️  LSTM non trouvé")
        preds["lstm_df"] = None

    # MLP
    mlp_path = os.path.join(
        PROCESSED_DIR,
        f"mlp_predictions_{currency.lower()}.csv"
    )
    if os.path.exists(mlp_path):
        df = pd.read_csv(mlp_path, index_col=0,
                         parse_dates=True)
        df.index.name = "Date"
        preds["mlp_df"] = df
        print(f"   ✅ MLP chargé : {df.shape}")
    else:
        print(f"   ⚠️  MLP non trouvé")
        preds["mlp_df"] = None

    return preds


# =============================================================
#  3. BATTERIE DE TESTS PAR DEVISE
# =============================================================

def run_all_dm_tests(currency: str) -> pd.DataFrame:
    """
    Lance tous les tests DM pour une devise.

    Paires testées pour chaque horizon : toutes les combinaisons
    2 à 2 parmi les modèles disponibles (Naïf, ARIMA, ARIMAX,
    LSTM, MLP).

    Loss : squared (RMSE-based) — standard en finance

    Returns:
        DataFrame avec tous les résultats DM
    """
    print(f"\n{'='*60}")
    print(f"  TESTS DIEBOLD-MARIANO — TND/{currency}")
    print(f"  Test period : {TEST_START} → {TEST_END}")
    print(f"{'='*60}")

    preds     = load_all_predictions(currency)
    arima_df  = preds.get("arima_df")
    arimax_df = preds.get("arimax_df")
    lstm_df   = preds.get("lstm_df")
    mlp_df    = preds.get("mlp_df")

    if arima_df is None:
        print("   ❌ Impossible de lancer les tests sans ARIMA")
        return pd.DataFrame()

    # Défini une seule fois : indépendant de l'horizon, donc sorti
    # de la boucle pour éviter tout risque de NameError ou de
    # réutilisation silencieuse d'une valeur d'un horizon précédent
    # si un horizon donné n'a pas de colonne Naif_J{h} correspondante.
    actual = arima_df["Actual"] if "Actual" in arima_df.columns \
             else arima_df.iloc[:, 0]

    results = []

    for horizon in HORIZONS:
        print(f"\n--- Horizon J+{horizon} ---")

        naive_col  = f"Naif_J{horizon}"
        arima_col  = f"ARIMA_J{horizon}"
        arimax_col = f"ARIMAX_J{horizon}"
        lstm_col   = f"LSTM_J{horizon}"
        mlp_col    = f"MLP_J{horizon}"

        # Récupérer les séries disponibles
        model_series = {}

        if naive_col in arima_df.columns and arima_col in arima_df.columns:
            model_series["Naïf"]  = arima_df[naive_col]
            model_series["ARIMA"] = arima_df[arima_col]

        if arimax_df is not None and arimax_col in arimax_df.columns:
            model_series["ARIMAX"] = arimax_df[arimax_col]

        if lstm_df is not None and lstm_col in lstm_df.columns:
            model_series["LSTM"] = lstm_df[lstm_col]

        if mlp_df is not None and mlp_col in mlp_df.columns:
            model_series["MLP"] = mlp_df[mlp_col]

        if len(model_series) < 2:
            print(f"   ⚠️  Pas assez de modèles disponibles")
            continue

        # Toutes les paires de comparaison
        model_names = list(model_series.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m1   = model_names[i]
                m2   = model_names[j]
                p1   = model_series[m1]
                p2   = model_series[m2]

                r = diebold_mariano_test(
                    actual, p1, p2, m1, m2,
                    horizon=horizon,
                    loss="squared"
                )
                r["Devise"] = f"TND/{currency}"
                results.append(r)

                sig  = "✅" if r["Significatif"] else "➖"
                dm   = r["DM stat"]
                pv   = r["p-value"]
                win  = r["Gagnant"]

                print(f"   {sig} {m1:8} vs {m2:8} | "
                      f"DM={dm:+.3f} | "
                      f"p={pv:.4f} | "
                      f"Winner: {win}")

    return pd.DataFrame(results)


# =============================================================
#  4. VISUALISATIONS
# =============================================================

def plot_dm_heatmap(dm_df: pd.DataFrame):
    """
    Heatmap des p-values pour tous les tests DM.
    Rouge = significatif, vert = non-significatif.
    """
    if dm_df.empty:
        return

    currencies = dm_df["Devise"].unique()
    n_curr     = len(currencies)

    fig, axes = plt.subplots(
        1, n_curr, figsize=(8 * n_curr, 7)
    )
    if n_curr == 1:
        axes = [axes]

    fig.suptitle(
        "Diebold-Mariano Test (1995) — p-values\n"
        "p < 0.05 : statistically significant difference\n"
        "* p<0.10  ** p<0.05  *** p<0.01",
        fontweight="bold", fontsize=12
    )

    for ax, currency in zip(axes, currencies):
        sub = dm_df[dm_df["Devise"] == currency]

        # Créer les paires uniques
        pairs    = (sub["Modèle 1"] + " vs " +
                    sub["Modèle 2"]).unique()
        horizons = [f"J+{h}" for h in HORIZONS]

        # Matrice p-values
        p_matrix = np.full(
            (len(pairs), len(horizons)), np.nan
        )
        dm_matrix = np.full_like(p_matrix, np.nan)

        for idx_p, pair in enumerate(pairs):
            m1, m2 = pair.split(" vs ")
            for idx_h, h_str in enumerate(horizons):
                row = sub[
                    (sub["Modèle 1"] == m1) &
                    (sub["Modèle 2"] == m2) &
                    (sub["Horizon"]  == h_str)
                ]
                if not row.empty:
                    p_matrix[idx_p, idx_h]  = row["p-value"].values[0]
                    dm_matrix[idx_p, idx_h] = row["DM stat"].values[0]

        # Heatmap
        cmap = plt.cm.RdYlGn_r
        im   = ax.imshow(
            p_matrix, cmap=cmap,
            vmin=0, vmax=0.15, aspect="auto"
        )

        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels(horizons, fontsize=11,
                           fontweight="bold")
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(pairs, fontsize=9)
        ax.set_title(f"{currency}", fontweight="bold",
                     fontsize=13, pad=10)

        # Annotations : p-value + étoiles significativité
        for i in range(len(pairs)):
            for j in range(len(horizons)):
                pv = p_matrix[i, j]
                dm = dm_matrix[i, j]
                if np.isnan(pv):
                    continue

                sig = ("***" if pv < 0.01 else
                       "**"  if pv < 0.05 else
                       "*"   if pv < 0.10 else "ns")

                color = "white" if pv < 0.05 else "black"

                # Flèche indiquant le gagnant
                arrow = "↑" if dm > 0 else "↓"

                ax.text(j, i,
                        f"p={pv:.3f}\n{sig} {arrow}",
                        ha="center", va="center",
                        fontsize=8, color=color,
                        fontweight="bold")

        plt.colorbar(im, ax=ax,
                     label="p-value", shrink=0.7)

    # Légende
    legend_elements = [
        mpatches.Patch(facecolor="#d73027",
                       label="p < 0.05 : Significant"),
        mpatches.Patch(facecolor="#fee08b",
                       label="p < 0.10 : Marginal"),
        mpatches.Patch(facecolor="#1a9850",
                       label="p > 0.10 : Not significant"),
    ]
    fig.legend(handles=legend_elements,
               loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    fname = os.path.join(FIGURES_DIR,
                         "diebold_mariano_heatmap.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n   💾 {fname}")


def plot_dm_summary_table(dm_df: pd.DataFrame):
    """
    Tableau visuel résumé : quel modèle gagne chaque comparaison.
    """
    if dm_df.empty:
        return

    # Compter les victoires par modèle
    winner_counts = (dm_df[dm_df["Significatif"]]
                     ["Gagnant"].value_counts())

    if winner_counts.empty:
        print("   ⚠️  Aucune comparaison significative")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Diebold-Mariano Test — Summary of Significant Results",
        fontweight="bold", fontsize=13
    )

    # 1. Nombre de victoires
    colors = {
        "ARIMAX": "#1E88E5",
        "ARIMA":  "#FF6B35",
        "LSTM":   "#43A047",
        "MLP":    "#8E24AA",
        "Naïf":   "gray",
    }
    bar_colors = [colors.get(m, "navy")
                  for m in winner_counts.index]

    axes[0].bar(
        winner_counts.index,
        winner_counts.values,
        color=bar_colors, alpha=0.8, edgecolor="black"
    )
    axes[0].set_title(
        "Number of significant wins\n(all horizons, both currencies)",
        fontweight="bold"
    )
    axes[0].set_ylabel("Number of wins (p < 0.05)")
    for i, (model, count) in enumerate(winner_counts.items()):
        axes[0].text(i, count + 0.05, str(count),
                     ha="center", fontweight="bold")

    # 2. Tableau des résultats significatifs
    sig_df = dm_df[dm_df["Significatif"]].copy()
    sig_df = sig_df[["Devise", "Horizon",
                      "Modèle 1", "Modèle 2",
                      "DM stat", "p-value", "Gagnant"]]

    if not sig_df.empty:
        axes[1].axis("off")
        table = axes[1].table(
            cellText=sig_df.values,
            colLabels=sig_df.columns,
            cellLoc="center",
            loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)

        # Colorer les en-têtes
        for j in range(len(sig_df.columns)):
            table[0, j].set_facecolor("#1A237E")
            table[0, j].set_text_props(color="white",
                                       fontweight="bold")

        # Colorer les lignes selon le gagnant
        for i, winner in enumerate(sig_df["Gagnant"]):
            color = colors.get(winner, "white")
            for j in range(len(sig_df.columns)):
                table[i+1, j].set_facecolor(
                    color + "33"
                    if len(color) == 7 else "#F5F5F5"
                )
    else:
        axes[1].text(0.5, 0.5,
                     "No significant results",
                     ha="center", va="center",
                     transform=axes[1].transAxes,
                     fontsize=14)

    plt.tight_layout()
    fname = os.path.join(FIGURES_DIR,
                         "diebold_mariano_summary.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   💾 {fname}")


# =============================================================
#  5. PIPELINE PRINCIPAL
# =============================================================

def run_pipeline() -> pd.DataFrame:
    """
    Lance tous les tests DM sur TND/USD et TND/EUR.

    Prérequis :
      - arima_predictions_usd.csv + arima_predictions_eur.csv
      - arimax_predictions_usd.csv + arimax_predictions_eur.csv
      - lstm_predictions_usd.csv + lstm_predictions_eur.csv
      - mlp_predictions_usd.csv + mlp_predictions_eur.csv

    Sorties :
      - diebold_mariano_results.csv
      - diebold_mariano_heatmap.png
      - diebold_mariano_summary.png
    """
    print("=" * 60)
    print("  TESTS DE DIEBOLD-MARIANO (1995)")
    print(f"  Modèles : Naïf, ARIMA, ARIMAX, LSTM, MLP")
    print(f"  Horizons : {HORIZONS}")
    print(f"  Test period : {TEST_START} → {TEST_END}")
    print(f"  Loss function : Squared error (RMSE-based)")
    print(f"  Correction : Harvey-Leybourne-Newbold (HLN)")
    print("=" * 60)

    all_results = []

    for currency in ["USD", "EUR"]:
        dm_results = run_all_dm_tests(currency)
        if not dm_results.empty:
            all_results.append(dm_results)

    if not all_results:
        print("\n❌ Aucun résultat — vérifier les fichiers CSV")
        return pd.DataFrame()

    final_df = pd.concat(all_results, ignore_index=True)

    # Trier pour lisibilité
    horizon_order  = {f"J+{h}": i for i, h in
                      enumerate(HORIZONS)}
    final_df["_h"] = final_df["Horizon"].map(horizon_order)
    final_df       = final_df.sort_values(
        ["Devise", "_h", "Modèle 1"]
    ).drop(columns="_h")

    # Affichage final
    print("\n" + "=" * 60)
    print("  TABLEAU FINAL — TESTS DIEBOLD-MARIANO")
    print("=" * 60)

    display_cols = [
        "Devise", "Horizon",
        "Modèle 1", "Modèle 2",
        "DM stat", "p-value",
        "Significatif", "Gagnant"
    ]
    print(final_df[display_cols].to_string(index=False))

    # Résumé statistique
    n_total = len(final_df)
    n_sig   = final_df["Significatif"].sum()
    print(f"\n   Tests significatifs (p<0.05) : "
          f"{n_sig}/{n_total} ({n_sig/n_total*100:.1f}%)")

    if n_sig > 0:
        print("\n   Gagnants significatifs :")
        winners = (final_df[final_df["Significatif"]]
                   .groupby("Gagnant").size()
                   .sort_values(ascending=False))
        for model, count in winners.items():
            print(f"   → {model:10} : {count} victoire(s)")

    # Sauvegarde
    out_path = os.path.join(
        PROCESSED_DIR, "diebold_mariano_results.csv"
    )
    final_df.to_csv(out_path, index=False)
    print(f"\n   💾 {out_path}")

    # Visualisations
    plot_dm_heatmap(final_df)
    plot_dm_summary_table(final_df)

    print("\n" + "=" * 60)
    print("  ✅ DIEBOLD-MARIANO TERMINÉ")
    print("=" * 60)

    return final_df


# =============================================================
#  EXÉCUTION DIRECTE
# =============================================================

if __name__ == "__main__":
    results = run_pipeline()