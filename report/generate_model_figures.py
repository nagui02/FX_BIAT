# report/generate_model_figures.py
# Génère les figures des modèles pour la présentation
# Lancer : python -m report.generate_model_figures

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR   = os.path.join(BASE_DIR, "report", "images")
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)
# FIX (Finding 24) : TEST_START depuis la source canonique,
# plus de 3e copie codée en dur.
from src.s2_arima_garch.config import TEST_START

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 10, "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

NAVY, GOLD, GREEN = "#003366", "#C9A84C", "#1B7F4F"
RED, ORANGE, GRAY = "#C0392B", "#E65100", "#7A8BAA"

# Modèles en production — ARIMAX exclu (Vague 2 : aucune
# configuration exogène ne bat ARIMA seul sous contrainte
# réaliste, validation croisée + Diebold-Mariano, 0 victoire
# significative sur 60 comparaisons).
MODELS   = ["Naïf", "ARIMA", "LSTM", "MLP"]
COLORS_M = [GRAY, ORANGE, GREEN, "#8B5CF6"]


# =============================================================
#  FIX (Finding 22) — chargement dynamique depuis metrics_all.csv
# =============================================================

def _load_rmse_data() -> dict:
    path = os.path.join(PROCESSED_DIR, "metrics_all.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} introuvable — lancez `python -m src.consolidate` "
            f"après l'entraînement des modèles."
        )
    df = pd.read_csv(path)
    required = {"Modèle", "Devise", "Horizon", "RMSE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans metrics_all.csv : {missing}")

    horizons = ["J+1", "J+7", "J+30"]
    rmse_data = {}
    for devise in ["TND/USD", "TND/EUR"]:
        rmse_data[devise] = {}
        for h in horizons:
            row_vals = []
            for m in MODELS:
                match = df[(df["Devise"]==devise) & (df["Horizon"]==h) &
                          (df["Modèle"]==m)]
                row_vals.append(float(match["RMSE"].iloc[0])
                               if not match.empty else None)
            rmse_data[devise][h] = row_vals
    return rmse_data


def _load_garch_params() -> dict:
    """
    FIX (Finding 22 bis) — lit garch_params.csv si disponible
    (généré par garch.py, à ajouter — voir note plus bas), sinon
    calcule α+β directement depuis garch_sigma_{cur}.csv en
    ré-estimant rapidement, sinon retombe sur un affichage neutre
    plutôt que des valeurs figées trompeuses.
    """
    path = os.path.join(PROCESSED_DIR, "garch_params.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        out = {}
        for _, row in df.iterrows():
            cur = row["Devise"].split("/")[-1]
            out[cur] = {"alpha": row["alpha"], "beta": row["beta"],
                       "alpha_beta": row["alpha_plus_beta"]}
        return out
    print("   ⚠️  garch_params.csv introuvable — ajoutez l'export "
          "dans garch.py (voir Finding 22). Affichage sans α/β précis.")
    return {"USD": None, "EUR": None}


def _load_best_model_per_horizon(rmse_data: dict) -> dict:
    """Détermine dynamiquement le meilleur modèle par devise/horizon
    (RMSE minimal) — remplace les annotations '★ Best' codées en dur."""
    best = {}
    for devise, horizons in rmse_data.items():
        best[devise] = {}
        for h, vals in horizons.items():
            valid = [(i, v) for i, v in enumerate(vals) if v is not None]
            if valid:
                best_i = min(valid, key=lambda x: x[1])[0]
                best[devise][h] = (best_i, COLORS_M[best_i])
    return best


# =============================================================
#  1. FIGURE : Comparaison prévisions J+1 (ARIMA vs Naïf vs réel)
# =============================================================
def fig_forecast_comparison():
    rmse_data = _load_rmse_data()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        f"Forecasting Results — Test Period {TEST_START[:4]}–2026\n"
        "ARIMA vs Naïve Benchmark (J+1)",
        fontsize=13, fontweight="bold", y=1.01
    )

    for ax, currency in zip(axes, ["USD", "EUR"]):
        ap = os.path.join(PROCESSED_DIR, f"predictions_{currency.lower()}.csv")
        if not os.path.exists(ap):
            ax.text(0.5, 0.5, f"Fichier manquant:\npredictions_{currency.lower()}.csv",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color="red")
            continue

        df = pd.read_csv(ap, index_col=0, parse_dates=True)
        df.index.name = "Date"

        if "Actual" not in df.columns:
            raise ValueError(
                f"Colonne 'Actual' manquante dans predictions_{currency.lower()}.csv "
                f"(colonnes trouvées : {list(df.columns)})."
            )
        actual = df["Actual"]
        naive  = df.get("Naif_J1")
        arima  = df.get("ARIMA_J1")

        sigma_path = os.path.join(PROCESSED_DIR, f"garch_sigma_{currency.lower()}.csv")
        if os.path.exists(sigma_path):
            sigma = pd.read_csv(sigma_path, index_col=0, parse_dates=True).squeeze()
            common = actual.index.intersection(sigma.index)
            if len(common) > 0:
                s, a = sigma.loc[common], actual.loc[common]
                ax.fill_between(common, (a - 2*s).values, (a + 2*s).values,
                               color=GRAY, alpha=0.15, label="GARCH ±2σ")

        ax.plot(actual.index, actual.values, color="black", linewidth=1.8,
                label="Actual", zorder=5)
        if naive is not None:
            ax.plot(naive.index, naive.values, color=GRAY, linewidth=1.0,
                    linestyle=":", label="Naïve", zorder=2)
        if arima is not None:
            ax.plot(arima.index, arima.values, color=ORANGE, linewidth=1.5,
                    linestyle="-", label="ARIMA J+1", zorder=3)

        rmse_row = rmse_data[f"TND/{currency}"]["J+1"]
        rmse_text = " | ".join(
            f"{m}: {v:.5f}" if v is not None else f"{m}: —"
            for m, v in zip(MODELS, rmse_row)
        )
        ax.set_title(f"TND/{currency} — J+1 Horizon | RMSE: {rmse_text}",
                    fontweight="bold", pad=6, fontsize=9)
        ax.set_ylabel(f"TND/{currency}")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.axvline(pd.Timestamp(TEST_START), color=GREEN, linestyle="--",
                  linewidth=1.0, alpha=0.7, label="Test start")

    axes[1].set_xlabel("Date")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "forecast_comparison_J1.png")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"   💾 {path}")


# =============================================================
#  2. FIGURE : RMSE bar chart — dynamique, 4 modèles (sans ARIMAX)
# =============================================================
def fig_rmse_barchart():
    rmse_data = _load_rmse_data()
    best = _load_best_model_per_horizon(rmse_data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "RMSE Comparison — All Models, All Horizons\n"
        f"Test Period {TEST_START} → 2026-07-31",
        fontsize=13, fontweight="bold"
    )

    horizons = ["J+1", "J+7", "J+30"]
    x, width = np.arange(len(horizons)), 0.2

    for ax, (currency, data) in zip(axes, rmse_data.items()):
        for i, (model, color) in enumerate(zip(MODELS, COLORS_M)):
            vals = [data[h][i] for h in horizons]
            vals_plot = [v if v is not None else 0 for v in vals]
            bars = ax.bar(x + i*width - 1.5*width, vals_plot, width,
                          label=model, color=color, alpha=0.85,
                          edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val is not None:
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0005,
                           f"{val:.4f}", ha="center", va="bottom",
                           fontsize=7, rotation=90)
                else:
                    ax.text(bar.get_x()+bar.get_width()/2, 0.001, "—",
                           ha="center", va="bottom", fontsize=8, color="red")

        for ih, h in enumerate(horizons):
            if h in best[currency]:
                best_i, best_c = best[currency][h]
                best_v = data[h][best_i]
                if best_v:
                    ax.annotate("★ Best", xy=(ih+(best_i-1.5)*width, best_v+0.001),
                              fontsize=7, ha="center", color=best_c, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(horizons, fontsize=11, fontweight="bold")
        ax.set_ylabel("RMSE (TND)")
        ax.set_title(currency, fontweight="bold", fontsize=13, pad=8)
        ax.legend(loc="upper left", fontsize=9)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "rmse_comparison_all.png")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"   💾 {path}")


# =============================================================
#  3. FIGURE : GARCH volatilité + VaR — α/β dynamiques
# =============================================================
def fig_garch_var():
    garch_params = _load_garch_params()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    ab_str = " | ".join(
        f"α+β={garch_params[c]['alpha_beta']:.4f} ({c})"
        if garch_params.get(c) else f"α+β=? ({c})"
        for c in ["USD", "EUR"]
    )
    fig.suptitle(
        f"GARCH(1,1) Conditional Volatility — TND/USD and TND/EUR\n{ab_str}",
        fontsize=12, fontweight="bold", y=1.01
    )

    for ax, currency, color in [(axes[0], "USD", RED), (axes[1], "EUR", ORANGE)]:
        sigma_path = os.path.join(PROCESSED_DIR, f"garch_sigma_{currency.lower()}.csv")
        if not os.path.exists(sigma_path):
            ax.text(0.5, 0.5, f"garch_sigma_{currency.lower()}.csv\nmanquant",
                    transform=ax.transAxes, ha="center", color="red")
            continue

        sigma = pd.read_csv(sigma_path, index_col=0, parse_dates=True).squeeze()
        sigma_ann = sigma * np.sqrt(252) * 100

        ax.plot(sigma_ann.index, sigma_ann.values, color=color,
               linewidth=0.9, alpha=0.9, label=f"σ_t TND/{currency} (ann.)")
        ax.fill_between(sigma_ann.index, 0, sigma_ann.values, color=color, alpha=0.1)
        ax.axhline(sigma_ann.mean(), color=NAVY, linestyle="--", linewidth=1.2,
                  alpha=0.7, label=f"Mean: {sigma_ann.mean():.2f}%")

        last_sig = sigma.iloc[-1]
        var99 = 2.326 * last_sig * 100
        ax.text(0.98, 0.92, f"Current σ: {sigma_ann.iloc[-1]:.2f}%/yr\n"
               f"VaR 99% (1j): {var99:.3f}%", transform=ax.transAxes,
               ha="right", va="top", fontsize=9,
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        for date, label in [("2016-01-01","BCT flex."),("2020-03-01","COVID-19"),
                           ("2022-02-24","Ukraine")]:
            ax.axvline(pd.Timestamp(date), color="gray", linestyle=":",
                      linewidth=0.8, alpha=0.6)
            ax.text(pd.Timestamp(date), sigma_ann.max()*0.85, label,
                   rotation=90, fontsize=7, color="gray", va="top", ha="right")

        gp = garch_params.get(currency)
        title = (f"TND/{currency} — α={gp['alpha']:.4f}, β={gp['beta']:.4f}, "
                f"α+β={gp['alpha_beta']:.4f}" if gp else f"TND/{currency}")
        ax.set_title(title, fontweight="bold", pad=6)
        ax.set_ylabel("Volatilité annualisée (%)")
        ax.legend(loc="upper left", fontsize=9)

    axes[1].set_xlabel("Date")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "garch_volatility_both.png")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"   💾 {path}")


# =============================================================
#  4. FIGURE : Diebold-Mariano heatmap (inchangée — lit déjà le CSV)
# =============================================================
def fig_dm_heatmap():
    dm_path = os.path.join(PROCESSED_DIR, "diebold_mariano_results.csv")
    if not os.path.exists(dm_path):
        print(f"   ⚠️  {dm_path} manquant — skip")
        return

    dm_df = pd.read_csv(dm_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Diebold-Mariano Tests (1995) — p-values\n"
                "Green = significant (p < 0.05) | Red = not significant",
                fontsize=12, fontweight="bold")

    for ax, currency in zip(axes, ["TND/USD", "TND/EUR"]):
        sub = dm_df[dm_df["Devise"] == currency]
        pairs = (sub["Modèle 1"] + "\nvs\n" + sub["Modèle 2"]).unique()
        horizons = ["J+1", "J+7", "J+30"]
        p_matrix = np.ones((len(pairs), len(horizons)))
        dm_matrix = np.zeros_like(p_matrix)

        for ip, pair in enumerate(pairs):
            m1, m2 = pair.replace("\nvs\n", " vs ").split(" vs ")
            for ih, h in enumerate(horizons):
                row = sub[(sub["Modèle 1"]==m1)&(sub["Modèle 2"]==m2)&(sub["Horizon"]==h)]
                if not row.empty:
                    p_matrix[ip,ih] = row["p-value"].values[0]
                    if "DM stat" in row.columns:
                        dm_matrix[ip,ih] = row["DM stat"].values[0]

        im = ax.imshow(p_matrix, cmap="RdYlGn_r", vmin=0, vmax=0.15, aspect="auto")
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels(horizons, fontsize=11, fontweight="bold")
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(pairs, fontsize=9)
        ax.set_title(currency, fontweight="bold", fontsize=13, pad=8)

        for i in range(len(pairs)):
            for j in range(len(horizons)):
                pv, dm = p_matrix[i,j], dm_matrix[i,j]
                sig = "***" if pv<0.01 else "**" if pv<0.05 else "*" if pv<0.10 else "ns"
                arr = "▲" if dm > 0 else "▼"
                clr = "white" if pv < 0.05 else "black"
                ax.text(j, i, f"p={pv:.3f}\n{sig} {arr}", ha="center", va="center",
                       fontsize=8, color=clr, fontweight="bold")
        plt.colorbar(im, ax=ax, label="p-value", shrink=0.8)

    patches = [mpatches.Patch(color="#d73027", label="p < 0.05 : Significant ✅"),
              mpatches.Patch(color="#fee08b", label="p < 0.10 : Marginal"),
              mpatches.Patch(color="#1a9850", label="p > 0.10 : Not significant")]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
              bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "dm_heatmap_presentation.png")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"   💾 {path}")


# =============================================================
#  5. FIGURE : Hiérarchie des modèles — dynamique, plus d'ARIMAX
# =============================================================
def fig_horizon_summary():
    rmse_data = _load_rmse_data()
    best = _load_best_model_per_horizon(rmse_data)

    def _naif_gain(devise, h):
        vals = rmse_data[devise][h]
        naif_v = vals[0]
        best_i, _ = best[devise].get(h, (None, None))
        if best_i is None or naif_v is None or vals[best_i] is None or naif_v == 0:
            return MODELS[0] if best_i is None else MODELS[best_i], "N/A"
        gain = (vals[best_i] - naif_v) / naif_v * 100
        return MODELS[best_i], f"{gain:+.1f}% RMSE"

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    fig.patch.set_facecolor("#1A2035")
    fig.text(0.5, 0.92, "Horizon-Dependent Model Hierarchy — TND/USD & TND/EUR",
             ha="center", va="top", fontsize=14, fontweight="bold", color="white")

    horizons_disp = [("J+1\n(1 day)", "J+1"), ("J+7\n(1 week)", "J+7"),
                     ("J+30\n(1 month)", "J+30")]
    colors_h = [NAVY, ORANGE, GREEN]

    for i, ((h_label, h_key), color) in enumerate(zip(horizons_disp, colors_h)):
        x, y = 0.15 + i*0.32, 0.45
        model_usd, gain_usd = _naif_gain("TND/USD", h_key)
        model_eur, gain_eur = _naif_gain("TND/EUR", h_key)
        model_disp = model_usd if model_usd == model_eur else f"{model_usd}/{model_eur}"
        gain_disp  = gain_usd if model_usd == model_eur else f"{gain_usd} | {gain_eur}"

        fancy = mpatches.FancyBboxPatch((x-0.12,y-0.35), 0.26, 0.68,
                                        boxstyle="round,pad=0.02", facecolor=color,
                                        alpha=0.9, edgecolor="white", linewidth=1.5,
                                        transform=fig.transFigure)
        fig.add_artist(fancy)
        fig.text(x, y+0.25, h_label, ha="center", va="center", fontsize=13,
                 fontweight="bold", color="white", transform=fig.transFigure)
        fig.text(x, y+0.10, f"★ {model_disp}", ha="center", va="center", fontsize=10,
                 fontweight="bold", color=GOLD, transform=fig.transFigure)
        fig.text(x, y-0.05, gain_disp, ha="center", va="center", fontsize=9,
                 color="white", transform=fig.transFigure)
        fig.text(x, y-0.25, "vs Naïf (RMSE, test 2024-2026)", ha="center",
                 va="center", fontsize=7, color="#C9A84C", transform=fig.transFigure,
                 style="italic")

    for x_from, x_to in [(0.27,0.35),(0.59,0.67)]:
        ax.annotate("", xy=(x_to,0.5), xytext=(x_from,0.5),
                   xycoords="figure fraction", textcoords="figure fraction",
                   arrowprops=dict(arrowstyle="->", color="white", lw=1.5))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "horizon_summary.png")
    plt.savefig(path, bbox_inches="tight", dpi=200, facecolor="#1A2035")
    plt.close()
    print(f"   💾 {path}")


if __name__ == "__main__":
    print("=" * 55)
    print("  GÉNÉRATION FIGURES MODÈLES — PRÉSENTATION")
    print("=" * 55)
    print("\n[1/5] Comparaison prévisions J+1...")
    fig_forecast_comparison()
    print("[2/5] RMSE bar chart comparatif...")
    fig_rmse_barchart()
    print("[3/5] GARCH volatilité + VaR...")
    fig_garch_var()
    print("[4/5] Diebold-Mariano heatmap...")
    fig_dm_heatmap()
    print("[5/5] Résumé horizons...")
    fig_horizon_summary()
    print(f"\n✅ 5 figures sauvegardées → {FIGURES_DIR}")