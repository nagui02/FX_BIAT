# =============================================================
#  report/generate_all_figures.py
#  Génération de TOUTES les figures du rapport (Chapitres 2 et 3)
#  Fusion de generate_figures.py (données) + generate_model_figures.py
#  (modèles) — un seul point d'entrée, un seul dossier de sortie.
#  Auteur : Youssef Neji | MINDS ENIT | 2025-2026
#
#  Lancer depuis la racine : python -m report.generate_all_figures
#
#  Toutes les dates affichées dans les titres sont dérivées
#  dynamiquement de la période réelle des données chargées — ne
#  nécessite pas de mise à jour manuelle à chaque extension du
#  dataset ou de la fenêtre train/test.
#
#  Sorties → report/images/ :
#    Chapitre 2 (données)   : cours_historiques, log_rendements,
#      distribution_logret, macro_variables, new_macro_variables,
#      correlation_matrix, regimes_change
#    Chapitre 3 (modèles)   : forecast_comparison_J1,
#      rmse_comparison_all, garch_volatility_both,
#      dm_heatmap_presentation, horizon_summary
# =============================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------
#  Configuration partagée
# -------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR   = os.path.join(BASE_DIR, "report", "images")
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)
try:
    from src.s2_arima_garch.config import TRAIN_START, TRAIN_END, TEST_START, TEST_END
    _CONFIG_OK = True
except ImportError:
    TRAIN_START, TRAIN_END = "2012-01-01", "2023-12-31"
    TEST_START,  TEST_END  = "2024-01-01", "2026-07-31"
    _CONFIG_OK = False

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 10,
    "figure.dpi":      150,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "grid.linestyle":  "--",
})

# Palette — Chapitre 2 (données)
C_USD, C_EUR   = "#1E88E5", "#FF6B35"
C_BCT          = "#43A047"
C_FED, C_BCE   = "#1E88E5", "#FF6B35"
C_BRENT, C_VIX = "#2E7D32", "#C62828"
C_IPC, C_DXY   = "#5E35B1", "#00695C"
C_US10Y, C_GOLD = "#6D4C41", "#F9A825"
C_EUR10Y       = "#4527A0"

# Palette — Chapitre 3 (modèles)
NAVY, GOLD, GREEN = "#003366", "#C9A84C", "#1B7F4F"
RED, ORANGE, GRAY = "#C0392B", "#E65100", "#7A8BAA"

# Modèles en production — ARIMAX exclu (validation croisée +
# Diebold-Mariano : 0 victoire significative sur 60 comparaisons
# sous contrainte réaliste — voir Chapitre 3, Discussion).
MODELS   = ["Naïf", "ARIMA", "LSTM", "MLP"]
COLORS_M = [GRAY, ORANGE, GREEN, "#8B5CF6"]


def period_str(df: pd.DataFrame) -> str:
    """Chaîne 'AAAA–AAAA' dérivée dynamiquement du dataset."""
    return f"{df.index.min().year}–{df.index.max().year}"


def make_regimes(df: pd.DataFrame) -> list:
    """
    Construit les 3 régimes de change réels (bornes fixes,
    ancrées dans l'histoire économique du dinar), avec la borne
    de fin du dernier régime dérivée dynamiquement de la
    dernière observation du dataset — ne devient jamais obsolète.
    """
    last_date = df.index.max().strftime("%Y-%m-%d")
    return [
        ("2004-01-01", "2011-12-31", "#90CAF9", "Quasi-fixité\n(2004–2011)"),
        ("2012-01-01", "2015-12-31", "#FFE082", "Transition\n(2012–2015)"),
        ("2016-01-01", last_date,    "#EF9A9A",
         f"Dépréciation accélérée\n(2016–{df.index.max().year})"),
    ]


def add_regime_bands(ax, regimes):
    for start, end, color, label in regimes:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.15, color=color, label=label)


def add_regime_vlines(ax):
    for date, color, lw in [
        ("2012-01-01", "gray",  1.0),
        ("2016-01-01", "red",   1.0),
    ]:
        ax.axvline(pd.Timestamp(date), color=color,
                   linestyle="--", linewidth=lw, alpha=0.6)


def add_train_test_line(ax, label_y_frac=0.95):
    """
    Ligne verticale marquant la séparation train/test, dérivée
    directement de s2_arima_garch.config (TEST_START) plutôt que
    codée en dur — reste toujours synchronisée avec la fenêtre
    réellement utilisée par les modèles.
    """
    split_date = pd.Timestamp(TEST_START)
    ax.axvline(split_date, color="black", linestyle=":",
               linewidth=1.4, alpha=0.8, zorder=4)
    ymin, ymax = ax.get_ylim()
    ax.text(split_date, ymin + label_y_frac * (ymax - ymin),
            "  Train | Test", fontsize=7, color="black",
            fontweight="bold", va="top")


# =============================================================
#  ============  CHAPITRE 2 — FIGURES DONNÉES  ================
# =============================================================

# --------------------- Figure 2.1 -----------------------------
def fig_cours_historiques(df, regimes):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"Évolution des cours spot TND/USD et TND/EUR ({period_str(df)})\n"
        "Source : Banque Centrale de Tunisie (BCT)",
        fontsize=13, fontweight="bold", y=1.01
    )

    for ax, col, color, label, ylabel in [
        (axes[0], "TND_USD", C_USD, "TND/USD", "TND pour 1 USD"),
        (axes[1], "TND_EUR", C_EUR, "TND/EUR", "TND pour 1 EUR"),
    ]:
        add_regime_bands(ax, regimes)
        ax.plot(df.index, df[col], color=color,
                linewidth=1.2, label=label, zorder=3)
        add_regime_vlines(ax)
        add_train_test_line(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(label, fontweight="bold", pad=6)

        idx_max = df[col].idxmax()
        idx_min = df[col].idxmin()
        ax.annotate(
            f"Max : {df[col].max():.4f}\n({idx_max.strftime('%b %Y')})",
            xy=(idx_max, df[col].max()),
            xytext=(20, -30), textcoords="offset points",
            fontsize=8, color="darkred",
            arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8)
        )
        ax.annotate(
            f"Min : {df[col].min():.4f}\n({idx_min.strftime('%b %Y')})",
            xy=(idx_min, df[col].min()),
            xytext=(20, 20), textcoords="offset points",
            fontsize=8, color="darkgreen",
            arrowprops=dict(arrowstyle="->", color="darkgreen", lw=0.8)
        )

    patches = [
        mpatches.Patch(color=c, alpha=0.4, label=l.replace("\n", " "))
        for _, _, c, l in regimes
    ]
    axes[0].legend(handles=patches, loc="upper left",
                   fontsize=8, framealpha=0.9)

    axes[1].set_xlabel("Date")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "cours_historiques.png")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"   💾 {path}")


# --------------------- Figure 2.2 -----------------------------
def fig_log_rendements(df):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        f"Log-rendements journaliers TND/USD et TND/EUR ({period_str(df)})\n"
        r"$r_t = \ln(S_t / S_{t-1})$",
        fontsize=13, fontweight="bold", y=1.01
    )

    for ax, col, color, label in [
        (axes[0], "LogRet_TND_USD", C_USD, "LogRet TND/USD"),
        (axes[1], "LogRet_TND_EUR", C_EUR, "LogRet TND/EUR"),
    ]:
        ax.plot(df.index, df[col], color=color,
                linewidth=0.6, alpha=0.85, label=label)
        ax.axhline(0, color="black", linewidth=0.8,
                   linestyle="--", alpha=0.5)
        add_regime_vlines(ax)

        sigma = df[col].std()
        ax.fill_between(df.index, -2*sigma, 2*sigma,
                         color=color, alpha=0.07,
                         label=r"Zone $\pm 2\sigma$")

        std_val = df[col].std() * np.sqrt(252) * 100
        ax.set_title(
            f"{label} — Vol. annualisée : {std_val:.2f}%",
            fontweight="bold", pad=6
        )
        ax.set_ylabel("Log-rendement")
        ax.legend(loc="upper left", fontsize=8)

    axes[1].set_xlabel("Date")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "log_rendements.png")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"   💾 {path}")


# --------------------- Figure 2.4 (variables d'origine) --------
def fig_macro_variables(df):
    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    fig.suptitle(
        "Variables macroéconomiques retenues comme features\n"
        "Sources : FRED (Federal Reserve Bank of St. Louis), BCT, INS",
        fontsize=13, fontweight="bold", y=1.01
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df.index, df["Brent_USD"], color=C_BRENT,
             linewidth=1.0, label="Brent USD/baril")
    ax1.fill_between(df.index, 0, df["Brent_USD"], color=C_BRENT, alpha=0.08)
    add_regime_vlines(ax1)
    ax1.set_title("Prix du Brent (USD/baril)\nCode FRED : DCOILBRENTEU",
                   fontweight="bold", pad=6)
    ax1.set_ylabel("USD/baril")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df.index, df["VIX"], color=C_VIX, linewidth=0.8, label="VIX")
    ax2.axhline(20, color="orange", linestyle="--",
                linewidth=1.0, alpha=0.7, label="Seuil stress (20)")
    ax2.fill_between(df.index, df["VIX"], 20, where=df["VIX"] > 20,
                      color="red", alpha=0.15, label="Zone de stress")
    add_regime_vlines(ax2)
    ax2.set_title("Indice VIX — Volatilité mondiale\nCode FRED : VIXCLS",
                   fontweight="bold", pad=6)
    ax2.set_ylabel("Indice VIX")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df.index, df["TauxFed_USD"], color=C_FED,
             linewidth=1.2, label="Taux Fed (USD)")
    ax3.plot(df.index, df["TauxBCE_EUR"], color=C_BCE,
             linewidth=1.2, label="Taux BCE (EUR)")
    ax3.plot(df.index, df["TauxMMBCT"], color=C_BCT,
             linewidth=1.5, label="Taux BCT (TND)", linestyle="-.")
    add_regime_vlines(ax3)
    ax3.set_title("Taux directeurs Fed / BCE / BCT (%)\nFRED : DFF · ECBDFR · Source : BCT",
                   fontweight="bold", pad=6)
    ax3.set_ylabel("Taux (%)")
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df.index, df["EURUSD"], color="#8E24AA",
             linewidth=1.0, label="EUR/USD")
    ax4.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    add_regime_vlines(ax4)
    ax4.set_title("Taux de change EUR/USD\nCode FRED : DEXUSEU",
                   fontweight="bold", pad=6)
    ax4.set_ylabel("EUR pour 1 USD")
    ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(df.index, df["IPC_USA"], color=C_IPC,
             linewidth=1.2, label="IPC USA")
    ax5.fill_between(df.index, df["IPC_USA"].min(), df["IPC_USA"],
                      color=C_IPC, alpha=0.08)
    add_regime_vlines(ax5)
    ax5.set_title("IPC États-Unis (base 1982–84 = 100)\nCode FRED : CPIAUCSL",
                   fontweight="bold", pad=6)
    ax5.set_ylabel("Indice IPC")
    ax5.set_xlabel("Date")
    ax5.legend(fontsize=8)

    ax6 = fig.add_subplot(gs[2, 1])
    ipc_tun = df["IPC_Tunisie"].dropna()
    ax6.plot(ipc_tun.index, ipc_tun.values,
             color="#F57F17", linewidth=1.5, label="IPC Tunisie")
    ax6.fill_between(ipc_tun.index, 100, ipc_tun.values,
                      color="#F57F17", alpha=0.12)
    ax6.axhline(100, color="gray", linestyle="--",
                linewidth=0.8, alpha=0.6, label="Base 100 (2015)")
    ax6.set_title("IPC Tunisie (base 100 = 2015)\nSource : INS Tunisie",
                   fontweight="bold", pad=6)
    ax6.set_ylabel("Indice IPC")
    ax6.set_xlabel("Date")
    ax6.legend(fontsize=8)
    ax6.text(0.02, 0.15,
             "⚠ Données disponibles\nuniquement 2015–2025",
             transform=ax6.transAxes,
             fontsize=8, color="darkorange",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    path = os.path.join(FIGURES_DIR, "macro_variables.png")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"   💾 {path}")


# --------------------- Figure additionnelle (DXY/US10Y/Gold/Bund) ---
def fig_new_macro_variables(df):
    """
    Variables macro complémentaires : Dollar Index (DXY), taux
    Trésor US 10 ans, prix de l'or, taux Bund EUR 10 ans — déjà
    listées au Tableau 2.1 mais pas encore illustrées par une
    figure dans le chapitre actuel. N'affiche que les colonnes
    réellement présentes (robuste si une série FRED est absente).
    """
    new_vars = [
        ("DXY",      "Dollar Index (DXY)",        C_DXY,    "FRED : DTWEXBGS"),
        ("US10Y",    "Taux Trésor US 10 ans (%)",  C_US10Y,  "FRED : DGS10"),
        ("Gold_USD", "Prix de l'or (USD/once)",    C_GOLD,   "FRED : GOLDAMGBD228NLBM"),
        ("EUR10Y",   "Taux Bund EUR 10 ans (%)",   C_EUR10Y, "FRED : IRLTLT01DEM156N"),
    ]
    available = [v for v in new_vars if v[0] in df.columns
                 and df[v[0]].notna().any()]

    if not available:
        print("   ⚠️  Aucune des nouvelles variables macro "
              "(DXY, US10Y, Gold_USD, EUR10Y) n'est disponible "
              "— figure ignorée")
        return

    n = len(available)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    fig.suptitle(
        f"Variables macroéconomiques complémentaires ({period_str(df)})\n"
        f"Source : FRED",
        fontsize=13, fontweight="bold", y=1.02
    )

    for ax, (col, label, color, src) in zip(axes, available):
        series = df[col].dropna()
        ax.plot(series.index, series.values, color=color,
                linewidth=1.1, label=label)
        ax.fill_between(series.index, series.min(), series.values,
                         color=color, alpha=0.08)
        add_regime_vlines(ax)
        ax.set_title(f"{label}\n{src}", fontweight="bold", pad=6)
        ax.set_ylabel(label)
        ax.legend(fontsize=8)

    for ax in axes[len(available):]:
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "new_macro_variables.png")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"   💾 {path}")


# --------------------- Figure 2.5 ------------------------------
def fig_correlation(df):
    base_cols = ["TND_USD", "TND_EUR", "Brent_USD", "VIX",
                 "TauxFed_USD", "TauxBCE_EUR", "TauxMMBCT",
                 "EURUSD", "IPC_USA"]
    base_labels = {"TND_USD": "TND/USD", "TND_EUR": "TND/EUR",
                   "Brent_USD": "Brent", "VIX": "VIX",
                   "TauxFed_USD": "Fed", "TauxBCE_EUR": "BCE",
                   "TauxMMBCT": "BCT", "EURUSD": "EUR/USD",
                   "IPC_USA": "IPC USA"}

    extra_cols = {"DXY": "DXY", "US10Y": "US10Y",
                  "Gold_USD": "Or", "EUR10Y": "Bund 10Y"}
    for col, label in extra_cols.items():
        if col in df.columns and df[col].notna().any():
            base_cols.append(col)
            base_labels[col] = label

    cols   = [c for c in base_cols if c in df.columns]
    labels = [base_labels[c] for c in cols]

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.suptitle(
        f"Matrice de corrélation de Pearson — Dataset complet ({period_str(df)})",
        fontsize=13, fontweight="bold", y=1.01
    )

    im = ax.imshow(corr.values, cmap=plt.cm.RdBu_r, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val   = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color,
                    fontweight="bold" if abs(val) > 0.7 else "normal")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Coefficient de Pearson")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "correlation_matrix.png")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"   💾 {path}")


# --------------------- Figure 2.6 ------------------------------
def fig_regimes(df, regimes):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        f"Régimes de change du Dinar Tunisien ({period_str(df)})\n"
        "Source : BCT — Annotation : auteur",
        fontsize=13, fontweight="bold", y=1.01
    )

    for ax, col, color, label in [
        (axes[0], "TND_USD", C_USD, "TND/USD"),
        (axes[1], "TND_EUR", C_EUR, "TND/EUR"),
    ]:
        for start, end, rc, rl in regimes:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       alpha=0.2, color=rc)
            mid = pd.Timestamp(start) + (
                pd.Timestamp(end) - pd.Timestamp(start)) / 2
            ax.text(mid, df[col].max() * 0.97, rl,
                    ha="center", va="top", fontsize=8, color="black",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor=rc, alpha=0.5))

        add_train_test_line(ax, label_y_frac=0.85)

        ax.plot(df.index, df[col], color=color,
                linewidth=1.3, zorder=3, label=label)
        ax.set_ylabel(f"TND pour 1 {label.split('/')[1]}")
        ax.set_title(label, fontweight="bold", pad=6)
        ax.legend(loc="upper left", fontsize=9)

        depr = (df[col].iloc[-1] - df[col].iloc[0]) / df[col].iloc[0] * 100
        ax.text(0.98, 0.05,
                f"Dépréciation totale : +{depr:.0f}%",
                transform=ax.transAxes, ha="right",
                fontsize=9, color="darkred",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    axes[1].set_xlabel("Date")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "regimes_change.png")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"   💾 {path}")


# --------------------- Figure 2.3 ------------------------------
def fig_distribution_logret(df):
    from scipy import stats as scipy_stats

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Distribution des log-rendements journaliers ({period_str(df)})\n"
        "Comparaison avec la loi normale théorique",
        fontsize=13, fontweight="bold"
    )

    for ax, col, color, label in [
        (axes[0], "LogRet_TND_USD", C_USD, "TND/USD"),
        (axes[1], "LogRet_TND_EUR", C_EUR, "TND/EUR"),
    ]:
        series = df[col].dropna()
        mu, sigma = series.mean(), series.std()
        kurt = scipy_stats.kurtosis(series)
        skew = scipy_stats.skew(series)

        ax.hist(series, bins=100, density=True,
                color=color, alpha=0.55, label="Distribution empirique")

        x = np.linspace(series.min(), series.max(), 300)
        y = scipy_stats.norm.pdf(x, mu, sigma)
        ax.plot(x, y, color="black", linewidth=2,
                linestyle="--", label="Normale théorique")

        for mult, ls, lbl in [(2, ":", r"$\pm 2\sigma$"),
                               (3, "-.", r"$\pm 3\sigma$")]:
            for sign in [-1, 1]:
                ax.axvline(sign * mult * sigma, color="gray",
                           linestyle=ls, linewidth=0.8,
                           label=lbl if sign == 1 else "")

        ax.set_title(
            f"LogRet {label}\n"
            f"μ={mu:.6f}  σ={sigma:.6f}\n"
            f"Kurtosis={kurt:.2f}  Skewness={skew:.2f}",
            fontweight="bold", pad=8
        )
        ax.set_xlabel("Log-rendement journalier")
        ax.set_ylabel("Densité de probabilité")

        handles = [
            mpatches.Patch(color=color, alpha=0.6, label="Empirique"),
            plt.Line2D([0], [0], color="black",
                       linestyle="--", label="Normale théorique"),
        ]
        ax.legend(handles=handles, fontsize=9)

        textstr = (f"Obs : {len(series):,}\n"
                   f"Min : {series.min():.4f}\n"
                   f"Max : {series.max():.4f}")
        ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "distribution_logret.png")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"   💾 {path}")


# =============================================================
#  ============  CHAPITRE 3 — FIGURES MODÈLES  =================
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
    path = os.path.join(PROCESSED_DIR, "garch_params.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        out = {}
        for _, row in df.iterrows():
            cur = row["Devise"].split("/")[-1]
            out[cur] = {"alpha": row["alpha"], "beta": row["beta"],
                       "alpha_beta": row["alpha_plus_beta"]}
        return out
    print("   ⚠️  garch_params.csv introuvable — affichage sans α/β précis.")
    return {"USD": None, "EUR": None}


def _load_best_model_per_horizon(rmse_data: dict) -> dict:
    best = {}
    for devise, horizons in rmse_data.items():
        best[devise] = {}
        for h, vals in horizons.items():
            valid = [(i, v) for i, v in enumerate(vals) if v is not None]
            if valid:
                best_i = min(valid, key=lambda x: x[1])[0]
                best[devise][h] = (best_i, COLORS_M[best_i])
    return best


# --------------------- forecast_comparison_J1 -------------------
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


# --------------------- rmse_comparison_all -----------------------
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
                              fontsize=7, ha="center", color=best_c, fontweight="bold",
                              fontname="DejaVu Sans")

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


# --------------------- garch_volatility_both ----------------------
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


# --------------------- dm_heatmap_presentation ---------------------
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

    patches = [mpatches.Patch(color="#1a9850", label="p < 0.05 : Significant"),
              mpatches.Patch(color="#fee08b", label="p < 0.10 : Marginal"),
              mpatches.Patch(color="#d73027", label="p > 0.10 : Not significant")]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
              bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "dm_heatmap_presentation.png")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"   💾 {path}")


# --------------------- horizon_summary ------------------------------
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
                 fontweight="bold", color=GOLD, transform=fig.transFigure,
                 fontname="DejaVu Sans")
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


# =============================================================
#  EXÉCUTION
# =============================================================
def main():
    print("=" * 60)
    print("  GÉNÉRATION DE TOUTES LES FIGURES DU RAPPORT")
    print("  Chapitre 2 (Données) + Chapitre 3 (Modèles)")
    print("=" * 60)

    df = pd.read_csv(
        os.path.join(PROCESSED_DIR, "dataset_final.csv"),
        parse_dates=["Date"], index_col="Date"
    )
    print(f"✅ Dataset chargé : {df.shape}")
    print(f"   Période : {df.index.min().date()} → {df.index.max().date()}")

    if _CONFIG_OK:
        print(f"   Fenêtre train/test (depuis s2_arima_garch.config) : "
              f"train {TRAIN_START}→{TRAIN_END} | test {TEST_START}→{TEST_END}")
    else:
        print(f"   ⚠️  Import de s2_arima_garch.config impossible — "
              f"valeurs de secours utilisées ({TEST_START})")

    regimes = make_regimes(df)

    print("\n--- Chapitre 2 : Données ---")
    chapitre2_figs = [
        ("[1/7] Cours de change historiques...",       lambda: fig_cours_historiques(df, regimes)),
        ("[2/7] Log-rendements journaliers...",        lambda: fig_log_rendements(df)),
        ("[3/7] Distribution des log-rendements...",   lambda: fig_distribution_logret(df)),
        ("[4/7] Variables macroéconomiques...",        lambda: fig_macro_variables(df)),
        ("[5/7] Nouvelles variables macro...",         lambda: fig_new_macro_variables(df)),
        ("[6/7] Matrice de corrélation...",             lambda: fig_correlation(df)),
        ("[7/7] Régimes de change annotés...",          lambda: fig_regimes(df, regimes)),
    ]
    for msg, fn in chapitre2_figs:
        print(msg)
        try:
            fn()
        except Exception as e:
            print(f"   ❌ Échec : {e}")

    print("\n--- Chapitre 3 : Modèles ---")
    chapitre3_figs = [
        ("[1/5] Comparaison prévisions J+1...", fig_forecast_comparison),
        ("[2/5] RMSE bar chart comparatif...",  fig_rmse_barchart),
        ("[3/5] GARCH volatilité + VaR...",     fig_garch_var),
        ("[4/5] Diebold-Mariano heatmap...",    fig_dm_heatmap),
        ("[5/5] Résumé horizons...",            fig_horizon_summary),
    ]
    for msg, fn in chapitre3_figs:
        print(msg)
        try:
            fn()
        except Exception as e:
            print(f"   ❌ Échec : {e} — fichier(s) source manquant(s), "
                  f"cette figure sera ignorée")

    print("\n--- Chapitre 4 : Risque ---")
    chapitre4_figs = [
        ("[1/4] VaR par profil (4 méthodes)...", fig_var_comparison_ch4),
        ("[2/4] Heatmap validation Kupiec...",    fig_kupiec_heatmap),
        ("[3/4] Diversification Desk BIAT...",    fig_diversification_desk_biat),
        ("[4/4] Stress testing comparatif...",    fig_stress_comparison),
    ]
    for msg, fn in chapitre4_figs:
        print(msg)
        try:
            fn()
        except Exception as e:
            print(f"   ❌ Échec : {e} — fichier(s) source manquant(s) "
                  f"(var_results.csv / backtest_kupiec.csv / stress_results.csv / "
                  f"historical_scenarios.csv), cette figure sera ignorée")

    print("\n--- Chapitre 5 : Couverture ---")
    chapitre5_figs = [
        ("[1/2] Diagramme de payoff Forward vs Option...", fig_option_payoff_diagram),
        ("[2/2] Visuel seuils de rentabilité...",           fig_breakeven_visual),
    ]
    for msg, fn in chapitre5_figs:
        print(msg)
        try:
            fn()
        except Exception as e:
            print(f"   ❌ Échec : {e} — fichier(s) source manquant(s) "
                  f"(forward_results.csv / options_results.csv / "
                  f"hedging_comparison_full.csv), cette figure sera ignorée")

    n_saved = len([f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")])
    print(f"\n✅ Terminé — {n_saved} figure(s) au total dans {FIGURES_DIR}")


# =============================================================
#  ============  CHAPITRE 4 — FIGURES RISQUE  ==================
# =============================================================

def fig_var_comparison_ch4():
    """Bar chart VaR par profil mono-devise, 4 méthodes x 2 seuils."""
    path = os.path.join(PROCESSED_DIR, "var_results.csv")
    df = pd.read_csv(path)
    df = df[df["profile"] != "Desk BIAT"].copy()

    profiles = ["ALPHA SARL", "BETA Export", "M. GAMMA", "Dette BIAT"]
    methods  = ["Historique", "Paramétrique (GARCH)", "Monte Carlo",
               "Monte Carlo Student-t"]
    colors_m = [GRAY, ORANGE, GREEN, "#8B5CF6"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("VaR par méthode — profils mono-devise",
                fontsize=13, fontweight="bold")

    for ax, profile in zip(axes.flatten(), profiles):
        sub = df[df["profile"] == profile]
        cur = sub["currency"].iloc[0]
        x = np.arange(2)  # 95%, 99%
        width = 0.2
        for i, (m, c) in enumerate(zip(methods, colors_m)):
            vals = [sub[(sub["method"]==m)&(sub["confidence"]==a)]["VaR_value"].values
                   for a in [0.95, 0.99]]
            vals = [v[0] if len(v) else 0 for v in vals]
            bars = ax.bar(x + (i-1.5)*width, vals, width, label=m, color=c,
                         alpha=0.9, edgecolor="white", linewidth=0.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x()+b.get_width()/2, v, f"{v:,.0f}",
                       ha="center", va="bottom", fontsize=6.5, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(["95%", "99%"], fontweight="bold")
        ax.set_title(f"{profile} ({cur})", fontweight="bold", fontsize=11)
        ax.set_ylabel(f"VaR ({cur})")
        ax.set_ylim(0, ax.get_ylim()[1]*1.25)

    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
              bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path_out = os.path.join(FIGURES_DIR, "var_comparison_ch4.png")
    plt.savefig(path_out, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"   💾 {path_out}")


def fig_kupiec_heatmap():
    """Heatmap validation Kupiec — méthode x (devise,confiance)."""
    path = os.path.join(PROCESSED_DIR, "backtest_kupiec.csv")
    df = pd.read_csv(path)

    methods = ["Historique", "Paramétrique (GARCH)", "Monte Carlo",
              "Monte Carlo Student-t"]
    cols = [("TND/USD", 0.95), ("TND/USD", 0.99),
           ("TND/EUR", 0.95), ("TND/EUR", 0.99)]
    col_labels = ["USD\n95%", "USD\n99%", "EUR\n95%", "EUR\n99%"]

    valid_matrix = np.zeros((len(methods), len(cols)))
    lr_matrix    = np.zeros((len(methods), len(cols)))
    for i, m in enumerate(methods):
        for j, (cur, conf) in enumerate(cols):
            row = df[(df["method"]==m)&(df["currency"]==cur)&(df["confidence"]==conf)]
            if not row.empty:
                valid_matrix[i,j] = 1 if bool(row["modele_valide"].iloc[0]) else 0
                lr_matrix[i,j] = row["LR_stat"].iloc[0]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(valid_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels, fontweight="bold")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Validation Kupiec — méthode × devise × confiance\n"
                "Vert = validé | Rouge = rejeté", fontweight="bold")

    for i in range(len(methods)):
        for j in range(len(cols)):
            status = "Oui" if valid_matrix[i,j]==1 else "Non"
            color = "black" if valid_matrix[i,j]==1 else "white"
            ax.text(j, i, f"LR={lr_matrix[i,j]:.2f}\n{status}",
                   ha="center", va="center", fontsize=9,
                   color=color, fontweight="bold")

    plt.tight_layout()
    path_out = os.path.join(FIGURES_DIR, "kupiec_heatmap.png")
    plt.savefig(path_out, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"   💾 {path_out}")


def fig_diversification_desk_biat():
    """Bar chart VaR non-diversifiée vs diversifiée — Desk BIAT."""
    path = os.path.join(PROCESSED_DIR, "var_results.csv")
    df = pd.read_csv(path)
    df = df[df["profile"] == "Desk BIAT"].copy()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(2)
    width = 0.3

    undiv = [df[df["confidence"]==a]["VaR_non_diversifie"].values[0] for a in [0.95,0.99]]
    div   = [df[df["confidence"]==a]["VaR_diversifie"].values[0] for a in [0.95,0.99]]
    benef = [df[df["confidence"]==a]["benefice_pct"].values[0] for a in [0.95,0.99]]

    b1 = ax.bar(x - width/2, undiv, width, label="VaR non-diversifiée\n(somme EUR+USD)",
               color=RED, alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + width/2, div, width, label="VaR diversifiée\n(avec corrélation)",
               color=GREEN, alpha=0.85, edgecolor="white")

    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x()+b.get_width()/2, h, f"{h:,.0f}", ha="center",
                   va="bottom", fontsize=9, fontweight="bold")

    for i, (xi, pct) in enumerate(zip(x, benef)):
        ax.annotate(f"-{pct:.1f}%", xy=(xi, max(undiv[i],div[i])*1.08),
                   ha="center", fontsize=11, fontweight="bold", color=NAVY)

    ax.set_xticks(x)
    ax.set_xticklabels(["95%", "99%"], fontweight="bold")
    ax.set_ylabel("VaR (TND)")
    ax.set_title("Bénéfice de diversification — Desk BIAT (50% EUR / 50% USD)",
                fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, max(undiv)*1.25)
    plt.tight_layout()
    path_out = os.path.join(FIGURES_DIR, "diversification_desk_biat.png")
    plt.savefig(path_out, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"   💾 {path_out}")


def fig_stress_comparison():
    """Comparaison chocs synthétiques vs scénarios historiques, en % d'exposition."""
    stress = pd.read_csv(os.path.join(PROCESSED_DIR, "stress_results.csv"))
    hist   = pd.read_csv(os.path.join(PROCESSED_DIR, "historical_scenarios.csv"))

    profiles = ["ALPHA SARL", "BETA Export", "M. GAMMA", "Dette BIAT"]
    fig, ax = plt.subplots(figsize=(13, 6))

    labels = ["Choc 10%", "Choc 20%", "Choc 30%",
             "Dévaluation\n2018", "COVID-19", "Guerre\nUkraine"]
    x = np.arange(len(labels))
    width = 0.2
    colors_p = [NAVY, ORANGE, GREEN, RED]

    for i, profile in enumerate(profiles):
        s = stress[stress["profile"]==profile].sort_values("choc_pct")["perte_pct_exposition"].tolist()
        h_df = hist[hist["profile"]==profile]
        h_vals = []
        for scen in ["Dévaluation 2018", "COVID-19", "Guerre Ukraine"]:
            row = h_df[h_df["scenario"]==scen]
            h_vals.append(row["perte_pct_exposition"].values[0] if not row.empty else 0)
        vals = s + h_vals
        ax.bar(x + (i-1.5)*width, vals, width, label=profile,
              color=colors_p[i], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.axvline(2.5, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Perte (% de l'exposition)")
    ax.set_title("Stress testing — chocs synthétiques vs scénarios historiques réels\n"
                "(comparaison normalisée en % d'exposition, tailles de profils très différentes)",
                fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.text(1, ax.get_ylim()[1]*0.92, "Chocs synthétiques", ha="center",
           fontsize=9, style="italic", color=GRAY)
    ax.text(4, ax.get_ylim()[1]*0.92, "Scénarios historiques réels", ha="center",
           fontsize=9, style="italic", color=GRAY)
    plt.tight_layout()
    path_out = os.path.join(FIGURES_DIR, "stress_comparison.png")
    plt.savefig(path_out, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"   💾 {path_out}")


# =============================================================
#  ============  CHAPITRE 5 — FIGURES COUVERTURE  ==============
# =============================================================

def fig_option_payoff_diagram():
    """Diagramme de payoff — Sans couverture vs Forward vs Option (CALL, importateur)."""
    fwd_path = os.path.join(PROCESSED_DIR, "forward_results.csv")
    opt_path = os.path.join(PROCESSED_DIR, "options_results.csv")

    # Valeurs par défaut (ALPHA SARL J+30) si fichiers indisponibles
    K, F, premium = 3.37756, 3.39369, 0.019563
    if os.path.exists(fwd_path) and os.path.exists(opt_path):
        fwd = pd.read_csv(fwd_path)
        opt = pd.read_csv(opt_path)
        row_f = fwd[(fwd["profile"]=="ALPHA SARL") & (fwd["horizon_j"]==30)]
        row_o = opt[(opt["profile"]=="ALPHA SARL") & (opt["horizon_j"]==30)]
        if not row_f.empty and not row_o.empty:
            K = row_o["strike_atm"].values[0]
            F = row_f["forward"].values[0]
            premium = row_o["prime_unitaire"].values[0]

    S_T = np.linspace(K*0.90, K*1.10, 300)
    unhedged = S_T
    forward_hedged = np.full_like(S_T, F)
    option_hedged = np.minimum(S_T, K) + premium

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(S_T, unhedged, color=GRAY, linewidth=2, linestyle=":",
           label="Non couvert (achat au comptant)")
    ax.plot(S_T, forward_hedged, color=NAVY, linewidth=2.2,
           label="Couvert par Forward")
    ax.plot(S_T, option_hedged, color=GOLD, linewidth=2.2,
           label="Couvert par Option CALL")

    ax.axvline(K, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(K, ax.get_ylim()[0] if False else min(unhedged)*0.999,
           f"K = {K:.4f}", rotation=90, fontsize=8, va="bottom", ha="right")

    ax.set_xlabel(r"Cours final à maturité $S_T$ (TND par devise)")
    ax.set_ylabel("Coût total pour acquérir 1 unité de devise (TND)")
    ax.set_title("Profils de payoff — importateur (ALPHA SARL, J+30)\n"
                "Protection symétrique (Forward) vs asymétrique (Option)",
                fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.annotate("Le Forward élimine\ntout le potentiel de gain",
               xy=(K*0.93, F), xytext=(K*0.93, F+0.012),
               fontsize=8, ha="center", color=NAVY,
               arrowprops=dict(arrowstyle="->", color=NAVY, lw=1))
    ax.annotate("L'Option plafonne la perte\net garde le potentiel de gain\n(moins la prime)",
               xy=(K*1.06, option_hedged[-1]), xytext=(K*1.045, option_hedged[-1]+0.015),
               fontsize=8, ha="center", color=GOLD,
               arrowprops=dict(arrowstyle="->", color=GOLD, lw=1))
    plt.tight_layout()
    path_out = os.path.join(FIGURES_DIR, "option_payoff_diagram.png")
    plt.savefig(path_out, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"   💾 {path_out}")


def fig_breakeven_visual():
    """Dot plot — spot / forward / seuil / prévision, normalisé en % du spot."""
    path = os.path.join(PROCESSED_DIR, "hedging_comparison_full.csv")
    df = pd.read_csv(path)
    df["label"] = df["profile"] + " J+" + df["horizon_j"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(df))

    for i, row in enumerate(df.itertuples()):
        spot = row.spot
        fwd_pct = (row.forward - spot) / spot * 100
        seuil_pct = (row.seuil_rentabilite_option - spot) / spot * 100
        fcst_pct = (row.cours_predit - spot) / spot * 100

        ax.plot([0, fwd_pct], [i, i], color=GRAY, linewidth=1, alpha=0.5, zorder=1)
        ax.scatter(0, i, color="black", marker="|", s=100, zorder=3)
        ax.scatter(fwd_pct, i, color=NAVY, marker="s", s=70, zorder=3,
                  label="Forward" if i==0 else "")
        ax.scatter(seuil_pct, i, color=GOLD, marker="*", s=160, zorder=4,
                  label="Seuil de rentabilité" if i==0 else "")
        rec_color = GREEN if row.recommandation_modele == "Option" else RED
        ax.scatter(fcst_pct, i, color=rec_color, marker="o", s=70, zorder=4,
                  edgecolor="black", linewidth=0.6,
                  label=None)

    ax.axvline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("Écart au cours spot actuel (%)")
    ax.set_title("Seuil de rentabilité vs prévision du modèle, par profil et horizon\n"
                "Point coloré = prévision (vert → Option recommandée, rouge → Forward recommandé)",
                fontweight="bold", fontsize=11)

    handles = [
        plt.Line2D([0],[0], marker="s", color="w", markerfacecolor=NAVY, markersize=9, label="Forward"),
        plt.Line2D([0],[0], marker="*", color="w", markerfacecolor=GOLD, markersize=13, label="Seuil de rentabilité"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=GREEN, markeredgecolor="black", markersize=9, label="Prévision (Option gagne)"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=RED, markeredgecolor="black", markersize=9, label="Prévision (Forward gagne)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8.5)
    ax.invert_yaxis()
    plt.tight_layout()
    path_out = os.path.join(FIGURES_DIR, "breakeven_visual.png")
    plt.savefig(path_out, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"   💾 {path_out}")


if __name__ == "__main__":
    main()