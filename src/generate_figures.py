# =============================================================
#  report/generate_figures.py
#  Génération des figures du Chapitre 1 — Données
#  Auteur : Youssef Neji | MINDS ENIT | 2025-2026
#  Lancer depuis la racine : python report/generate_figures.py
#
#  Toutes les dates affichées dans les titres sont dérivées
#  dynamiquement de la période réelle du dataset chargé — ne
#  nécessite plus de mise à jour manuelle à chaque extension
#  des données.
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
#  Configuration
# -------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR   = os.path.join(BASE_DIR, "report", "images")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Import de la fenêtre train/test depuis la config du pipeline
# de modélisation — source de vérité unique. Si l'import échoue
# (ex: exécution hors structure src/), on retombe sur des valeurs
# de secours explicitement signalées.
try:
    sys.path.insert(0, BASE_DIR)
    from src.s2_arima_garch.config import TRAIN_START, TRAIN_END, TEST_START, TEST_END
    _CONFIG_OK = True
except ImportError:
    TRAIN_START, TRAIN_END = "2012-01-01", "2023-12-31"
    TEST_START,  TEST_END  = "2024-01-01", "2026-07-14"
    _CONFIG_OK = False

# Style global
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

# Palette
C_USD   = "#1E88E5"
C_EUR   = "#FF6B35"
C_BCT   = "#43A047"
C_FED   = "#1E88E5"
C_BCE   = "#FF6B35"
C_BRENT = "#2E7D32"
C_VIX   = "#C62828"
C_IPC   = "#5E35B1"
C_DXY   = "#00695C"
C_US10Y = "#6D4C41"
C_GOLD  = "#F9A825"
C_EUR10Y = "#4527A0"


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
    directement de s2_arima_garch.config (TRAIN_END/TEST_START)
    plutôt que codée en dur — reste toujours synchronisée avec
    la fenêtre réellement utilisée par les modèles.
    """
    split_date = pd.Timestamp(TEST_START)
    ax.axvline(split_date, color="black", linestyle=":",
               linewidth=1.4, alpha=0.8, zorder=4)
    ymin, ymax = ax.get_ylim()
    ax.text(split_date, ymin + label_y_frac * (ymax - ymin),
            "  Train | Test", fontsize=7, color="black",
            fontweight="bold", va="top")


def period_str(df: pd.DataFrame) -> str:
    """Chaîne 'AAAA–AAAA' dérivée dynamiquement du dataset."""
    return f"{df.index.min().year}–{df.index.max().year}"


# =============================================================
#  FIGURE 1 — Cours de change historiques TND/USD et TND/EUR
# =============================================================
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


# =============================================================
#  FIGURE 2 — Log-rendements journaliers
# =============================================================
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


# =============================================================
#  FIGURE 3 — Variables macroéconomiques (variables d'origine)
# =============================================================
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


# =============================================================
#  FIGURE 3bis — Nouvelles variables macroéconomiques (ARIMAX)
# =============================================================
def fig_new_macro_variables(df):
    """
    Variables ajoutées pour enrichir ARIMAX : Dollar Index (DXY),
    taux Trésor US 10 ans, prix de l'or, taux Bund EUR 10 ans.
    N'affiche que les colonnes réellement présentes dans le
    dataset (robuste si l'une des 4 séries FRED est indisponible).
    """
    new_vars = [
        ("DXY",      "Dollar Index (DXY)",        C_DXY,
         "FRED : DTWEXBGS"),
        ("US10Y",    "Taux Trésor US 10 ans (%)",  C_US10Y,
         "FRED : DGS10"),
        ("Gold_USD", "Prix de l'or (USD/once)",    C_GOLD,
         "FRED : GOLDAMGBD228NLBM"),
        ("EUR10Y",   "Taux Bund EUR 10 ans (%)",   C_EUR10Y,
         "FRED : IRLTLT01DEM156N"),
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
        f"Nouvelles variables macroéconomiques — variables exogènes "
        f"additionnelles ARIMAX ({period_str(df)})\nSource : FRED",
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


# =============================================================
#  FIGURE 4 — Matrice de corrélation
# =============================================================
def fig_correlation(df):
    base_cols = ["TND_USD", "TND_EUR", "Brent_USD", "VIX",
                 "TauxFed_USD", "TauxBCE_EUR", "TauxMMBCT",
                 "EURUSD", "IPC_USA"]
    base_labels = {"TND_USD": "TND/USD", "TND_EUR": "TND/EUR",
                   "Brent_USD": "Brent", "VIX": "VIX",
                   "TauxFed_USD": "Fed", "TauxBCE_EUR": "BCE",
                   "TauxMMBCT": "BCT", "EURUSD": "EUR/USD",
                   "IPC_USA": "IPC USA"}

    # Ajout conditionnel des nouvelles variables si présentes
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


# =============================================================
#  FIGURE 5 — Régimes de change annotés
# =============================================================
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


# =============================================================
#  FIGURE 6 — Distribution des log-rendements
# =============================================================
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
#  EXÉCUTION
# =============================================================
def main():
    print("=" * 55)
    print("  GÉNÉRATION DES FIGURES — CHAPITRE 1 DONNÉES")
    print("=" * 55)

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
              f"valeurs de secours utilisées pour la ligne train/test "
              f"({TEST_START})")

    regimes = make_regimes(df)

    print("\n[1/7] Cours de change historiques...")
    fig_cours_historiques(df, regimes)

    print("[2/7] Log-rendements journaliers...")
    fig_log_rendements(df)

    print("[3/7] Variables macroéconomiques...")
    fig_macro_variables(df)

    print("[4/7] Nouvelles variables macroéconomiques (ARIMAX)...")
    fig_new_macro_variables(df)

    print("[5/7] Matrice de corrélation...")
    fig_correlation(df)

    print("[6/7] Régimes de change annotés...")
    fig_regimes(df, regimes)

    print("[7/7] Distribution des log-rendements...")
    fig_distribution_logret(df)

    print(f"\n✅ 7 figures sauvegardées dans {FIGURES_DIR}")


if __name__ == "__main__":
    main()