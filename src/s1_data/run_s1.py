# =============================================================
#  src/s1_data/run_s1.py
#  S1 — Construction du dataset final
#  Auteur : Youssef Neji | MINDS ENIT | 2025-2026
#
#  FICHIER UNIQUE — remplace et fusionne :
#    - update_data.py        (parsing BCT cours de change + BCT
#                              complémentaire + fetch FRED live)
#    - data.py                (nettoyage + fusion + dataset_final.csv)
#    - regenerate_bct_extra.py (doublon de update_bct_extra_data)
#    - merge_bct_features.py   (doublon de load_bct_extra)
#
#  Entrée  : data/raw/*.xls (BCT, HTML déguisé) + data/raw/*.csv
#            (FRED) + data/raw/*.xlsx (BCT/INS annuel)
#  Sortie  : data/processed/dataset_final.csv
#
#  Usage :
#    python -m src.s1_data.run_s1                # build depuis data/raw
#    python -m src.s1_data.run_s1 --fetch-fred    # + télécharge FRED avant
#
#  Note régime de change :
#    - 2004-2011 : TND quasi-fixe
#    - 2012-2015 : transition
#    - 2016-2026 : dépréciation progressive accélérée
#    → TRAIN_START/TRAIN_END/TEST_START/TEST_END restent définis
#      dans s2_arima_garch/config.py et s3_lstm_xgboost/config.py
# =============================================================

import os
import sys
import numpy as np
import pandas as pd
from datetime import date
from bs4 import BeautifulSoup

# --------------------------------------------------------------
#  0. CONFIGURATION
# --------------------------------------------------------------
BASE_DIR      = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

START_DATE = "2004-01-01"
# Date de coupure figée volontairement pour rester cohérente avec les
# figures/résultats déjà produits dans le rapport (Chapitres 1-3).
# À avancer manuellement + régénérer le rapport si besoin.
END_DATE   = "2026-08-11"

PUBLICATION_LAG_DAYS = 20
# Délai de publication supposé (jours calendaires) entre la date de
# fin de mois d'une observation BCT complémentaire et sa disponibilité
# publique réelle. Appliqué AVANT le forward-fill journalier pour
# éviter toute fuite d'information (look-ahead bias) dans S2/S3.

BCT_RATES_FILE = "BCT_statistque.xls"

# Nom exact du fichier source (tel que téléchargé depuis bct.gov.tn)
# → (nom du checkpoint CSV, colonnes attendues dans le tbody, en ordre)
BCT_EXTRA_FILES = {
    "SITUATION MENSUELLE DE LA BCT "
    "(Encours de fin de période en mille dinars).xls": (
        "BCT_reserves.csv", ["reserves_officielles_mdt"]
    ),
    "EVOLUTION DES TRANSACTIONS SUR LE MARCHE DES CHANGES AU "
    "COMPTANT (DEVISE CONTRE DINAR) (EN MDT).xls": (
        "BCT_interventions.csv", ["interventions_bct_mdt"]
    ),
    "TRANSACTIONS DE CHANGE AU COMPTANT "
    "(DEVISE CONTRE DINAR) (EN MDT).xls": (
        "BCT_transactions.csv",
        ["transactions_devises_dinar_mdt", "total_transactions_comptant_mdt"]
    ),
}

FRED_FILES = {
    "Brent_USD":    "DCOILBRENTEU.csv",
    "TauxFed_USD":  "DFF.csv",
    "TauxBCE_EUR":  "ECBDFR.csv",
    "IPC_USA":      "CPIAUCSL.csv",
    "VIX":          "VIXCLS.csv",
    "EURUSD":       "DEXUSEU.csv",
    "DXY":          "DTWEXBGS.csv",
    "US10Y":        "DGS10.csv",
    "Gold_USD":     "GOLDAMGBD228NLBM.csv",
    "EUR10Y":       "IRLTLT01DEM156N.csv",
    "EXP7230":      "EXP7230.csv",
    "IMP7230":      "IMP7230.csv",
}
# Séries pandas-datareader correspondantes (utilisées seulement si
# --fetch-fred est passé) : DXY/EXP7230/IMP7230 gardent le même nom
# de série FRED que de colonne, les autres diffèrent.
FRED_SERIES_ID = {
    "Brent_USD": "DCOILBRENTEU", "TauxFed_USD": "DFF",
    "TauxBCE_EUR": "ECBDFR", "IPC_USA": "CPIAUCSL",
    "VIX": "VIXCLS", "EURUSD": "DEXUSEU", "DXY": "DTWEXBGS",
    "US10Y": "DGS10", "Gold_USD": "GOLDAMGBD228NLBM",
    "EUR10Y": "IRLTLT01DEM156N", "EXP7230": "EXP7230", "IMP7230": "IMP7230",
}

SOCIO_TAUX_BCT_FILE = "Taux_directeur_BCT.xlsx"
SOCIO_IPC_FILE       = "IPCTUN.xlsx"

OUTPUT_PATH = os.path.join(PROCESSED_DIR, "dataset_final.csv")


# --------------------------------------------------------------
#  1. PARSING BRUT — fichiers BCT (HTML déguisé en .xls)
#
#  Format commun à tous les exports bct.gov.tn (cours de change ET
#  fichiers complémentaires) :
#    - <thead> : ligne d'en-tête + une ligne <tr><td> par date
#      (format JJ/MM/AAAA)
#    - <tbody class="bct-mod-content-1"> : une ligne par date,
#      len(col_names) cellules <td> par ligne, alignées avec <thead>
#    - Virgule décimale française, espace/nbsp en séparateur de
#      milliers, "-"/vide = valeur manquante
#    - Une ligne finale vide (0 <td>) est systématiquement présente
#      et doit être ignorée
# --------------------------------------------------------------

def _to_float_bct(v: str):
    """Convertit une valeur BCT (virgule française, espaces =
    milliers) en float. '-'/vide/N/A → NaN."""
    v = v.replace("\xa0", " ").strip()
    if v in ("", "-", "--", "N/A"):
        return np.nan
    v = v.replace(" ", "").replace(",", ".")
    try:
        return float(v)
    except ValueError:
        return np.nan


def _parse_bct_html_table(filepath: str, col_names: list) -> pd.DataFrame:
    """Parse un fichier BCT .xls (HTML déguisé) au format commun
    décrit ci-dessus. Fonctionne aussi bien pour le fichier des
    cours de change (2 colonnes) que pour les fichiers complémentaires
    (1 ou 2 colonnes)."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    soup = BeautifulSoup(content, "lxml")

    theads = soup.find_all("thead")
    if not theads:
        raise ValueError(f"Aucun <thead> trouvé dans {filepath}")
    date_rows = theads[0].find_all("tr")
    dates = []
    for r in date_rows[1:]:  # ligne 0 = en-tête
        td = r.find("td")
        if td:
            txt = td.get_text(strip=True)
            if txt and txt != "Date/Indicateurs":
                dates.append(txt)

    tbody = soup.find("tbody", class_="bct-mod-content-1")
    if tbody is None:
        raise ValueError(
            f"Aucun <tbody class='bct-mod-content-1'> trouvé dans {filepath}"
        )
    val_rows = tbody.find_all("tr")

    n_cols = len(col_names)
    values = {c: [] for c in col_names}
    for r in val_rows:
        tds = r.find_all("td")
        if len(tds) == n_cols:
            for c, td in zip(col_names, tds):
                values[c].append(td.get_text(strip=True))

    n_vals = len(next(iter(values.values()))) if values[col_names[0]] else 0
    if len(dates) != n_vals:
        raise ValueError(
            f"Désalignement dans {os.path.basename(filepath)} : "
            f"{len(dates)} dates vs {n_vals} lignes de valeurs "
            f"({n_cols} colonne(s) attendue(s))"
        )

    data = {"Date": pd.to_datetime(dates, format="%d/%m/%Y")}
    for c in col_names:
        data[c] = [_to_float_bct(v) for v in values[c]]

    df = pd.DataFrame(data).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def load_bct_rates() -> pd.DataFrame:
    """Parse BCT_statistque.xls → DataFrame[TND_USD, TND_EUR].
    Sauvegarde un checkpoint dans data/raw/BCT_taux_change.csv."""
    print("📥 Parsing cours de change BCT (BCT_statistque.xls)...")
    path = os.path.join(RAW_DIR, BCT_RATES_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier manquant : {path}\n"
            f"→ Télécharger depuis https://www.bct.gov.tn/bct/siteprod/cours_tn.jsp"
        )

    df = _parse_bct_html_table(path, ["TND_USD", "TND_EUR"])

    n_nan = df["TND_USD"].isna().sum()
    if n_nan > 0:
        print(f"   ⚠️  {n_nan} valeurs manquantes ('-') — laissées en NaN, "
              f"gérées ensuite par clean_rate_series()")

    checkpoint = os.path.join(RAW_DIR, "BCT_taux_change.csv")
    df.to_csv(checkpoint)
    print(f"   ✅ {len(df)} obs | {df.index[0].date()} → {df.index[-1].date()}")
    print(f"   TND/USD : {df['TND_USD'].iloc[-1]:.4f} | "
          f"TND/EUR : {df['TND_EUR'].iloc[-1]:.4f}  "
          f"(checkpoint : {os.path.basename(checkpoint)})")
    return df


def _warn_unrecognized_xls():
    """Signale les .xls présents dans data/raw/ mais non reconnus —
    aide à repérer immédiatement un problème de nom de fichier
    (ex : suffixe '(1)' d'un second téléchargement)."""
    expected = {BCT_RATES_FILE.lower()} | {f.lower() for f in BCT_EXTRA_FILES}
    if not os.path.isdir(RAW_DIR):
        return
    present = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".xls")]
    unexpected = [f for f in present if f.lower() not in expected]
    if unexpected:
        print(f"   ℹ️  .xls présents mais non reconnus (ignorés) : {unexpected}")


def load_bct_extra_raw() -> pd.DataFrame:
    """
    Parse les 3 fichiers BCT complémentaires (réserves, interventions,
    transactions), calcule les features dérivées stationnaires, puis
    applique le décalage de publication anti look-ahead bias.

      - reserves_mom_pct     : variation mensuelle (%) des réserves
      - bct_market_share_pct : part de la BCT dans les transactions
        au comptant totales (proxy d'intensité d'intervention)

    Chaque fichier est optionnel : un fichier manquant est ignoré
    avec un avertissement (ne bloque pas le pipeline). Les 2 features
    dérivées ne sont ajoutées que si leurs colonnes sources sont
    toutes disponibles.
    """
    print("\n📥 Parsing BCT complémentaire "
          "(réserves / interventions / transactions)...")

    frames = []
    for fname, (checkpoint_name, col_names) in BCT_EXTRA_FILES.items():
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            print(f"   ⚠️  {fname} introuvable — ignoré "
                  f"(conserve le checkpoint existant si présent)")
            continue
        df = _parse_bct_html_table(path, col_names)
        frames.append(df)

        checkpoint = os.path.join(RAW_DIR, checkpoint_name)
        df.to_csv(checkpoint)
        print(f"   ✅ {checkpoint_name} : {len(df)} obs | "
              f"{df.index[0].date()} → {df.index[-1].date()}")

    if not frames:
        print("   ⚠️  Aucune donnée BCT complémentaire disponible")
        return pd.DataFrame()

    bct_extra = pd.concat(frames, axis=1, join="outer").sort_index()

    # --- Features dérivées (calculées AVANT le décalage) ---
    if "reserves_officielles_mdt" in bct_extra.columns:
        bct_extra["reserves_mom_pct"] = (
            bct_extra["reserves_officielles_mdt"].pct_change().round(4)
        )

    if {"interventions_bct_mdt", "total_transactions_comptant_mdt"} <= set(bct_extra.columns):
        bct_extra["bct_market_share_pct"] = (
            bct_extra["interventions_bct_mdt"]
            / bct_extra["total_transactions_comptant_mdt"]
        ).round(4)

    # --- Décalage de publication (anti look-ahead bias) ---
    bct_extra.index = bct_extra.index + pd.Timedelta(days=PUBLICATION_LAG_DAYS)
    bct_extra.index.name = "Date"

    print(f"   ✅ BCT complémentaire fusionné : {list(bct_extra.columns)} "
          f"(décalage publication : +{PUBLICATION_LAG_DAYS}j)")
    return bct_extra


# --------------------------------------------------------------
#  2. VARIABLES MACRO — FRED (12 séries) + BCT/INS annuel (xlsx)
# --------------------------------------------------------------

def load_fred_csv(filepath: str, col_name: str) -> pd.Series:
    """Charge un CSV FRED (2 colonnes : date, valeur — quel que
    soit le nom d'en-tête d'origine). Valeurs non numériques → NaN."""
    df = pd.read_csv(filepath)
    df.columns = ["Date", col_name]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    return df[col_name]


def load_macro_fred() -> pd.DataFrame:
    """Charge et consolide les 12 séries FRED depuis data/raw/."""
    print("📥 Chargement variables macro FRED...")
    series = {}
    for col_name, filename in FRED_FILES.items():
        path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(path):
            print(f"   ⚠️  Fichier manquant : {filename} — ignoré")
            continue
        s = load_fred_csv(path, col_name).loc[START_DATE:END_DATE]
        print(f"   ✅ {col_name:12} : {len(s)} obs | {s.isna().sum()} NaN bruts")
        series[col_name] = s
    macro = pd.DataFrame(series)
    macro.index.name = "Date"
    return macro


def _parse_annual_xlsx(filepath: str, year_lo: int = 2004,
                       year_hi: int = 2025) -> dict:
    """Lit un xlsx BCT/INS au format : ligne 0 = années, ligne 1 =
    valeurs. Retourne {année: valeur} pour les années dans
    [year_lo, year_hi]."""
    xl = pd.ExcelFile(filepath)
    df = xl.parse(xl.sheet_names[0])
    data = {}
    for col in df.columns:
        try:
            yr  = int(float(df.iloc[0][col]))
            val = float(df.iloc[1][col])
            if year_lo <= yr <= year_hi and not pd.isna(val):
                data[yr] = val
        except (ValueError, TypeError):
            pass
    return data


def load_taux_mm_bct() -> pd.Series:
    """Taux marché monétaire BCT (Taux_directeur_BCT.xlsx), annuel
    2004-2024 → mensuel par forward-fill."""
    print("📥 Chargement taux directeur BCT...")
    path = os.path.join(RAW_DIR, SOCIO_TAUX_BCT_FILE)
    if not os.path.exists(path):
        print(f"   ⚠️  {SOCIO_TAUX_BCT_FILE} introuvable — TauxMMBCT sera vide")
        return pd.Series(dtype=float, name="TauxMMBCT")

    data = _parse_annual_xlsx(path, 2004, 2025)
    if not data:
        print("   ⚠️  Aucune donnée extraite — colonne TauxMMBCT sera vide")
        return pd.Series(dtype=float, name="TauxMMBCT")

    idx = pd.to_datetime([f"{yr}-01-01" for yr in sorted(data)])
    s   = pd.Series([data[yr] for yr in sorted(data)], index=idx, name="TauxMMBCT")
    s_monthly = s.resample("ME").ffill()

    print(f"   ✅ {len(data)} années | {min(data)}–{max(data)}")
    print(f"   Valeurs : min={min(data.values()):.2f}% | max={max(data.values()):.2f}%")
    return s_monthly


def load_ipc_tunisie() -> pd.Series:
    """IPC Tunisie (IPCTUN.xlsx), annuel 2015-2025 → mensuel par
    interpolation linéaire. NaN maintenus volontairement 2004-2014."""
    print("📥 Chargement IPC Tunisie...")
    path = os.path.join(RAW_DIR, SOCIO_IPC_FILE)
    if not os.path.exists(path):
        print(f"   ⚠️  {SOCIO_IPC_FILE} introuvable — IPC_Tunisie sera vide")
        return pd.Series(dtype=float, name="IPC_Tunisie")

    data = _parse_annual_xlsx(path, 2004, 2025)
    if not data:
        print("   ⚠️  Aucune donnée extraite")
        return pd.Series(dtype=float, name="IPC_Tunisie")

    idx_annual = pd.to_datetime([f"{yr}-01-01" for yr in sorted(data)])
    s_annual   = pd.Series([data[yr] for yr in sorted(data)],
                          index=idx_annual, name="IPC_Tunisie")

    idx_monthly = pd.date_range(start=f"{min(data)}-01-01",
                                end=f"{max(data)}-12-31", freq="MS")
    s_monthly = s_annual.reindex(idx_monthly).interpolate(method="linear")

    print(f"   ✅ {len(data)} années | {min(data)}–{max(data)}")
    return s_monthly


# --------------------------------------------------------------
#  3. NETTOYAGE DES COURS DE CHANGE
# --------------------------------------------------------------

def clean_rate_series(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Nettoie une série de cours de change en 4 étapes :
      1. Réindexage sur jours ouvrés (comble les jours fériés tunisiens)
      2. Interpolation linéaire des NaN (+ ffill/bfill aux extrémités)
      3. Suppression des outliers (z-score glissant 252j, centré,
         seuil |z|>4 — pas un z-score global, pour ne pas confondre
         les 3 régimes de change)
      4. Log-rendements : r_t = ln(S_t / S_{t-1})
    """
    print(f"\n   🔧 Nettoyage {col}...")

    bdays = pd.date_range(start=df.index.min(), end=df.index.max(), freq="B")
    df    = df.reindex(bdays)
    df.index.name = "Date"
    n_missing = df[col].isna().sum()
    print(f"      Réindexage    : {len(df)} obs | {n_missing} NaN (jours fériés comblés)")

    df[col] = df[col].interpolate(method="linear").ffill().bfill()
    print(f"      Interpolation : {n_missing} NaN → 0")

    rolling_mean = df[col].rolling(window=252, center=True, min_periods=60).mean()
    rolling_std  = df[col].rolling(window=252, center=True, min_periods=60).std()
    z     = np.abs((df[col] - rolling_mean) / rolling_std)
    n_out = (z > 4).sum()
    df.loc[z > 4, col] = np.nan
    df[col] = df[col].interpolate(method="linear").ffill().bfill()
    print(f"      Outliers (|z|>4, fenêtre glissante 252j) supprimés : {n_out}")

    df[f"LogRet_{col}"] = np.log(df[col] / df[col].shift(1))
    print(f"      ✅ {col} prêt : {len(df)} obs")

    return df


# --------------------------------------------------------------
#  4. FUSION FINALE
# --------------------------------------------------------------

def build_dataset(
    rates:       pd.DataFrame,
    macro_fred:  pd.DataFrame,
    taux_bct:    pd.Series,
    ipc_tunisie: pd.Series,
    bct_extra:   pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Fusionne toutes les sources en un dataset journalier (jours ouvrés).

    Règles d'alignement :
      - Base     : cours de change nettoyés (jours ouvrés)
      - Journalier FRED → join direct + ffill + bfill
      - Mensuel (IPC_USA, TauxMMBCT, EUR10Y, EXP7230, IMP7230)
        → resample B + ffill + bfill
      - IPC Tunisie → resample B + ffill + bfill
        (NaN maintenus sur 2004-2014 par manque de données source)
      - BCT complémentaire → resample B + ffill SEULEMENT (pas de
        bfill : le décalage de publication déjà appliqué dans
        load_bct_extra_raw() rend les NaN avant la 1ère publication
        légitimes, pas des trous à combler)
    """
    print("\n🔗 Construction du dataset final...")

    dataset = rates.copy()

    if not macro_fred.empty:
        macro_daily = macro_fred.resample("B").ffill()
        dataset = dataset.join(macro_daily, how="left")
        for col in macro_fred.columns:
            dataset[col] = dataset[col].ffill().bfill()
        print(f"   ✅ Macro FRED : {list(macro_fred.columns)}")

    if taux_bct is not None and len(taux_bct) > 0:
        taux_daily = taux_bct.resample("B").ffill()
        dataset    = dataset.join(taux_daily.rename("TauxMMBCT"), how="left")
        dataset["TauxMMBCT"] = dataset["TauxMMBCT"].ffill().bfill()
        print(f"   ✅ TauxMMBCT : propagé sur toute la période")

    if ipc_tunisie is not None and len(ipc_tunisie) > 0:
        idx_bdays = pd.date_range(start=ipc_tunisie.index[0],
                                  end=ipc_tunisie.index[-1], freq="B")
        ipc_daily = ipc_tunisie.reindex(idx_bdays).interpolate(method="linear")
        ipc_daily.name = "IPC_Tunisie"
        dataset   = dataset.join(ipc_daily, how="left")
        dataset["IPC_Tunisie"] = dataset["IPC_Tunisie"].ffill()
        n_nan = dataset["IPC_Tunisie"].isna().sum()
        print(f"   ✅ IPC_Tunisie : disponible 2015–2025 | {n_nan} NaN maintenus (2004–2014)")

    if bct_extra is not None and not bct_extra.empty:
        bct_daily = bct_extra.resample("B").ffill()
        dataset   = dataset.join(bct_daily, how="left")
        for col in bct_extra.columns:
            dataset[col] = dataset[col].ffill()
        n_nan_head = dataset[list(bct_extra.columns)].isna().sum().max()
        print(f"   ✅ BCT complémentaire : {list(bct_extra.columns)} "
              f"| {n_nan_head} NaN max maintenus (avant 1ère publication)")

    dataset = dataset.loc[START_DATE:END_DATE]
    dataset = dataset.dropna(subset=["TND_USD", "TND_EUR"], how="all")

    print(f"\n   📊 Dataset final : {dataset.shape[0]} lignes × {dataset.shape[1]} colonnes")
    print(f"   Période         : {dataset.index[0].date()} → {dataset.index[-1].date()}")
    print(f"   Colonnes        : {list(dataset.columns)}")
    print(f"\n   Qualité des données (NaN par colonne) :")

    nan_counts = dataset.isna().sum()
    total      = len(dataset)
    for col, n in nan_counts.items():
        pct    = n / total * 100
        status = "✅" if n == 0 else ("⚠️ " if pct < 1 else ("🔶" if pct < 30 else "❌"))
        print(f"      {status} {col:<28} : {n:>4} NaN ({pct:.1f}%)")

    return dataset


# --------------------------------------------------------------
#  5. OPTIONNEL — TÉLÉCHARGEMENT FRED EN LIGNE (--fetch-fred)
# --------------------------------------------------------------

def fetch_fred_live():
    """Télécharge les 12 séries FRED via pandas-datareader et les
    écrit dans data/raw/ (écrase les fichiers existants). Nécessite
    une connexion internet + pandas-datareader installé. Optionnel :
    le pipeline fonctionne sans, à partir des CSV déjà présents."""
    print("\n📥 Téléchargement FRED (--fetch-fred)...")
    try:
        import pandas_datareader.data as web
    except ImportError:
        print("   ❌ pandas-datareader non installé — "
              "pip install pandas-datareader")
        return

    today = date.today().strftime("%Y-%m-%d")
    for col_name, series_id in FRED_SERIES_ID.items():
        try:
            s = web.DataReader(series_id, "fred", start=START_DATE, end=today)
            s.columns = [series_id]
            s.index.name = "DATE"
            out_path = os.path.join(RAW_DIR, FRED_FILES[col_name])
            s.reset_index().to_csv(out_path, index=False)
            print(f"   ✅ {col_name:12} : {len(s)} obs | "
                  f"dernière : {s.index[-1].strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"   ⚠️  {col_name:12} : erreur — {e}")


# --------------------------------------------------------------
#  6. PIPELINE PRINCIPAL
# --------------------------------------------------------------

def run_s1(fetch_fred: bool = False) -> pd.DataFrame:
    """
    Pipeline complet S1 : data/raw/* → data/processed/dataset_final.csv

    Étapes :
      1. (optionnel) Téléchargement FRED en ligne
      2. Cours de change BCT (parsing .xls → nettoyage)
      3. Variables macro FRED (12 séries)
      4. Variables macro BCT/INS annuelles (taux marché monétaire, IPC)
      5. BCT complémentaire (réserves, interventions, transactions —
         features dérivées + décalage publication anti look-ahead)
      6. Fusion sur jours ouvrés → dataset_final.csv
    """
    print("=" * 60)
    print("  S1 — CONSTRUCTION DU DATASET FINAL — BIAT STAGE")
    print(f"  Auteur : Youssef Neji | MINDS ENIT | 2025-2026")
    print(f"  Période : {START_DATE} → {END_DATE}")
    print("=" * 60)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if fetch_fred:
        fetch_fred_live()

    # 1. Cours BCT
    rates_raw = load_bct_rates()

    # 2. Nettoyage
    print("\n--- Nettoyage cours de change ---")
    eur_clean = clean_rate_series(rates_raw[["TND_EUR"]].copy(), "TND_EUR")
    usd_clean = clean_rate_series(rates_raw[["TND_USD"]].copy(), "TND_USD")
    eur_clean.to_csv(os.path.join(PROCESSED_DIR, "tnd_eur_clean.csv"))
    usd_clean.to_csv(os.path.join(PROCESSED_DIR, "tnd_usd_clean.csv"))
    print(f"\n   💾 tnd_eur_clean.csv")
    print(f"   💾 tnd_usd_clean.csv")

    # 3. Macro FRED
    print("\n--- Variables macro FRED ---")
    macro_fred = load_macro_fred()
    macro_fred.to_csv(os.path.join(PROCESSED_DIR, "macro_fred.csv"))
    print(f"   💾 macro_fred.csv")

    # 4. Macro BCT / INS
    print("\n--- Variables macro BCT / INS ---")
    taux_bct    = load_taux_mm_bct()
    ipc_tunisie = load_ipc_tunisie()

    # 5. BCT complémentaire
    print("\n--- Variables BCT complémentaires ---")
    _warn_unrecognized_xls()
    bct_extra = load_bct_extra_raw()

    # 6. Fusion + export
    rates_merged = eur_clean.join(usd_clean, how="outer")
    dataset      = build_dataset(rates_merged, macro_fred, taux_bct,
                                 ipc_tunisie, bct_extra)

    dataset.to_csv(OUTPUT_PATH)
    print(f"\n   💾 dataset_final.csv → {OUTPUT_PATH}")

    print("\n" + "=" * 60)
    print("  ✅ S1 TERMINÉ — dataset prêt pour S2 / S3 / S4")
    print("=" * 60)

    return dataset


# --------------------------------------------------------------
#  EXÉCUTION DIRECTE
# --------------------------------------------------------------

if __name__ == "__main__":
    fetch = "--fetch-fred" in sys.argv
    dataset = run_s1(fetch_fred=fetch)

    print("\n--- Aperçu ---")
    print(dataset.head(5).to_string())
    print("\n--- Statistiques TND ---")
    print(dataset[["TND_USD", "TND_EUR"]].describe().to_string())
    print(f"\nShape : {dataset.shape}")