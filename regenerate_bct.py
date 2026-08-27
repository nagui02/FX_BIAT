# =============================================================
#  regenerate_bct.py
#  Régénère BCT_taux_change.csv depuis BCT_statistque.xls (HTML)
#  Auteur : Youssef Neji | MINDS ENIT | 2025
#
#  Structure du fichier source (HTML déguisé en .xls) :
#    - Table 1 : liste des dates (une par <tr> dans <thead>)
#    - Table 2 : valeurs Dollar US / Euro alignées ligne par ligne
#                (tbody class="bct-mod-content-1")
#    - Séparateur décimal : virgule (format français)
#    - Valeurs déjà en unités directes (pas de /100000 nécessaire)
#    - Certaines lignes contiennent "-" (valeur manquante) → NaN
#
#  Usage : python regenerate_bct.py <input.xls> <output.csv>
# =============================================================

import sys
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


def to_float(v: str):
    """Convertit une valeur BCT (virgule française) en float. '-' → NaN."""
    v = v.replace(",", ".").strip()
    if v in ("", "-", "--", "N/A"):
        return np.nan
    return float(v)


def parse_bct_html(filepath: str) -> pd.DataFrame:
    """
    Parse le fichier BCT_statistque.xls (HTML) et retourne
    un DataFrame Date (index) | TND_USD | TND_EUR.
    """
    print(f"📥 Lecture {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    soup = BeautifulSoup(content, "lxml")

    # --- Table des dates ---
    date_thead = soup.find_all("thead")[0]
    date_rows  = date_thead.find_all("tr")
    dates = []
    for r in date_rows[1:]:  # ligne 0 = en-tête "Date/Indicateurs"
        td = r.find("td")
        if td:
            txt = td.get_text(strip=True)
            if txt and txt != "Date/Indicateurs":
                dates.append(txt)

    # --- Table des valeurs Dollar / Euro ---
    tbody = soup.find("tbody", class_="bct-mod-content-1")
    val_rows = tbody.find_all("tr")
    usd_vals, eur_vals = [], []
    for r in val_rows:
        tds = r.find_all("td")
        if len(tds) == 2:
            usd_vals.append(tds[0].get_text(strip=True))
            eur_vals.append(tds[1].get_text(strip=True))

    if len(dates) != len(usd_vals) or len(dates) != len(eur_vals):
        raise ValueError(
            f"❌ Désalignement : {len(dates)} dates vs "
            f"{len(usd_vals)} USD / {len(eur_vals)} EUR"
        )

    print(f"   ✅ {len(dates)} lignes alignées "
          f"(dates ↔ valeurs)")

    df = pd.DataFrame({
        "Date":    pd.to_datetime(dates, format="%d/%m/%Y"),
        "TND_USD": [to_float(v) for v in usd_vals],
        "TND_EUR": [to_float(v) for v in eur_vals],
    })
    df = df.set_index("Date").sort_index()

    # Vérifications qualité
    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        print(f"   ⚠️  {n_dup} dates dupliquées — conservation "
              f"de la dernière occurrence")
        df = df[~df.index.duplicated(keep="last")]

    n_nan_usd = df["TND_USD"].isna().sum()
    n_nan_eur = df["TND_EUR"].isna().sum()
    print(f"   Valeurs manquantes ('-') : "
          f"USD={n_nan_usd} | EUR={n_nan_eur}")
    print(f"   Période : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"   TND/USD : min={df['TND_USD'].min():.4f} | "
          f"max={df['TND_USD'].max():.4f}")
    print(f"   TND/EUR : min={df['TND_EUR'].min():.4f} | "
          f"max={df['TND_EUR'].max():.4f}")

    return df


if __name__ == "__main__":
    input_path  = sys.argv[1] if len(sys.argv) > 1 else "data/raw/BCT_statistque.xls"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/raw/BCT_taux_change.csv"

    df = parse_bct_html(input_path)

    # NOTE : les NaN ('-') ne sont PAS interpolés ici.
    # clean_rate_series() dans data.py s'en charge déjà
    # (réindexage jours ouvrés + interpolation linéaire).
    df.to_csv(output_path)
    print(f"\n💾 {output_path} régénéré — {len(df)} lignes")