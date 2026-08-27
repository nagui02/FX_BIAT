# =============================================================
#  dashboard/pages/5_Markowitz.py
#  BIAT FX Intelligence — Portefeuille (Markowitz)
#  Auteur : Youssef Neji | MINDS ENIT | 2026
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys

st.set_page_config(page_title="Portefeuille — BIAT FX",
                   page_icon="📐", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE_DIR, "..", "data", "processed")
ROOT = os.path.join(BASE_DIR, "..")
sys.path.insert(0, ROOT)

from src.s4_risk.config import CLIENT_PROFILES
from utils.ai_interpret import get_markowitz_interpretation

NAVY, GOLD, GREEN = "#0F5FA8", "#C9A84C", "#22A567"
RED, ORANGE, DIM  = "#E5484D", "#E8974A", "#8B96AC"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{--bg:#0A0E17;--surface:#131A2A;--surface2:#1B2438;
  --gold:#C9A84C;--text:#EDF1F7;--dim:#8B96AC;--border:rgba(255,255,255,.07);
  --mono:'IBM Plex Mono',monospace;--sans:'Inter',sans-serif;--disp:'Sora',sans-serif}
html,body,.stApp{background:var(--bg)!important;color:var(--text)!important;
  font-family:var(--sans)}
section[data-testid="stSidebar"]{background:#070A11!important}
section[data-testid="stSidebar"] *{color:var(--text)!important}
h1,h2,h3,h4{font-family:var(--disp)!important;color:var(--text)!important}
.ptitle{font-family:var(--disp);font-weight:800;font-size:1.7rem;
  color:var(--text);border-left:4px solid var(--gold);
  padding-left:1rem;margin-bottom:.2rem}

.kpi-card{background:linear-gradient(160deg,#0A1A33,#0F2542);
  border-radius:14px;padding:1.2rem;text-align:center;
  border:1px solid rgba(201,168,76,.25)}
.kpi-card.tangent{border-color:var(--gold);border-width:2px;
  box-shadow:0 0 20px rgba(201,168,76,.2)}
.kpi-lbl{font-family:var(--sans);font-size:.68rem;color:rgba(255,255,255,.55);
  text-transform:uppercase;letter-spacing:.07em}
.kpi-val{font-family:var(--mono);font-size:1.55rem;font-weight:600;
  color:var(--gold);font-variant-numeric:tabular-nums}
.kpi-sub{font-family:var(--mono);font-size:.7rem;color:var(--dim);margin-top:.2rem}

.profile-card{background:var(--surface);border:1px solid var(--border);
  border-radius:12px;padding:1rem 1.2rem;margin-bottom:1rem}
.profile-card b{color:var(--gold);font-family:var(--mono)}

.box{background:var(--surface);border-left:3px solid var(--gold);
  border-radius:0 10px 10px 0;padding:.8rem 1.1rem;font-size:.85rem;
  color:var(--text);margin-top:.5rem;line-height:1.55}
.box b{color:var(--gold);font-family:var(--mono)}
.box.ok{border-left-color:var(--green)}
.box.warn{border-left-color:var(--red)}

.badge{display:inline-block;padding:2px 10px;border-radius:20px;
  font-family:var(--mono);font-size:.68rem;font-weight:600;margin-right:6px}
.badge-ai{background:rgba(139,92,246,.15);color:#8B5CF6;
  border:1px solid rgba(139,92,246,.3)}
.badge-off{background:rgba(139,150,172,.18);color:var(--dim);
  border:1px solid rgba(139,150,172,.28)}

[data-testid="stMetricValue"]{font-family:var(--mono)!important;
  color:var(--gold)!important}
[data-testid="stMetricLabel"]{color:var(--dim)!important}
.stTabs [data-baseweb="tab"]{color:var(--dim)!important}
.stTabs [data-baseweb="tab"][aria-selected="true"]{color:var(--gold)!important;
  border-bottom-color:var(--gold)!important}
.stAlert,.stInfo,.stWarning{background:var(--surface)!important;color:var(--text)!important}
.stDataFrame{font-family:var(--mono)!important}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load_summary():  return pd.read_csv(f"{PROC}/markowitz_summary.csv")
@st.cache_data
def load_cml():       return pd.read_csv(f"{PROC}/markowitz_cml.csv")
@st.cache_data
def load_frontier():  return pd.read_csv(f"{PROC}/markowitz_frontier.csv")


def dark_layout(**kw):
    base = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
               font=dict(color="#EDF1F7", family="IBM Plex Mono"),
               legend=dict(bgcolor="rgba(0,0,0,0)"),
               margin=dict(l=55,r=18,t=28,b=38))
    base.update(kw)
    return base

def axes(fig):
    kw = dict(gridcolor="rgba(255,255,255,.05)", showgrid=True)
    fig.update_xaxes(**kw); fig.update_yaxes(**kw)


def find_row(df, needle):
    m = df[df["portefeuille"].str.contains(needle, regex=False)]
    return m.iloc[0] if not m.empty else None


# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="ptitle">Portefeuille — Optimisation Markowitz</div>',
           unsafe_allow_html=True)
st.caption("Allocation EUR/USD — Desk BIAT · frontière efficiente · "
          "Capital Market Line")

missing = [f for f in ("markowitz_summary.csv", "markowitz_cml.csv",
                       "markowitz_frontier.csv")
          if not os.path.exists(os.path.join(PROC, f))]
if missing:
    st.warning(f"Fichiers manquants : {', '.join(missing)} — "
              f"lancez `python -m src.s5_hedging.run_s5`")
    st.stop()

summary_df = load_summary()
cml_df = load_cml()
frontier_df = load_frontier()

row_current   = find_row(summary_df, "actuelle")
row_minvar    = find_row(summary_df, "Variance minimale")
row_maxsharpe = find_row(summary_df, "Sharpe maximal")
if row_current is None or row_minvar is None or row_maxsharpe is None:
    st.warning("markowitz_summary.csv ne contient pas les 3 portefeuilles "
              "attendus (Position actuelle / Variance minimale / Sharpe "
              "maximal) — relancez `run_s5.py`.")
    st.stop()
cml = cml_df.iloc[0]

profile = CLIENT_PROFILES.get("Desk BIAT", {})

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("⚙ Configuration")
st.sidebar.markdown(f"""
<div class="profile-card">
<b>Desk BIAT</b><br>
<span style="color:var(--dim);font-size:.8rem">{profile.get('role','')}</span><br>
<span style="color:var(--dim);font-size:.78rem">{profile.get('description','')}</span><br><br>
Devise : <b>EUR + USD</b><br>
Montant : <b>{profile.get('amount',0):,.0f}</b> TND
</div>
""", unsafe_allow_html=True)

groq_key = os.environ.get("GROQ_API_KEY", "")
ai_avail_mk = bool(groq_key)
st.sidebar.markdown("---")
use_ai_mk = (st.sidebar.checkbox("🤖 Interprétation Groq", value=False,
                                 key="ai_mk")
            if ai_avail_mk else False)
st.sidebar.caption("✅ Clé Groq détectée" if ai_avail_mk
                   else "⚙ Mode offline")

# ── KPI row ───────────────────────────────────────────────────
kc = st.columns(4)
for col, (label, row, tag) in zip(kc[:3], [
    ("Position actuelle", row_current, ""),
    ("Variance minimale", row_minvar, ""),
    ("Sharpe maximal", row_maxsharpe, "tangent"),
]):
    with col:
        st.markdown(f"""<div class="kpi-card {tag}">
          <div class="kpi-lbl">{label}</div>
          <div class="kpi-val">{row['poids_EUR_pct']:.0f}/{row['poids_USD_pct']:.0f}</div>
          <div class="kpi-sub">EUR/USD · Sharpe {row['sharpe_ratio']:.3f}</div>
          </div>""", unsafe_allow_html=True)
with kc[3]:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-lbl">Taux TND (référence)</div>
      <div class="kpi-val">{cml['taux_TND']*100:.2f}%</div>
      <div class="kpi-sub">actif sans risque</div>
      </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Conclusion CML ───────────────────────────────────────────
tnd_domine = bool(cml["TND_domine_toute_la_frontiere"])
rf = float(cml["taux_TND"])
tangent_ret = float(cml["rendement_tangent_FX"])
tangent_vol = float(cml["volatilite_tangent_FX"])

box_cls = "ok" if tnd_domine else "warn"
if tnd_domine:
    conclusion_txt = (
        f"✅ Le taux TND (<b>{rf*100:.2f}%</b>) domine intégralement la "
        f"frontière EUR/USD — écart de <b>{cml['ecart_pts']:.2f} points</b> "
        f"avec le meilleur rendement FX atteignable "
        f"(<b>{cml['rendement_FX_max_possible']*100:.2f}%</b>). Aucune "
        f"combinaison de devises n'égale le cash TND à ce niveau de risque."
    )
else:
    conclusion_txt = (
        f"⚠️ Le portefeuille tangent EUR/USD (<b>{tangent_ret*100:.2f}%</b>) "
        f"dépasse le taux TND (<b>{rf*100:.2f}%</b>) — une allocation en "
        f"devises peut améliorer le couple rendement/risque par rapport "
        f"au cash TND seul."
    )
st.markdown(f'<div class="box {box_cls}">{conclusion_txt}</div>',
           unsafe_allow_html=True)

badge = ('<span class="badge badge-ai">🤖 Groq</span>'
        if use_ai_mk and ai_avail_mk else
        '<span class="badge badge-off">⚙ Offline</span>')
interp_kwargs = dict(
    current_eur_pct=float(row_current["poids_EUR_pct"]),
    current_sharpe=float(row_current["sharpe_ratio"]),
    minvar_eur_pct=float(row_minvar["poids_EUR_pct"]),
    minvar_sharpe=float(row_minvar["sharpe_ratio"]),
    maxsharpe_eur_pct=float(row_maxsharpe["poids_EUR_pct"]),
    maxsharpe_sharpe=float(row_maxsharpe["sharpe_ratio"]),
    rf_pct=rf*100, tangent_return_pct=tangent_ret*100,
    ecart_pts=float(cml["ecart_pts"]), tnd_domine=tnd_domine,
)
if use_ai_mk and ai_avail_mk:
    with st.spinner("Analyse Groq..."):
        interp = get_markowitz_interpretation(use_ai=True, **interp_kwargs)
else:
    interp = get_markowitz_interpretation(use_ai=False, **interp_kwargs)
st.markdown(f'<div class="box">{badge} {interp}</div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["⚖ Comparaison", "📈 Frontière & CML", "🔢 Détails techniques"])

# ═══════════════════════════════════════════════════════════
#  TAB 1 — Comparaison des allocations
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Allocation par portefeuille")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="EUR", x=summary_df["portefeuille"],
                         y=summary_df["poids_EUR_pct"], marker_color=NAVY))
    fig.add_trace(go.Bar(name="USD", x=summary_df["portefeuille"],
                         y=summary_df["poids_USD_pct"], marker_color=ORANGE))
    layout = dark_layout(height=360, barmode="stack",
                        yaxis_title="Allocation (%)",
                        legend=dict(orientation="h",
                                  bgcolor="rgba(0,0,0,0)"))
    fig.update_layout(**layout)
    axes(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Rendement / Volatilité / Sharpe")
    fig_rvs = go.Figure()
    fig_rvs.add_trace(go.Bar(name="Rendement ann. (%)",
                             x=summary_df["portefeuille"],
                             y=summary_df["rendement_ann_pct"],
                             marker_color=GREEN))
    fig_rvs.add_trace(go.Bar(name="Volatilité ann. (%)",
                             x=summary_df["portefeuille"],
                             y=summary_df["volatilite_ann_pct"],
                             marker_color=RED))
    layout2 = dark_layout(height=340, barmode="group",
                         yaxis_title="%",
                         legend=dict(orientation="h",
                                   bgcolor="rgba(0,0,0,0)"))
    fig_rvs.update_layout(**layout2)
    axes(fig_rvs)
    st.plotly_chart(fig_rvs, use_container_width=True)

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════
#  TAB 2 — Frontière efficiente + Capital Market Line
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Frontière efficiente EUR/USD + Capital Market Line")

    fig2 = go.Figure()

    if not frontier_df.empty:
        fig2.add_trace(go.Scatter(
            x=(frontier_df["volatilite"]*100), y=(frontier_df["rendement"]*100),
            mode="lines", name="Frontière efficiente",
            line=dict(color=DIM, width=1.5)))

    x_max = max(float(frontier_df["volatilite"].max()) if not frontier_df.empty else tangent_vol,
               tangent_vol) * 1.15
    slope = float(cml["pente_CML"])
    fig2.add_trace(go.Scatter(
        x=[0, x_max*100], y=[rf*100, (rf + slope*x_max)*100],
        mode="lines", name="Capital Market Line",
        line=dict(color=GOLD, width=2, dash="dash")))

    fig2.add_trace(go.Scatter(
        x=[0], y=[rf*100], mode="markers+text",
        marker=dict(size=12, color=GREEN, symbol="diamond"),
        text=["TND"], textposition="top center", textfont=dict(color=GREEN),
        name="TND (sans risque)"))
    fig2.add_trace(go.Scatter(
        x=[float(row_current["volatilite_ann_pct"])],
        y=[float(row_current["rendement_ann_pct"])],
        mode="markers+text", marker=dict(size=11, color="white"),
        text=["Actuel"], textposition="top center",
        name="Position actuelle"))
    fig2.add_trace(go.Scatter(
        x=[float(row_minvar["volatilite_ann_pct"])],
        y=[float(row_minvar["rendement_ann_pct"])],
        mode="markers+text", marker=dict(size=11, color=NAVY),
        text=["Min-Var"], textposition="top center",
        name="Variance minimale"))
    fig2.add_trace(go.Scatter(
        x=[tangent_vol*100], y=[tangent_ret*100],
        mode="markers+text", marker=dict(size=14, color=GOLD, symbol="star"),
        text=["Tangent"], textposition="top center",
        name="Sharpe maximal"))

    layout3 = dark_layout(height=460,
                         xaxis_title="Volatilité annualisée (%)",
                         yaxis_title="Rendement annualisé (%)",
                         legend=dict(orientation="h",
                                   bgcolor="rgba(0,0,0,0)"))
    fig2.update_layout(**layout3)
    axes(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f'<div class="box">📐 Pente de la CML : '
               f'<b>{slope:.4f}</b> ({"négative" if slope<0 else "positive"}) '
               f'— le "prix du risque" : rendement additionnel obtenu par '
               f'unité de volatilité prise en s\'éloignant du TND pur vers '
               f'le portefeuille tangent.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  TAB 3 — Détails techniques
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("**Comparaison des portefeuilles**")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.markdown("**Capital Market Line**")
    st.dataframe(cml_df, use_container_width=True, hide_index=True)
    if not frontier_df.empty:
        st.markdown("**Frontière efficiente (points calculés)**")
        st.dataframe(frontier_df.round(5), use_container_width=True,
                    hide_index=True)

st.divider()
st.caption("Markowitz · Capital Market Line — "
          "Youssef Neji — MINDS ENIT 2026")