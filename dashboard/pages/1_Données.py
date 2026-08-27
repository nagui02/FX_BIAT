# =============================================================
#  dashboard/pages/1_Données.py
#  BIAT FX Intelligence — Données de marché
#  Auteur : Youssef Neji | MINDS ENIT | 2026
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json, os

st.set_page_config(page_title="Données — BIAT FX",
                   page_icon="📊", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE_DIR, "..", "data", "processed")
LIVE = os.path.join(BASE_DIR, "..", "data", "live")

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
.live-badge{display:inline-flex;align-items:center;gap:5px;
  background:rgba(34,165,103,.15);color:#22A567;padding:3px 11px;
  border-radius:20px;font-family:var(--mono);font-size:.68rem;
  border:1px solid rgba(34,165,103,.3)}
.pulse{width:6px;height:6px;border-radius:50%;background:#22A567;
  animation:pulse 1.6s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.box{background:var(--surface);border-left:3px solid var(--gold);
  border-radius:0 10px 10px 0;padding:.8rem 1.1rem;font-size:.85rem;
  color:var(--text);margin-top:.5rem;line-height:1.55}
.box b{color:var(--gold);font-family:var(--mono)}
[data-testid="stMetricValue"]{font-family:var(--mono)!important;
  color:var(--gold)!important}
[data-testid="stMetricLabel"]{color:var(--dim)!important}
.stSelectbox>div>div,.stDateInput>div>div{background:var(--surface2)!important;
  color:var(--text)!important;border-color:var(--border)!important}
.stTabs [data-baseweb="tab"]{color:var(--dim)!important;
  font-family:var(--sans)!important}
.stTabs [data-baseweb="tab"][aria-selected="true"]{color:var(--gold)!important;
  border-bottom-color:var(--gold)!important}
.stButton>button{background:var(--surface2)!important;color:var(--text)!important;
  border:1px solid rgba(201,168,76,.25)!important;border-radius:7px!important;
  font-weight:600!important;font-size:.78rem!important}
.stButton>button:hover{background:var(--gold)!important;color:#001327!important}
.stAlert,.stInfo{background:var(--surface)!important;color:var(--text)!important}
.stDataFrame{font-family:var(--mono)!important}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

REGIMES = {
    "Quasi-fixité (2004–2011)":
        ("2004-01-01","2011-12-31","rgba(15,95,168,.10)"),
    "Transition (2012–2015)":
        ("2012-01-01","2015-12-31","rgba(201,168,76,.09)"),
    "Dépréciation (2016–2023)":
        ("2016-01-01","2023-12-31","rgba(229,72,77,.08)"),
    "Test (2024–2026)":
        ("2024-01-01","2026-12-31","rgba(34,165,103,.08)"),
}
PERIODS = {"1S":"7D","1M":"30D","3M":"90D","6M":"180D","1A":"365D","MAX":None}
VOL_WIN = {"5j":5,"20j":20,"60j":60,"252j":252}
LBL = {"TND_USD":"TND/USD","TND_EUR":"TND/EUR","Brent_USD":"Brent",
       "VIX":"VIX","TauxFed_USD":"Fed","TauxBCE_EUR":"BCE",
       "TauxMMBCT":"BCT","EURUSD":"EUR/USD","IPC_USA":"IPC USA"}

@st.cache_data(ttl=300)
def load_df():
    return pd.read_csv(os.path.join(PROC,"dataset_final.csv"),
                       parse_dates=["Date"], index_col="Date")

@st.cache_data(ttl=300)
def load_live():
    p = os.path.join(LIVE,"latest_rates.json")
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return {}

df, live = load_df(), load_live()

def period_bar(key, src):
    sel = st.session_state.get(f"p_{key}","MAX")
    cols = st.columns(len(PERIODS))
    for col,(lbl,_) in zip(cols, PERIODS.items()):
        if col.button(lbl, key=f"{key}_{lbl}", use_container_width=True):
            sel = lbl
            st.session_state[f"p_{key}"] = lbl
            st.rerun()
    offset = PERIODS[sel]
    return src if offset is None else src.loc[src.index[-1]-pd.Timedelta(offset):]

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

def regimes(fig, show, rows=None):
    if not show: return
    for _,(s,e,fill) in REGIMES.items():
        kw = dict(x0=s,x1=e,fillcolor=fill,layer="below",line_width=0)
        if rows:
            for r in rows: fig.add_vrect(**kw,row=r,col=1)
        else: fig.add_vrect(**kw)

# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="ptitle">Données de Marché</div>',
           unsafe_allow_html=True)
st.caption(f"{len(df):,} observations · "
          f"2004-01-02 → {df.index[-1].date()} · BCT · FRED · INS")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("⚙ Filtres")
st.sidebar.markdown("---")
regime_sel = st.sidebar.selectbox("Régime", ["Tous"]+list(REGIMES.keys()))
devises = st.sidebar.multiselect("Devises", ["TND/USD","TND/EUR"],
                                 default=["TND/USD","TND/EUR"])
vol_lbl = st.sidebar.selectbox("Fenêtre volatilité", list(VOL_WIN.keys()),
                               index=1)
vol_w = VOL_WIN[vol_lbl]
show_reg = st.sidebar.checkbox("Bandes de régimes", value=True)
st.sidebar.markdown("---")
st.sidebar.caption(
    "**5j** réactive · **20j** standard risque · **60j** stress "
    "prolongé · **252j** tendance structurelle"
)

df_base = (df.loc[REGIMES[regime_sel][0]:REGIMES[regime_sel][1]]
           if regime_sel != "Tous" else df)

# ── Live KPIs ─────────────────────────────────────────────────
if live:
    r, m = live.get("rates",{}), live.get("macro",{})
    upd = live.get("last_updated","")[:16].replace("T"," ")
    st.markdown(f'<span class="live-badge"><span class="pulse"></span>'
               f'LIVE — {upd}</span>', unsafe_allow_html=True)
    usd, eur = r.get("TND_USD",{}), r.get("TND_EUR",{})
    cols = st.columns(6)
    vals = [("TND/USD",f"{usd.get('value',0):.4f}",
             f"{usd.get('change_1d',0):+.4f}",True),
            ("TND/EUR",f"{eur.get('value',0):.4f}",
             f"{eur.get('change_1d',0):+.4f}",True),
            ("Brent",f"{m.get('Brent_USD','—')}",None,False),
            ("VIX",f"{m.get('VIX','—')}",None,False),
            ("EUR/USD",f"{m.get('EURUSD','—')}",None,False),
            ("Taux Fed",f"{m.get('TauxFed','—')}%",None,False)]
    for col,(l,v,d,inv) in zip(cols, vals):
        with col:
            st.metric(l, v, d, delta_color="inverse" if inv else "normal")
    st.divider()

# ── Cours historiques ─────────────────────────────────────────
st.subheader("Cours de change historiques")
df_ch = period_bar("cours", df_base)

if devises:
    cmap = {"TND/USD":NAVY,"TND/EUR":ORANGE}
    n = len(devises)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=[f"TND/{d.split('/')[1]}"
                                       for d in devises],
                        vertical_spacing=.06)
    for i, dev in enumerate(devises, 1):
        col = "TND_EUR" if "EUR" in dev else "TND_USD"
        s = df_ch[col].dropna()
        regimes(fig, show_reg, rows=[i])
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=dev,
                                 line=dict(color=cmap[dev], width=1.6),
                                 hovertemplate="%{x|%d/%m/%Y}<br>"
                                 f"{dev}: %{{y:.4f}}<extra></extra>"),
                     row=i, col=1)
    axes(fig)
    fig.update_layout(
        height=210*n, hovermode="x unified",
        **dark_layout(legend=dict(orientation="h", y=1.06,
                                  x=1, xanchor="right"))
    )
    st.plotly_chart(fig, use_container_width=True)

    if len(df_ch) > 1:
        parts = []
        for dev in devises:
            col = "TND_EUR" if "EUR" in dev else "TND_USD"
            d = ((df_ch[col].iloc[-1]-df_ch[col].iloc[0])
                /df_ch[col].iloc[0]*100)
            parts.append(f"TND/{dev.split('/')[1]}: "
                        f"<b>{d:+.1f}%</b> "
                        f"({'dépréciation' if d>0 else 'appréciation'})")
        per = st.session_state.get("p_cours","MAX")
        st.markdown(f'<div class="box">Période <b>{per}</b> — '
                   f'{" · ".join(parts)}. TND/USD structurellement '
                   f'plus volatile (chocs pétrole + politique Fed).'
                   f'</div>', unsafe_allow_html=True)

st.divider()

# ── Log-rendements ───────────────────────────────────────────
st.subheader("Log-rendements et volatilité")
df_lr = period_bar("logret", df_base)
t1, t2 = st.tabs(["Rendements journaliers", f"Volatilité ({vol_lbl})"])

with t1:
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("LogRet TND/USD","LogRet TND/EUR"),
                         vertical_spacing=.08)
    for row,(col,color,dev) in enumerate(
        [("LogRet_TND_USD",NAVY,"USD"),("LogRet_TND_EUR",ORANGE,"EUR")], 1):
        s = df_lr[col].dropna()
        if len(s)==0: continue
        sig = s.std()
        fig2.add_trace(go.Scatter(
            x=s.index.tolist()+s.index.tolist()[::-1],
            y=[2*sig]*len(s)+[-2*sig]*len(s), fill="toself",
            fillcolor="rgba(139,150,172,.08)",
            line=dict(color="rgba(0,0,0,0)"), name="±2σ",
            showlegend=(row==1)), row=row, col=1)
        fig2.add_trace(go.Scatter(x=s.index, y=s.values, name=f"TND/{dev}",
                                  line=dict(color=color,width=.7),
                                  opacity=.88), row=row, col=1)
        fig2.add_hline(y=0, line_dash="dot",
                       line_color="rgba(255,255,255,.15)", row=row, col=1)
        fig2.update_yaxes(title_text=f"σ ann.={sig*np.sqrt(252)*100:.2f}%",
                          row=row, col=1)
    regimes(fig2, show_reg, rows=[1,2])
    axes(fig2)
    fig2.update_layout(height=400, hovermode="x unified", **dark_layout())
    st.plotly_chart(fig2, use_container_width=True)

    lr_u, lr_e = df_lr["LogRet_TND_USD"].dropna(), df_lr["LogRet_TND_EUR"].dropna()
    if len(lr_u)>1:
        v_u = lr_u.std()*np.sqrt(252)*100
        v_e = lr_e.std()*np.sqrt(252)*100
        kurt, skew = lr_u.kurtosis(), lr_u.skew()
        st.markdown(f'<div class="box">Volatilité — USD '
                   f'<b>{v_u:.2f}%/an</b> vs EUR <b>{v_e:.2f}%/an</b>. '
                   f'Kurtosis={kurt:.1f} (queues épaisses), '
                   f'Skewness={skew:.2f}.</div>', unsafe_allow_html=True)

with t2:
    fig3 = go.Figure()
    for col,color,dev in [("LogRet_TND_USD",NAVY,"USD"),
                          ("LogRet_TND_EUR",ORANGE,"EUR")]:
        s = df_lr[col].dropna()
        if len(s)<vol_w: continue
        v = s.rolling(vol_w).std()*np.sqrt(252)*100
        fig3.add_trace(go.Scatter(x=v.index, y=v.values,
                                  name=f"Vol {dev}", line=dict(color=color,width=1.3),
                                  fill="tozeroy",
                                  fillcolor=f"rgba({15 if dev=='USD' else 232},"
                                  f"{95 if dev=='USD' else 151},"
                                  f"{168 if dev=='USD' else 74},.08)"))
    regimes(fig3, show_reg)
    axes(fig3)
    fig3.update_layout(yaxis_title="Vol. annualisée (%)", height=340,
                       hovermode="x unified", **dark_layout())
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Macro ─────────────────────────────────────────────────────
st.subheader("Variables macroéconomiques")
df_mac = period_bar("macro", df_base)
tab1, tab2, tab3 = st.tabs(["Énergie & Risque", "Taux d'intérêt", "Inflation"])

with tab1:
    fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("Brent (USD/bbl)","VIX"),
                         vertical_spacing=.1)
    if "Brent_USD" in df_mac.columns:
        b = df_mac["Brent_USD"].dropna()
        fig4.add_trace(go.Scatter(x=b.index, y=b.values, name="Brent",
                                  line=dict(color=GREEN,width=1.3),
                                  fill="tozeroy",
                                  fillcolor="rgba(34,165,103,.08)"),
                     row=1, col=1)
    if "VIX" in df_mac.columns:
        v = df_mac["VIX"].dropna()
        fig4.add_trace(go.Scatter(x=v.index, y=v.values, name="VIX",
                                  line=dict(color=RED,width=1.1)), row=2, col=1)
        fig4.add_hrect(y0=20, y1=float(v.max())+5,
                      fillcolor="rgba(229,72,77,.07)", line_width=0,
                      row=2, col=1)
    regimes(fig4, show_reg, rows=[1,2])
    axes(fig4)
    fig4.update_layout(height=400, hovermode="x unified", **dark_layout())
    st.plotly_chart(fig4, use_container_width=True)

with tab2:
    fig5 = go.Figure()
    for col,lbl,color in [("TauxFed_USD","Fed",NAVY),
                          ("TauxBCE_EUR","BCE",ORANGE),
                          ("TauxMMBCT","BCT",GREEN)]:
        if col in df_mac.columns:
            s = df_mac[col].dropna()
            fig5.add_trace(go.Scatter(x=s.index, y=s.values, name=lbl,
                                      line=dict(color=color,width=1.7)))
    regimes(fig5, show_reg)
    axes(fig5)
    fig5.update_layout(yaxis_title="Taux (%)", height=360,
                       hovermode="x unified", **dark_layout())
    st.plotly_chart(fig5, use_container_width=True)

with tab3:
    fig6 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("IPC USA","IPC Tunisie"),
                         vertical_spacing=.1)
    if "IPC_USA" in df_mac.columns:
        s = df_mac["IPC_USA"].dropna()
        fig6.add_trace(go.Scatter(x=s.index, y=s.values, name="IPC USA",
                                  line=dict(color="#8B5CF6",width=1.3)),
                     row=1, col=1)
    if "IPC_Tunisie" in df_mac.columns:
        s = df_mac["IPC_Tunisie"].dropna()
        if len(s)>0:
            fig6.add_trace(go.Scatter(x=s.index, y=s.values,
                                      name="IPC Tunisie",
                                      line=dict(color=GOLD,width=1.4)),
                         row=2, col=1)
    axes(fig6)
    fig6.update_layout(height=400, hovermode="x unified", **dark_layout())
    st.plotly_chart(fig6, use_container_width=True)

st.divider()

# ── Corrélations ──────────────────────────────────────────────
st.subheader("Corrélations — recalculées sur la période")
df_corr = period_bar("corr", df_base)
cols_c = [c for c in ["TND_USD","TND_EUR","Brent_USD","VIX","TauxFed_USD",
                      "TauxBCE_EUR","TauxMMBCT","EURUSD","IPC_USA"]
         if c in df_corr.columns]

cl, cr = st.columns(2)
with cl:
    corr = df_corr[cols_c].corr().round(2)
    lbls = [LBL.get(c,c) for c in cols_c]
    fig7 = px.imshow(corr, x=lbls, y=lbls, text_auto=True,
                     color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig7.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#EDF1F7"),
                       margin=dict(l=18,r=18,t=18,b=18))
    st.plotly_chart(fig7, use_container_width=True)

with cr:
    r_lv = df_corr["TND_USD"].corr(df_corr["TND_EUR"])
    r_lr = df_corr["LogRet_TND_USD"].corr(df_corr["LogRet_TND_EUR"])
    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(x=df_corr["TND_USD"], y=df_corr["TND_EUR"],
                              mode="markers",
                              marker=dict(size=2.2, color=df_corr.index.year,
                                         colorscale="Viridis", showscale=True,
                                         colorbar=dict(title="Année",
                                                      thickness=11))))
    fig8.add_annotation(x=.05, y=.97, xref="paper", yref="paper",
                        text=f"r niveaux={r_lv:.4f}<br>r log-ret={r_lr:.4f}",
                        showarrow=False, bgcolor="rgba(19,26,42,.9)",
                        bordercolor=GOLD, borderwidth=1,
                        font=dict(size=11, color="#EDF1F7",
                                 family="IBM Plex Mono"))
    axes(fig8)
    fig8.update_layout(height=400, xaxis_title="TND/USD",
                       yaxis_title="TND/EUR", **dark_layout())
    st.plotly_chart(fig8, use_container_width=True)
    st.markdown(f'<div class="box">r log-ret=<b>{r_lr:.4f}</b> — '
               f'{"mouvements corrélés à court terme" if r_lr>.5 else "dynamiques indépendantes"}.'
               f'</div>', unsafe_allow_html=True)

st.divider()
st.subheader("Statistiques descriptives")
df_st = period_bar("stats", df_base)
cols_s = [c for c in ["TND_USD","TND_EUR","Brent_USD","VIX","TauxFed_USD",
                      "TauxBCE_EUR","TauxMMBCT","EURUSD"]
         if c in df_st.columns]
stats = df_st[cols_s].describe().round(4)
stats.columns = [LBL.get(c,c) for c in cols_s]
st.dataframe(stats, use_container_width=True)

st.caption(f"Source : BCT · FRED · INS · Youssef Neji — MINDS ENIT 2026")