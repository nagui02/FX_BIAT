# =============================================================
#  dashboard/pages/4_Couverture.py
#  BIAT FX Intelligence — Couverture de change
#  Auteur : Youssef Neji | MINDS ENIT | 2026
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys

st.set_page_config(page_title="Couverture — BIAT FX",
                   page_icon="🛡️", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE_DIR, "..", "data", "processed")
ROOT = os.path.join(BASE_DIR, "..")
sys.path.insert(0, ROOT)

from src.s4_risk.config import CLIENT_PROFILES
from utils.ai_interpret import get_hedging_interpretation

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
.kpi-lbl{font-family:var(--sans);font-size:.68rem;color:rgba(255,255,255,.55);
  text-transform:uppercase;letter-spacing:.07em}
.kpi-val{font-family:var(--mono);font-size:1.55rem;font-weight:600;
  color:var(--gold);font-variant-numeric:tabular-nums}
.kpi-sub{font-family:var(--mono);font-size:.7rem;color:var(--dim);margin-top:.2rem}

.reco-card{background:linear-gradient(135deg,#001327,#003A70);
  border-radius:16px;padding:1.6rem;text-align:center;
  border:1px solid rgba(201,168,76,.35);margin:1rem 0}
.reco-instrument{font-family:var(--disp);font-weight:800;font-size:2rem;
  color:var(--gold)}
.reco-sub{font-family:var(--sans);font-size:.82rem;
  color:rgba(255,255,255,.6);margin-top:.4rem}

.profile-card{background:var(--surface);border:1px solid var(--border);
  border-radius:12px;padding:1rem 1.2rem;margin-bottom:1rem}
.profile-card b{color:var(--gold);font-family:var(--mono)}

.box{background:var(--surface);border-left:3px solid var(--gold);
  border-radius:0 10px 10px 0;padding:.8rem 1.1rem;font-size:.85rem;
  color:var(--text);margin-top:.5rem;line-height:1.55}
.box b{color:var(--gold);font-family:var(--mono)}

.badge{display:inline-block;padding:2px 10px;border-radius:20px;
  font-family:var(--mono);font-size:.68rem;font-weight:600;margin-right:6px}
.badge-ai{background:rgba(139,92,246,.15);color:#8B5CF6;
  border:1px solid rgba(139,92,246,.3)}
.badge-off{background:rgba(139,150,172,.18);color:var(--dim);
  border:1px solid rgba(139,150,172,.28)}

[data-testid="stMetricValue"]{font-family:var(--mono)!important;
  color:var(--gold)!important}
[data-testid="stMetricLabel"]{color:var(--dim)!important}
.stSelectbox>div>div{background:var(--surface2)!important;
  color:var(--text)!important;border-color:var(--border)!important}
.stTabs [data-baseweb="tab"]{color:var(--dim)!important}
.stTabs [data-baseweb="tab"][aria-selected="true"]{color:var(--gold)!important;
  border-bottom-color:var(--gold)!important}
.stAlert,.stInfo{background:var(--surface)!important;color:var(--text)!important}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load_forward():   return pd.read_csv(f"{PROC}/forward_results.csv")
@st.cache_data
def load_options():   return pd.read_csv(f"{PROC}/options_results.csv")
@st.cache_data
def load_comparison(): return pd.read_csv(f"{PROC}/hedging_comparison.csv")


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


# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="ptitle">Couverture de Change</div>',
           unsafe_allow_html=True)
st.caption("Forward vs Options (Garman-Kohlhagen) — recommandation "
          "pilotée par les modèles de prévision")

fwd_df = load_forward()
opt_df = load_options()
comp_df = load_comparison()

hedgeable_profiles = [p for p in CLIENT_PROFILES
                      if CLIENT_PROFILES[p]["currency"] != "MULTI"]

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("⚙ Configuration")
profile_name = st.sidebar.selectbox("Profil client", hedgeable_profiles)
horizon = st.sidebar.selectbox("Horizon de couverture", [7, 30],
                               format_func=lambda h: f"J+{h}")
profile = CLIENT_PROFILES[profile_name]

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div class="profile-card">
<b>{profile_name}</b><br>
<span style="color:var(--dim);font-size:.8rem">{profile['role']}</span><br>
<span style="color:var(--dim);font-size:.78rem">{profile['description']}</span><br><br>
Devise : <b>{profile['currency']}</b><br>
Montant : <b>{profile['amount']:,.0f}</b> TND
</div>
""", unsafe_allow_html=True)

st.sidebar.caption("Desk BIAT non disponible ici — position "
                  "multi-devises, voir page Risque pour le "
                  "bénéfice de diversification.")

groq_key = os.environ.get("GROQ_API_KEY","")
ai_avail_hedge = bool(groq_key)
st.sidebar.markdown("---")
use_ai_hedge = (st.sidebar.checkbox("🤖 Interprétation Groq", value=False,
                                    key="ai_hedge")
               if ai_avail_hedge else False)
st.sidebar.caption("✅ Clé Groq détectée" if ai_avail_hedge
                   else "⚙ Mode offline")

# ── Data pour ce profil/horizon ─────────────────────────────────
fwd_row  = fwd_df[(fwd_df["profile"]==profile_name) &
                  (fwd_df["horizon_j"]==horizon)]
opt_row  = opt_df[(opt_df["profile"]==profile_name) &
                  (opt_df["horizon_j"]==horizon)]
comp_row = comp_df[(comp_df["profile"]==profile_name) &
                   (comp_df["horizon_j"]==horizon)]

if fwd_row.empty or opt_row.empty:
    st.warning("Données indisponibles pour ce profil/horizon.")
    st.stop()

f, o = fwd_row.iloc[0], opt_row.iloc[0]
c = comp_row.iloc[0] if not comp_row.empty else None

# ── KPI row ───────────────────────────────────────────────────
kc = st.columns(5)
with kc[0]:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-lbl">Spot actuel</div>
      <div class="kpi-val">{f['spot']:.4f}</div>
      <div class="kpi-sub">TND/{profile['currency']}</div></div>""",
      unsafe_allow_html=True)
with kc[1]:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-lbl">Forward J+{horizon}</div>
      <div class="kpi-val">{f['forward']:.4f}</div>
      <div class="kpi-sub">{f['sens'].split('(')[0].strip()}</div>
      </div>""", unsafe_allow_html=True)
with kc[2]:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-lbl">Prime Option</div>
      <div class="kpi-val">{o['prime_totale_tnd']:,.0f}</div>
      <div class="kpi-sub">TND · {o['type_option']}</div></div>""",
      unsafe_allow_html=True)
with kc[3]:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-lbl">Coût couverture Fwd</div>
      <div class="kpi-val">{f['cout_couverture_tnd']:,.0f}</div>
      <div class="kpi-sub">TND vs spot</div></div>""",
      unsafe_allow_html=True)
with kc[4]:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-lbl">Delta Option</div>
      <div class="kpi-val">{o['delta']:.3f}</div>
      <div class="kpi-sub">sensibilité au spot</div></div>""",
      unsafe_allow_html=True)

st.markdown("---")

# ── Recommandation ────────────────────────────────────────────
if c is not None and pd.notna(c["cours_predit"]):
    reco = c["recommandation_modele"]
    color = GOLD if reco=="Forward" else "#8B5CF6"
    st.markdown(f"""
<div class="reco-card">
  <div class="kpi-lbl">Recommandation pilotée par le modèle</div>
  <div class="reco-instrument" style="color:{color}">{reco}</div>
  <div class="reco-sub">Cours prédit à J+{horizon} : 
    <b style="color:var(--gold)">{c['cours_predit']:.5f}</b> · 
    Seuil de rentabilité Option : 
    <b style="color:var(--gold)">{c['seuil_rentabilite_option']:.5f}</b>
  </div>
</div>""", unsafe_allow_html=True)

    badge = ('<span class="badge badge-ai">🤖 Groq</span>'
            if use_ai_hedge and ai_avail_hedge else
            '<span class="badge badge-off">⚙ Offline</span>')

    interp_kwargs = dict(
        profile=profile_name, currency=profile["currency"],
        instrument_type=o["type_option"], horizon=horizon,
        forward_cost=f["cout_couverture_tnd"],
        option_premium=o["prime_totale_tnd"],
        breakeven=c["seuil_rentabilite_option"],
        forecast=c["cours_predit"], recommendation=reco
    )

    if use_ai_hedge and ai_avail_hedge:
        with st.spinner("Analyse Groq..."):
            interp = get_hedging_interpretation(use_ai=True, **interp_kwargs)
    else:
        interp = get_hedging_interpretation(use_ai=False, **interp_kwargs)

    st.markdown(f'<div class="box">{badge} {interp}</div>',
               unsafe_allow_html=True)
else:
    st.info("Prévision indisponible pour ce cours/horizon — "
           "recommandation basée sur le coût brut uniquement.")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["⚖ Comparaison", "📈 Diagramme de Payoff", "🔢 Détails techniques"])

# ═══════════════════════════════════════════════════════════
#  TAB 1 — Comparaison
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Forward vs Option — tous horizons")
    all_fwd = fwd_df[fwd_df["profile"]==profile_name]
    all_opt = opt_df[opt_df["profile"]==profile_name]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Coût Forward",
                         x=[f"J+{h}" for h in all_fwd["horizon_j"]],
                         y=all_fwd["cout_couverture_tnd"],
                         marker_color=GOLD))
    fig.add_trace(go.Bar(name="Prime Option",
                         x=[f"J+{h}" for h in all_opt["horizon_j"]],
                         y=all_opt["prime_totale_tnd"],
                         marker_color="#8B5CF6"))
    layout = dark_layout(height=380, barmode="group",
                        yaxis_title="Coût (TND)",
                        legend=dict(orientation="h",
                                  bgcolor="rgba(0,0,0,0)"))
    fig.update_layout(**layout)
    axes(fig)
    st.plotly_chart(fig, use_container_width=True)

    if not comp_df.empty:
        sub = comp_df[comp_df["profile"]==profile_name]
        st.dataframe(sub, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════
#  TAB 2 — Payoff
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"### Diagramme de Payoff à J+{horizon}")

    spot_range = np.linspace(f["spot"]*0.92, f["spot"]*1.08, 200)
    K = o["strike_atm"]
    prime = o["prime_unitaire"]
    direction = profile["direction"]

    if o["type_option"] == "CALL":
        opt_payoff = -direction * (np.maximum(spot_range - K, 0) - prime)
    else:
        opt_payoff = -direction * (np.maximum(K - spot_range, 0) - prime)

    fwd_payoff = -direction * (f["forward"] - spot_range)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=spot_range, y=fwd_payoff*profile["amount"],
                              name="Forward", line=dict(color=GOLD,width=2)))
    fig2.add_trace(go.Scatter(x=spot_range, y=opt_payoff*profile["amount"],
                              name=f"Option {o['type_option']}",
                              line=dict(color="#8B5CF6",width=2)))
    fig2.add_vline(x=f["spot"], line_dash="dot",
                   line_color="rgba(255,255,255,.3)",
                   annotation_text="Spot actuel")
    fig2.add_hline(y=0, line_dash="dot",
                   line_color="rgba(255,255,255,.2)")

    layout2 = dark_layout(height=420,
                         xaxis_title=f"Cours TND/{profile['currency']} final",
                         yaxis_title="Gain/Perte vs sans couverture (TND)",
                         legend=dict(orientation="h",
                                   bgcolor="rgba(0,0,0,0)"))
    fig2.update_layout(**layout2)
    axes(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="box">📐 Le Forward (ligne droite) fixe '
               'un résultat certain quel que soit le cours final. '
               'L\'Option (ligne coudée) plafonne la perte au montant '
               'de la prime, mais conserve un potentiel de gain si le '
               'marché évolue favorablement.</div>',
               unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  TAB 3 — Détails techniques
# ═══════════════════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Forward — Parité des taux couverts**")
        st.dataframe(f.to_frame().T, use_container_width=True,
                    hide_index=True)
    with col2:
        st.markdown("**Option — Garman-Kohlhagen**")
        st.dataframe(o.to_frame().T, use_container_width=True,
                    hide_index=True)

    st.markdown("---")
    kc = st.columns(3)
    with kc[0]: st.metric("Delta", f"{o['delta']:.4f}")
    with kc[1]: st.metric("Gamma", f"{o['gamma']:.6f}")
    with kc[2]: st.metric("Vega", f"{o['vega']:.6f}")
    st.caption("Vega exprimé par point de volatilité (1%)")

st.divider()
st.caption("Forward · Garman-Kohlhagen — Youssef Neji — MINDS ENIT 2026")