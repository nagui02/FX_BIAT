# =============================================================
#  dashboard/pages/3_Risque.py
#  BIAT FX Intelligence — Risque de change
#  Auteur : Youssef Neji | MINDS ENIT | 2026
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys

st.set_page_config(page_title="Risque — BIAT FX",
                   page_icon="⚠️", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE_DIR, "..", "data", "processed")
ROOT = os.path.join(BASE_DIR, "..")
sys.path.insert(0, ROOT)

from src.s4_risk.config import CLIENT_PROFILES
from utils.ai_interpret import get_var_interpretation

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
.kpi-val{font-family:var(--mono);font-size:1.7rem;font-weight:600;
  color:var(--gold);font-variant-numeric:tabular-nums}
.kpi-sub{font-family:var(--mono);font-size:.72rem;color:var(--dim);margin-top:.2rem}

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
  font-family:var(--mono);font-size:.68rem;font-weight:600;margin:2px}
.badge-ok{background:rgba(34,165,103,.15);color:#22A567;
  border:1px solid rgba(34,165,103,.3)}
.badge-ko{background:rgba(229,72,77,.15);color:#E5484D;
  border:1px solid rgba(229,72,77,.3)}
.badge-ai{background:rgba(124,58,237,.18);color:#A78BFA;
  border:1px solid rgba(124,58,237,.3)}
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
.stDataFrame{font-family:var(--mono)!important}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load_var():      return pd.read_csv(f"{PROC}/var_results.csv")
@st.cache_data
def load_stress():    return pd.read_csv(f"{PROC}/stress_results.csv")
@st.cache_data
def load_historical(): return pd.read_csv(f"{PROC}/historical_scenarios.csv")
@st.cache_data
def load_kupiec():    return pd.read_csv(f"{PROC}/backtest_kupiec.csv")


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


def get_recommended_method(kupiec_df, currency, confidence):
    sub = kupiec_df[
        (kupiec_df["currency"]==f"TND/{currency}") &
        (kupiec_df["confidence"]==confidence) &
        (kupiec_df["modele_valide"]==True)
    ]
    if sub.empty:
        return None
    if "Historique" in sub["method"].values:
        return "Historique"
    return sub["method"].iloc[0]


# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="ptitle">Risque de Change</div>',
           unsafe_allow_html=True)
st.caption("VaR · CVaR · Stress testing · Backtesting Kupiec — "
          "aligné Bâle III")

var_df = load_var()
stress_df = load_stress()
hist_df = load_historical()
kupiec_df = load_kupiec()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("⚙ Configuration")
profile_name = st.sidebar.selectbox("Profil client", list(CLIENT_PROFILES.keys()))
confidence = st.sidebar.selectbox("Niveau de confiance", [0.95, 0.99],
                                  format_func=lambda x: f"{x*100:.0f}%")
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

groq_key = os.environ.get("GROQ_API_KEY","")
ai_avail_risk = bool(groq_key)
st.sidebar.markdown("---")
use_ai_risk = (st.sidebar.checkbox("🤖 Interprétation Groq", value=False,
                                   key="ai_risk")
              if ai_avail_risk else False)
st.sidebar.caption("✅ Clé Groq détectée" if ai_avail_risk
                   else "⚙ Mode offline")

is_multi = profile["currency"] == "MULTI"

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 VaR & CVaR", "⚠ Stress Testing", "📉 Scénarios Historiques",
     "🔬 Backtesting Kupiec"])

# ═══════════════════════════════════════════════════════════
#  TAB 1 — VaR & CVaR
# ═══════════════════════════════════════════════════════════
with tab1:
    if is_multi:
        st.markdown(f"### VaR Portefeuille Multi-Devises — {profile_name}")
        sub = var_df[(var_df["profile"]==profile_name) &
                    (var_df["confidence"]==confidence)]
        if not sub.empty:
            row = sub.iloc[0]
            kc = st.columns(4)
            with kc[0]:
                st.markdown(f"""<div class="kpi-card">
                  <div class="kpi-lbl">VaR EUR</div>
                  <div class="kpi-val">{row['VaR_EUR']:,.0f}</div>
                  <div class="kpi-sub">TND</div></div>""",
                  unsafe_allow_html=True)
            with kc[1]:
                st.markdown(f"""<div class="kpi-card">
                  <div class="kpi-lbl">VaR USD</div>
                  <div class="kpi-val">{row['VaR_USD']:,.0f}</div>
                  <div class="kpi-sub">TND</div></div>""",
                  unsafe_allow_html=True)
            with kc[2]:
                st.markdown(f"""<div class="kpi-card">
                  <div class="kpi-lbl">VaR Diversifiée</div>
                  <div class="kpi-val">{row['VaR_diversifie']:,.0f}</div>
                  <div class="kpi-sub">TND</div></div>""",
                  unsafe_allow_html=True)
            with kc[3]:
                st.markdown(f"""<div class="kpi-card">
                  <div class="kpi-lbl">Bénéfice Diversif.</div>
                  <div class="kpi-val">{row['benefice_pct']:.1f}%</div>
                  <div class="kpi-sub">vs somme simple</div></div>""",
                  unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["VaR EUR seule","VaR USD seule",
                   "Somme non-diversifiée","VaR diversifiée"],
                y=[row["VaR_EUR"], row["VaR_USD"],
                   row["VaR_non_diversifie"], row["VaR_diversifie"]],
                marker_color=[NAVY, ORANGE, RED, GREEN],
                text=[f"{v:,.0f}" for v in
                      [row["VaR_EUR"],row["VaR_USD"],
                       row["VaR_non_diversifie"],row["VaR_diversifie"]]],
                textposition="outside"
            ))
            layout = dark_layout(height=380, yaxis_title="VaR (TND)")
            fig.update_layout(**layout)
            axes(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f'<div class="box ok">🔗 Corrélation EUR/USD '
                       f'(252j) : <b>{row["correlation"]:.4f}</b>. '
                       f'La diversification réduit le risque de '
                       f'<b>{row["benefice_pct"]:.1f}%</b> '
                       f'({row["benefice_diversification"]:,.0f} TND '
                       f'économisés) par rapport à la somme simple '
                       f'des expositions individuelles.</div>',
                       unsafe_allow_html=True)
    else:
        sub = var_df[(var_df["profile"]==profile_name) &
                    (var_df["confidence"]==confidence)]
        recommended = get_recommended_method(
            kupiec_df, profile["currency"], confidence
        )

        if not sub.empty:
            rec_row = sub[sub["method"]==recommended]
            if rec_row.empty:
                rec_row = sub.iloc[[0]]
            r = rec_row.iloc[0]

            kc = st.columns(4)
            with kc[0]:
                st.markdown(f"""<div class="kpi-card">
                  <div class="kpi-lbl">VaR Recommandée</div>
                  <div class="kpi-val">{r['VaR_value']:,.0f}</div>
                  <div class="kpi-sub">TND · {r['method']}</div>
                  </div>""", unsafe_allow_html=True)
            with kc[1]:
                st.markdown(f"""<div class="kpi-card">
                  <div class="kpi-lbl">CVaR</div>
                  <div class="kpi-val">{r['CVaR_value']:,.0f}</div>
                  <div class="kpi-sub">TND · perte moyenne au-delà</div>
                  </div>""", unsafe_allow_html=True)
            with kc[2]:
                st.markdown(f"""<div class="kpi-card">
                  <div class="kpi-lbl">VaR %</div>
                  <div class="kpi-val">{r['VaR_pct']:.3f}%</div>
                  <div class="kpi-sub">de l'exposition</div></div>""",
                  unsafe_allow_html=True)
            with kc[3]:
                st.markdown(f"""<div class="kpi-card">
                  <div class="kpi-lbl">Exposition</div>
                  <div class="kpi-val">{profile['amount']:,.0f}</div>
                  <div class="kpi-sub">TND · {profile['currency']}</div>
                  </div>""", unsafe_allow_html=True)

            badge = ('<span class="badge badge-ai">🤖 Groq</span>'
                    if use_ai_risk and ai_avail_risk else
                    '<span class="badge badge-off">⚙ Offline</span>')

            interp_kwargs = dict(
                profile=profile_name, currency=profile["currency"],
                method=recommended or r["method"],
                var_value=r["VaR_value"], cvar_value=r["CVaR_value"],
                confidence=confidence, exposure=profile["amount"],
                kupiec_valid=bool(recommended)
            )
            if use_ai_risk and ai_avail_risk:
                with st.spinner("Analyse Groq..."):
                    interp = get_var_interpretation(use_ai=True, **interp_kwargs)
            else:
                interp = get_var_interpretation(use_ai=False, **interp_kwargs)

            box_cls = "ok" if recommended else "warn"
            st.markdown(f'<div class="box {box_cls}">{badge} {interp}</div>',
                       unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Comparaison des 4 méthodes**")
            fig = go.Figure()
            colors = {"Historique":GREEN, "Paramétrique (GARCH)":NAVY,
                     "Monte Carlo":ORANGE, "Monte Carlo Student-t":"#8B5CF6"}
            fig.add_trace(go.Bar(name="VaR", x=sub["method"],
                                 y=sub["VaR_value"],
                                 marker_color=[colors.get(m,DIM)
                                              for m in sub["method"]]))
            fig.add_trace(go.Bar(name="CVaR", x=sub["method"],
                                 y=sub["CVaR_value"],
                                 marker_color=[colors.get(m,DIM)
                                              for m in sub["method"]],
                                 marker_opacity=.5))
            layout = dark_layout(height=360, barmode="group",
                                yaxis_title="Valeur (TND)",
                                legend=dict(orientation="h",
                                          bgcolor="rgba(0,0,0,0)"))
            fig.update_layout(**layout)
            axes(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                sub[["method","VaR_pct","VaR_value","CVaR_pct",
                    "CVaR_value"]].round(4),
                use_container_width=True, hide_index=True
            )

# ═══════════════════════════════════════════════════════════
#  TAB 2 — Stress Testing
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"### Chocs synthétiques — {profile_name}")
    if is_multi:
        st.info("Stress testing synthétique non applicable au "
               "portefeuille multi-devises — voir Scénarios "
               "Historiques pour l'impact réel des chocs de marché.")
    else:
        sub = stress_df[stress_df["profile"]==profile_name]
        if not sub.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=sub["choc_pct"], y=sub["perte_estimee"],
                marker_color=[RED if i==2 else ORANGE if i==1 else GOLD
                             for i in range(len(sub))],
                text=[f"{v:,.0f} TND" for v in sub["perte_estimee"]],
                textposition="outside"
            ))
            layout = dark_layout(height=360,
                                yaxis_title="Perte estimée (TND)",
                                xaxis_title="Amplitude du choc")
            fig.update_layout(**layout)
            axes(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(sub, use_container_width=True, hide_index=True)
            st.markdown(f'<div class="box">📌 {sub.iloc[0]["sens_du_cours"].split(" de ")[0].capitalize()} '
                       f'défavorable pour ce profil. À 30%, la perte '
                       f'atteint <b>{sub.iloc[-1]["perte_estimee"]:,.0f} '
                       f'TND</b>, soit {sub.iloc[-1]["perte_pct_exposition"]:.0f}% '
                       f'de l\'exposition totale.</div>',
                       unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  TAB 3 — Scénarios Historiques
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"### Impact de chocs de marché réels — {profile_name}")
    if is_multi:
        st.info("Non disponible pour le portefeuille multi-devises.")
    else:
        sub = hist_df[hist_df["profile"]==profile_name]
        if not sub.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=sub["scenario"], y=sub["perte_estimee"],
                marker_color=[RED, ORANGE, "#8B5CF6"][:len(sub)],
                text=[f"{v:,.0f} TND" for v in sub["perte_estimee"]],
                textposition="outside"
            ))
            layout = dark_layout(height=360,
                                yaxis_title="Perte estimée (TND)")
            fig.update_layout(**layout)
            axes(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(sub, use_container_width=True, hide_index=True)

            worst = sub.loc[sub["perte_estimee"].idxmax()]
            st.markdown(f'<div class="box warn">⚠ Scénario le plus '
                       f'défavorable : <b>{worst["scenario"]}</b> '
                       f'({worst["periode"]}) — variation du cours de '
                       f'<b>{worst["variation_cours_pct"]:+.2f}%</b>, '
                       f'perte estimée <b>{worst["perte_estimee"]:,.0f} '
                       f'TND</b> ({worst["perte_pct_exposition"]:.2f}% '
                       f'de l\'exposition).</div>',
                       unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  TAB 4 — Backtesting Kupiec
# ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Validation statistique des méthodes de VaR")
    st.caption("Test de Kupiec (1995) sur la période out-of-sample "
              "2024-2026 — confirme si le taux réel de dépassement "
              "correspond au niveau de confiance théorique")

    badges_html = ""
    for _, row in kupiec_df.iterrows():
        cls = "badge-ok" if row["modele_valide"] else "badge-ko"
        icon = "✓" if row["modele_valide"] else "✗"
        badges_html += (f'<span class="badge {cls}">{icon} '
                       f'{row["method"]} · {row["currency"]} · '
                       f'{row["confidence"]*100:.0f}%</span>')
    st.markdown(badges_html, unsafe_allow_html=True)

    st.markdown("---")
    st.dataframe(
        kupiec_df[["method","currency","confidence","n_violations",
                  "violations_attendues","LR_stat",
                  "seuil_critique_5pct","modele_valide"]],
        use_container_width=True, hide_index=True
    )

    n_valid = kupiec_df["modele_valide"].sum()
    st.markdown(f'<div class="box">📐 <b>{n_valid}/{len(kupiec_df)}</b> '
               f'configurations validées. La VaR <b>Historique</b> est '
               f'la seule méthode validée sur les 4 combinaisons '
               f'devise × niveau de confiance — recommandée comme '
               f'méthode principale pour le pilotage du risque BIAT.'
               f'</div>', unsafe_allow_html=True)

st.divider()
st.caption("VaR · CVaR · Stress · Kupiec — Youssef Neji — MINDS ENIT 2026")