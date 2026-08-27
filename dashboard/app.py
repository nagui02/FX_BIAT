# =============================================================
#  dashboard/app.py
#  BIAT FX Intelligence — Page principale
#  Auteur : Youssef Neji | MINDS ENIT | 2026
# =============================================================

import streamlit as st
import pandas as pd
import json, os
from datetime import datetime

st.set_page_config(
    page_title="BIAT — FX Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIVE_DIR = os.path.join(BASE_DIR, "..", "data", "live")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  --bg:#0A0E17; --surface:#131A2A; --surface2:#1B2438;
  --navy:#003366; --navy-l:#0F5FA8;
  --gold:#C9A84C; --gold-b:#E8C875;
  --text:#EDF1F7; --dim:#8B96AC;
  --green:#22A567; --red:#E5484D;
  --border:rgba(255,255,255,.07);
  --mono:'IBM Plex Mono',monospace;
  --sans:'Inter',sans-serif;
  --disp:'Sora',sans-serif;
}
html,body,.stApp{background:var(--bg)!important;color:var(--text)!important;
  font-family:var(--sans)}
.stApp>header{background:var(--bg)!important}
section[data-testid="stSidebar"]{background:#070A11!important}
section[data-testid="stSidebar"] *{color:var(--text)!important}
h1,h2,h3,h4,h5{font-family:var(--disp)!important;color:var(--text)!important}
p,li,span,label{font-family:var(--sans)}

.ticker{
  display:flex;align-items:center;gap:0;
  background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:.7rem 0;margin-bottom:1.6rem;
  overflow-x:auto;box-shadow:0 4px 20px rgba(0,0,0,.35)
}
.ticker-item{
  flex:1;min-width:120px;text-align:center;
  padding:0 1.1rem;border-right:1px solid var(--border);
  white-space:nowrap
}
.ticker-item:last-child{border-right:none}
.tick-lbl{font-size:.62rem;color:var(--dim);text-transform:uppercase;
  letter-spacing:.09em;font-family:var(--sans)}
.tick-val{font-family:var(--mono);font-size:1.15rem;font-weight:600;
  color:var(--gold);font-variant-numeric:tabular-nums}
.tick-chg-up{color:var(--red);font-family:var(--mono);font-size:.7rem}
.tick-chg-dn{color:var(--green);font-family:var(--mono);font-size:.7rem}
.pulse{width:6px;height:6px;border-radius:50%;background:var(--green);
  display:inline-block;margin-right:6px;animation:pulse 1.6s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

.hero{
  background:linear-gradient(135deg,#001327 0%,#00284D 55%,#003A70 100%);
  border-radius:18px;padding:2.6rem 3rem;margin-bottom:1.8rem;
  border:1px solid rgba(201,168,76,.2);
  box-shadow:0 12px 40px rgba(0,0,0,.5);
  position:relative;overflow:hidden
}
.hero::after{
  content:'';position:absolute;top:-40%;right:-8%;width:340px;height:340px;
  background:radial-gradient(circle,rgba(201,168,76,.14),transparent 70%);
  border-radius:50%
}
.hero-wordmark{
  font-family:var(--disp);font-weight:800;font-size:1.9rem;
  background:var(--gold);color:#001327;display:inline-block;
  padding:.3rem 1rem;border-radius:8px;letter-spacing:.02em
}
.hero-title{font-family:var(--disp);font-weight:700;font-size:2.1rem;
  color:#fff;margin:1rem 0 .3rem;position:relative;z-index:1}
.hero-sub{color:rgba(255,255,255,.65);font-size:.95rem;
  position:relative;z-index:1;max-width:620px}
.hero-meta{color:rgba(255,255,255,.35);font-size:.74rem;margin-top:.6rem;
  font-family:var(--mono)}

.sec-label{
  font-family:var(--disp);font-weight:600;font-size:1.05rem;
  color:var(--text);margin:.4rem 0 1rem;
  display:flex;align-items:center;gap:.6rem
}
.sec-label::before{content:'';width:3px;height:1.1rem;background:var(--gold);
  border-radius:2px;display:inline-block}

.tile{
  background:var(--surface);border:1px solid var(--border);
  border-radius:14px;padding:1.5rem 1.2rem;height:100%;
  transition:all .2s ease;position:relative
}
.tile:hover{border-color:rgba(201,168,76,.35);
  transform:translateY(-3px);box-shadow:0 14px 32px rgba(0,0,0,.4)}
.tile-icon{font-size:1.9rem;margin-bottom:.7rem}
.tile-title{font-family:var(--disp);font-weight:700;font-size:1rem;
  color:var(--gold);margin-bottom:.15rem}
.tile-sub{font-size:.68rem;color:var(--dim);margin-bottom:.6rem;
  font-family:var(--mono);text-transform:uppercase;letter-spacing:.06em}
.tile-desc{font-size:.8rem;color:var(--dim);line-height:1.5;margin-bottom:.7rem}
.tile-tag{display:inline-block;font-family:var(--mono);font-size:.66rem;
  padding:3px 9px;border-radius:20px;background:rgba(0,51,102,.4);
  color:var(--gold);border:1px solid rgba(201,168,76,.18)}
.tile-lock{font-size:.68rem;color:var(--dim);margin-top:.5rem;
  font-style:italic}

.hr{height:1px;margin:1.8rem 0;
  background:linear-gradient(90deg,transparent,var(--gold),transparent);
  opacity:.5}

.info-card{background:var(--surface);border:1px solid var(--border);
  border-radius:12px;padding:1.3rem;height:100%}
.info-card h4{font-family:var(--disp);font-size:.9rem;color:var(--gold);
  margin:0 0 .8rem;text-transform:uppercase;letter-spacing:.05em}
.info-card p,.info-card li{color:var(--dim);font-size:.83rem;line-height:1.6}
.info-card table{width:100%;font-size:.78rem;font-family:var(--mono)}
.info-card td{color:var(--text);padding:2px 0}
.aug-status{display:flex;justify-content:space-between;
  font-size:.78rem;padding:.3rem 0;border-bottom:1px solid var(--border)}
.aug-status:last-child{border-bottom:none}
.aug-status b{font-family:var(--mono);color:var(--gold)}

.stButton>button{
  background:var(--surface2)!important;color:var(--text)!important;
  border:1px solid rgba(201,168,76,.25)!important;border-radius:8px!important;
  font-family:var(--sans)!important;font-weight:600!important;
  font-size:.82rem!important;transition:all .2s!important
}
.stButton>button:hover{
  background:var(--gold)!important;color:#001327!important;
  border-color:var(--gold)!important
}
[data-testid="stMetricValue"]{font-family:var(--mono)!important;
  color:var(--gold)!important}
[data-testid="stMetricLabel"]{color:var(--dim)!important;
  font-family:var(--sans)!important}
.stAlert,.stInfo,.stWarning{background:var(--surface)!important;
  color:var(--text)!important;border-color:var(--border)!important}

.footer{text-align:center;color:var(--dim);font-size:.72rem;
  font-family:var(--mono);padding:1.6rem 0;
  border-top:1px solid var(--border);margin-top:2rem}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_live():
    p = os.path.join(LIVE_DIR, "latest_rates.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=300)
def load_aug():
    p = os.path.join(LIVE_DIR, "predictions_aout2026.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}

live, aug = load_live(), load_aug()

st.markdown(f"""
<div class="hero">
  <span class="hero-wordmark">BIAT</span>
  <div class="hero-title">FX Risk Intelligence</div>
  <div class="hero-sub">Prédiction quantitative des cours de change
    et gestion du risque — TND/USD · TND/EUR · pipeline 2004–2026,
    validé statistiquement, testé en conditions réelles.</div>
  <div class="hero-meta">YOUSSEF NEJI · MINDS ENIT · STAGE INGÉNIEUR 2026</div>
</div>
""", unsafe_allow_html=True)

if live:
    r, m = live.get("rates", {}), live.get("macro", {})
    usd, eur = r.get("TND_USD", {}), r.get("TND_EUR", {})
    upd = live.get("last_updated", "")[:16].replace("T", " ")

    def chg(v):
        if v is None:
            return ""
        cls = "tick-chg-up" if v > 0 else "tick-chg-dn"
        sym = "▲" if v > 0 else "▼"
        return f'<div class="{cls}">{sym} {abs(v):.4f}</div>'

    items = [
        ("TND/USD", f"{usd.get('value',0):.4f}", chg(usd.get("change_1d"))),
        ("TND/EUR", f"{eur.get('value',0):.4f}", chg(eur.get("change_1d"))),
        ("BRENT",   f"{m.get('Brent_USD','—')}", ""),
        ("VIX",     f"{m.get('VIX','—')}", ""),
        ("EUR/USD", f"{m.get('EURUSD','—')}", ""),
        ("FED",     f"{m.get('TauxFed','—')}%", ""),
    ]
    html = '<div class="ticker">'
    for lbl, val, c in items:
        html += (f'<div class="ticker-item"><div class="tick-lbl">'
                 f'{lbl}</div><div class="tick-val">{val}</div>{c}</div>')
    html += (f'<div class="ticker-item"><div class="tick-lbl">'
             f'<span class="pulse"></span>LIVE</div>'
             f'<div style="font-family:var(--mono);font-size:.78rem;'
             f'color:var(--dim)">{upd}</div></div></div>')
    st.markdown(html, unsafe_allow_html=True)
else:
    st.warning("Données live indisponibles — `python src/update_data.py`")

st.markdown('<div class="sec-label">Modules</div>', unsafe_allow_html=True)

TILES = [
    {"icon":"📊","title":"Données","en":"Market Data","color":"",
     "desc":"Cours historiques, variables macro, corrélations "
            "dynamiques, distribution des rendements.",
     "tag":"2004–2026 live","page":"pages/1_Données.py","ready":True},
    {"icon":"📈","title":"Prévisions","en":"Forecasting","color":"gold",
     "desc":"ARIMAX/ARIMA/LSTM/MLP, sélection auto du meilleur "
            "modèle, prédiction instantanée à toute date.",
     "tag":"4 modèles · instant","page":"pages/2_Prévisions.py",
     "ready":True},
    {"icon":"⚠️","title":"Risque","en":"FX Risk","color":"red",
     "desc":"VaR historique/paramétrique/Monte Carlo, CVaR, "
            "stress testing sur profils clients BIAT.",
     "tag":"VaR · CVaR","page":"pages/3_Risque.py","ready":True},
    {"icon":"🛡️","title":"Couverture","en":"Hedging","color":"green",
     "desc":"Forward vs Options Garman-Kohlhagen, coût et P&L "
            "simulé par profil client.",
     "tag":"Forward · GK","page":"pages/4_Couverture.py",
     "ready":True},
    {"icon":"📐","title":"Portefeuille","en":"Portfolio","color":"",
     "desc":"Optimisation Markowitz EUR+USD, frontière "
            "efficiente, ratio de Sharpe.",
     "tag":"Markowitz","page":"pages/5_Markowitz.py","ready":True},
]

cols = st.columns(5)
for col, t in zip(cols, TILES):
    with col:
        lock = ("" if t["ready"] else
                '<div class="tile-lock">🔒 En développement</div>')
        st.markdown(f"""
        <div class="tile">
          <div class="tile-icon">{t['icon']}</div>
          <div class="tile-title">{t['title']}</div>
          <div class="tile-sub">{t['en']}</div>
          <div class="tile-desc">{t['desc']}</div>
          <div class="tile-tag">{t['tag']}</div>
          {lock}
        </div>
        """, unsafe_allow_html=True)
        if st.button("Ouvrir →" if t["ready"] else "Bientôt",
                     key=t["title"], use_container_width=True,
                     disabled=not t["ready"]):
            st.switch_page(t["page"])

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
    <h4>Projet</h4>
    <p>Stage ingénieur 2 mois · <b style="color:var(--text)">BIAT</b>
    (Banque Internationale Arabe de Tunisie). Outil quantitatif de
    prévision et gestion du risque de change pour le desk FX.</p>
    <p><b style="color:var(--text)">Encadrant :</b> Mr. Nader
    Trigui — BIAT</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
    <h4>Stack technique</h4>
    <table>
    <tr><td>Prévision</td><td style="text-align:right">
        ARIMA · ARIMAX · LSTM · MLP</td></tr>
    <tr><td>Volatilité</td><td style="text-align:right">
        GARCH(1,1)</td></tr>
    <tr><td>Risque</td><td style="text-align:right">
        VaR · CVaR · Kupiec</td></tr>
    <tr><td>Couverture</td><td style="text-align:right">
        Forward · Garman-Kohlhagen</td></tr>
    <tr><td>Données</td><td style="text-align:right">
        BCT · FRED · INS</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

with c3:
    if aug:
        fp = aug.get("forecast_period", {})
        gen = aug.get("generated_on", "")[:10]
        n_wait = sum(
            1 for v in aug.get("forecasts", {})
              .get("USD", {}).get("ARIMA", {})
              .get("J1_daily", {}).values()
            if v.get("actual") is None
        )
        n_total = fp.get("n_days", 0)
        st.markdown(f"""
        <div class="info-card">
        <h4>Validation Août 2026</h4>
        <div class="aug-status"><span>Généré le</span>
            <b>{gen}</b></div>
        <div class="aug-status"><span>Période</span>
            <b>{fp.get('start','')} → {fp.get('end','')}</b></div>
        <div class="aug-status"><span>Jours ouvrés</span>
            <b>{n_total}</b></div>
        <div class="aug-status"><span>En attente BCT</span>
            <b>{n_wait}</b></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-card">
        <h4>Validation Août 2026</h4>
        <p>Non générée — <code>python -m src.predict_august</code></p>
        </div>
        """, unsafe_allow_html=True)

st.markdown(
    f'<div class="footer">BIAT · Stage Ingénieur MINDS ENIT 2026 · '
    f'Youssef Neji · {datetime.now().strftime("%d/%m/%Y %H:%M")}</div>',
    unsafe_allow_html=True
)