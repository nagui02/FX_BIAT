# =============================================================
#  dashboard/pages/2_Prévisions.py
#  BIAT FX Intelligence — Prévisions
#  Auteur : Youssef Neji | MINDS ENIT | 2026
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, os, sys

st.set_page_config(page_title="Prévisions — BIAT FX",
                   page_icon="📈", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE_DIR, "..", "data", "processed")
LIVE = os.path.join(BASE_DIR, "..", "data", "live")
ROOT = os.path.join(BASE_DIR, "..")
sys.path.insert(0, ROOT)

try:
    from utils.ai_interpret import get_interpretation, hedging_recommendation
    AI_MODULE = True
except ImportError:
    AI_MODULE = False

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

.pred-card{background:linear-gradient(160deg,#0A1A33,#0F2542);
  border-radius:14px;padding:1.3rem;text-align:center;
  border:1px solid rgba(201,168,76,.25);box-shadow:0 8px 24px rgba(0,0,0,.4)}
.pred-card.sel{border-color:var(--gold);border-width:2px;
  box-shadow:0 0 20px rgba(201,168,76,.2)}
.pred-h{font-family:var(--mono);font-size:.64rem;color:rgba(255,255,255,.5);
  text-transform:uppercase;letter-spacing:.08em}
.pred-m{font-family:var(--sans);font-size:.72rem;color:var(--gold);
  margin:.3rem 0;font-weight:600}
.pred-v{font-family:var(--mono);font-size:1.9rem;font-weight:600;
  color:var(--gold);font-variant-numeric:tabular-nums}
.pred-up{color:#E5484D;font-family:var(--mono);font-size:.82rem}
.pred-dn{color:#22A567;font-family:var(--mono);font-size:.82rem}

.instant-card{background:linear-gradient(135deg,#001327,#003A70);
  border-radius:14px;padding:1.6rem;text-align:center;
  border:1px solid rgba(201,168,76,.3)}
.instant-v{font-family:var(--mono);font-size:2.3rem;font-weight:600;
  color:var(--gold)}
.instant-lbl{font-family:var(--sans);font-size:.75rem;
  color:rgba(255,255,255,.6);text-transform:uppercase;letter-spacing:.06em}

.box{background:var(--surface);border-left:3px solid var(--gold);
  border-radius:0 10px 10px 0;padding:.8rem 1.1rem;font-size:.85rem;
  color:var(--text);margin-top:.5rem;line-height:1.55}
.box b{color:var(--gold);font-family:var(--mono)}
.rec-box{background:var(--surface);border-radius:12px;padding:1.1rem;
  border:1px solid rgba(201,168,76,.2);margin-top:.8rem}
.rec-title{font-family:var(--disp);font-weight:700;font-size:.9rem;
  margin-bottom:.4rem}
.m-chip{display:inline-block;background:var(--surface2);
  border:1px solid var(--border);border-radius:7px;padding:.3rem .7rem;
  font-family:var(--mono);font-size:.72rem;color:var(--text);margin:.2rem .15rem}
.ai-badge{display:inline-block;background:linear-gradient(135deg,#4C1D95,#7C3AED);
  color:#fff;padding:2px 9px;border-radius:20px;font-family:var(--mono);
  font-size:.65rem;font-weight:600}
.offline-badge{display:inline-block;background:rgba(139,150,172,.18);
  color:var(--dim);border:1px solid rgba(139,150,172,.28);padding:2px 9px;
  border-radius:20px;font-family:var(--mono);font-size:.65rem}

[data-testid="stMetricValue"]{font-family:var(--mono)!important;
  color:var(--gold)!important}
[data-testid="stMetricLabel"]{color:var(--dim)!important}
.stSelectbox>div>div,.stDateInput>div>div{background:var(--surface2)!important;
  color:var(--text)!important;border-color:var(--border)!important}
.stTabs [data-baseweb="tab"]{color:var(--dim)!important}
.stTabs [data-baseweb="tab"][aria-selected="true"]{color:var(--gold)!important;
  border-bottom-color:var(--gold)!important}
.stButton>button{background:var(--surface2)!important;color:var(--text)!important;
  border:1px solid rgba(201,168,76,.25)!important;border-radius:7px!important;
  font-weight:600!important;font-size:.78rem!important}
.stButton>button:hover{background:var(--gold)!important;color:#001327!important}
.stAlert,.stInfo{background:var(--surface)!important;color:var(--text)!important}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PERIODS = {"1S":"7D","1M":"30D","3M":"90D","6M":"180D","1A":"365D","MAX":None}

# ── Data loaders (fichiers consolidés) ─────────────────────────
@st.cache_data
def load_predictions(cur: str) -> pd.DataFrame:
    p = os.path.join(PROC, f"predictions_{cur.lower()}.csv")
    if os.path.exists(p):
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df.index.name = "Date"
        return df
    return pd.DataFrame()

@st.cache_data
def load_sigma(cur: str):
    p = os.path.join(PROC, f"garch_sigma_{cur.lower()}.csv")
    if os.path.exists(p):
        return pd.read_csv(p, index_col=0, parse_dates=True).squeeze()
    return None

@st.cache_data
def load_metrics() -> pd.DataFrame:
    p = os.path.join(PROC, "metrics_all.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data
def load_dm() -> pd.DataFrame:
    p = os.path.join(PROC, "diebold_mariano_results.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data
def load_dataset():
    p = os.path.join(PROC, "dataset_final.csv")
    return pd.read_csv(p, parse_dates=["Date"], index_col="Date")


def get_best_model(metrics_df, currency, horizon):
    if metrics_df.empty:
        return "ARIMA"
    sub = metrics_df[
        (metrics_df["Devise"] == f"TND/{currency}") &
        (metrics_df["Horizon"] == f"J+{horizon}")
    ].dropna(subset=["RMSE"])
    if sub.empty:
        return "ARIMA"
    return sub.loc[sub["RMSE"].idxmin(), "Modèle"]


def dir_acc(actual_s, pred_s):
    if actual_s is None or pred_s is None:
        return None
    c = actual_s.index.intersection(pred_s.index)
    if len(c) < 2:
        return None
    a, p = actual_s.loc[c].values, pred_s.loc[c].values
    d_a, d_p = np.sign(np.diff(a)), np.sign(np.diff(p))
    m = d_a != 0
    return (float(np.mean(d_a[m] == d_p[m]) * 100)
            if m.sum() > 0 else None)


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

def period_bar(key, src):
    if src is None or len(src) == 0:
        return src
    sel = st.session_state.get(f"p_{key}","MAX")
    cols = st.columns(len(PERIODS))
    for col,(lbl,_) in zip(cols, PERIODS.items()):
        if col.button(lbl, key=f"{key}_{lbl}", use_container_width=True):
            sel = lbl
            st.session_state[f"p_{key}"] = lbl
            st.rerun()
    offset = PERIODS[sel]
    return src if offset is None else src.loc[src.index[-1]-pd.Timedelta(offset):]


# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="ptitle">Prévisions des Cours de Change</div>',
           unsafe_allow_html=True)
st.caption("Sélection automatique du meilleur modèle par tests de "
          "Diebold-Mariano · prédiction instantanée à toute date")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("⚙ Configuration")
currency = st.sidebar.selectbox("Devise", ["USD","EUR"],
                                format_func=lambda x: f"TND/{x}")
horizon = st.sidebar.selectbox("Horizon", [1,7,30],
    format_func=lambda h: {1:"J+1 — Demain", 7:"J+7 — Semaine",
                           30:"J+30 — Mois"}[h])

metrics_df = load_metrics()
auto_best  = get_best_model(metrics_df, currency, horizon)
st.sidebar.markdown(f"**Recommandé :** `{auto_best}`")

override = st.sidebar.checkbox("Override modèle", value=False)
model_choice = (
    st.sidebar.selectbox("Modèle", ["ARIMA","LSTM","MLP","Naïf"],
        index=(["ARIMA","LSTM","MLP","Naïf"].index(auto_best)
               if auto_best in ["ARIMA","LSTM","MLP","Naïf"] else 0))
    if override else auto_best
)

show_lstm  = st.sidebar.checkbox("LSTM", True)
show_mlp   = st.sidebar.checkbox("MLP", True)
show_naive = st.sidebar.checkbox("Naïf", True)
show_ci    = st.sidebar.checkbox("Bande GARCH ±2σ", True)

groq_key = os.environ.get("GROQ_API_KEY","")
ai_avail = bool(groq_key) and AI_MODULE
st.sidebar.markdown("---")
use_ai = (st.sidebar.checkbox("🤖 Interprétation Groq", value=False)
          if ai_avail else False)
st.sidebar.caption("✅ Clé Groq détectée" if ai_avail
                   else "⚙ Mode offline")

# ── Load ──────────────────────────────────────────────────────
preds     = load_predictions(currency)
df_all    = load_dataset()
dm_df     = load_dm()
sigma     = load_sigma(currency)
col_spot  = f"TND_{currency}"

actual = pd.Series(dtype=float)
if not preds.empty and "Actual" in preds.columns:
    actual = preds["Actual"].dropna()
elif col_spot in df_all.columns:
    actual = df_all[col_spot].dropna()

current_rate = float(actual.iloc[-1]) if len(actual) > 0 else 0.0
garch_vol = (float(sigma.iloc[-1])*np.sqrt(252)*100
            if sigma is not None and len(sigma)>0 else 0.0)

live_path = os.path.join(LIVE,"latest_rates.json")
live_data = {}
if os.path.exists(live_path):
    with open(live_path) as f:
        live_data = json.load(f)
macro   = live_data.get("macro",{})
vix_v   = float(macro.get("VIX",18))
brent_v = float(macro.get("Brent_USD",80))
fed_v   = float(macro.get("TauxFed",5))
bct_v   = 8.0

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Prévision", "📊 Comparaison", "🔬 Tests DM", "📉 GARCH"])

# ═══════════════════════════════════════════════════════════
#  TAB 1
# ═══════════════════════════════════════════════════════════
with tab1:
    def get_pred(model, h):
        col = f"{'Naif' if model=='Naïf' else model}_J{h}"
        if preds.empty or col not in preds.columns:
            return None
        s = preds[col].dropna()
        return float(s.iloc[-1]) if len(s) > 0 else None

    all_preds = {}
    for h in [1,7,30]:
        best = get_best_model(metrics_df, currency, h)
        v = get_pred(best, h)
        if v: all_preds[h] = (best, v)

    if all_preds:
        cols_p = st.columns(len(all_preds))
        h_lbl = {1:"J+1 · Demain", 7:"J+7 · Semaine", 30:"J+30 · Mois"}
        for col,(h,(mod,val)) in zip(cols_p, all_preds.items()):
            chg = (val-current_rate)/current_rate*100 if current_rate else 0
            cls = "pred-up" if chg>0 else "pred-dn"
            sym = "▲" if chg>0 else "▼"
            sel = "sel" if h==horizon else ""
            with col:
                st.markdown(f"""
<div class="pred-card {sel}">
  <div class="pred-h">{h_lbl[h]}</div>
  <div class="pred-m">{mod}</div>
  <div class="pred-v">{val:.4f}</div>
  <div class="{cls}">{sym} {abs(chg):.3f}%</div>
</div>""", unsafe_allow_html=True)
        st.caption(f"Référence TND/{currency} : "
                  f"**{current_rate:.4f}** (dernière obs.)")

    st.markdown("---")
    st.markdown(f"**Graphique — TND/{currency} J+{horizon} · "
              f"{model_choice}"
              + (" (auto)" if not override else " (manuel)") + "**")

    df_plt = period_bar("prev", actual.to_frame("v")
                        if len(actual)>0 else pd.DataFrame())
    act_plt = (actual.loc[df_plt.index]
              if df_plt is not None and len(df_plt)>0 and len(actual)>0
              else actual)

    fig = go.Figure()
    if show_ci and sigma is not None and len(act_plt)>0:
        c = act_plt.index.intersection(sigma.index)
        if len(c)>0:
            s_c, a_c = sigma.loc[c], act_plt.loc[c]
            h_f = np.sqrt(horizon)
            fig.add_trace(go.Scatter(
                x=c.tolist()+c.tolist()[::-1],
                y=(a_c+2*s_c*h_f).tolist()+(a_c-2*s_c*h_f).tolist()[::-1],
                fill="toself", fillcolor="rgba(139,150,172,.1)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"GARCH ±2σ·√{horizon}"))

    if len(act_plt)>0:
        fig.add_trace(go.Scatter(x=act_plt.index, y=act_plt.values,
                                 name="Réel", line=dict(color="white",width=1.8)))

    for cname,lbl,color,dash in [
        (f"Naif_J{horizon}","Naïf",DIM,"dot"),
        (f"ARIMA_J{horizon}","ARIMA",ORANGE,"dash"),
        (f"LSTM_J{horizon}","LSTM",GREEN,"dashdot"),
        (f"MLP_J{horizon}","MLP","#8B5CF6","dot"),
    ]:
        if lbl=="Naïf" and not show_naive: continue
        if lbl=="LSTM" and not show_lstm: continue
        if lbl=="MLP" and not show_mlp: continue
        if preds.empty or cname not in preds.columns: continue
        s = preds[cname].dropna()
        if len(act_plt)>0:
            s = s.loc[s.index>=act_plt.index[0]]
        w = 1.8 if lbl==model_choice else 1.0
        fig.add_trace(go.Scatter(x=s.index, y=s.values,
                                 name=(f"★ {lbl}" if lbl==model_choice else lbl),
                                 line=dict(color=color,width=w,dash=dash)))

    layout = dark_layout()
    layout.update(height=390, hovermode="x unified",
                  xaxis_title="Date", yaxis_title=f"TND/{currency}",
                  legend=dict(orientation="h",bgcolor="rgba(0,0,0,0)",
                             y=1.06,x=1,xanchor="right"))
    fig.update_layout(**layout)
    axes(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ── Interprétation ─────────────────────────────────────
    pred_val = get_pred(model_choice, horizon)
    if pred_val:
        st.markdown("---")
        col_i, col_m = st.columns([3,1])
        with col_m:
            st.metric("VIX", f"{vix_v:.1f}",
                     "⚠ Stress" if vix_v>20 else "✅ Stable")
            st.metric("Brent", f"${brent_v:.1f}",
                     "⚠ Élevé" if brent_v>90 else "✅ Modéré")
            st.metric("GARCH σ", f"{garch_vol:.2f}%/an")
        with col_i:
            badge = ('<span class="ai-badge">🤖 Groq</span>'
                    if use_ai and ai_avail else
                    '<span class="offline-badge">⚙ Offline</span>')
            kwargs = dict(pred=pred_val, current=current_rate,
                          horizon=horizon, currency=currency,
                          model=model_choice, garch_vol=garch_vol,
                          vix=vix_v, brent=brent_v, taux_fed=fed_v,
                          taux_bct=bct_v)
            if use_ai and ai_avail:
                with st.spinner("Analyse Groq..."):
                    interp = get_interpretation(use_ai=True, **kwargs)
            else:
                interp = get_interpretation(use_ai=False, **kwargs)
            st.markdown(f'<div class="box">{badge} {interp}</div>',
                       unsafe_allow_html=True)

            chg_p = ((pred_val-current_rate)/current_rate*100
                    if current_rate else 0)
            rec = hedging_recommendation(chg_p, horizon, currency)
            st.markdown(f'<div class="rec-box">'
                       f'<div class="rec-title" style="color:{rec["color"]}">'
                       f'{rec["icon"]} {rec["title"]}</div>{rec["text"]}</div>',
                       unsafe_allow_html=True)

    st.markdown("---")

    # ── Prédiction instantanée (nouveau, via inference.py) ──
    st.markdown("#### ⚡ Prédiction instantanée — n'importe quelle date")
    st.caption("Modèles pré-entraînés — réponse en moins d'une seconde")

    c_d, c_c, c_b = st.columns([2,1,1])
    with c_d:
        target_dt = st.date_input("Date cible",
            value=pd.Timestamp.now().date() + pd.Timedelta(days=14))
    with c_c:
        cur_inst = st.selectbox("Devise", ["USD","EUR"],
                                format_func=lambda x: f"TND/{x}", key="inst")
    with c_b:
        st.markdown("<br>", unsafe_allow_html=True)
        inst_btn = st.button("⚡ Prédire", use_container_width=True,
                             type="primary")

    if inst_btn:
        try:
            from src.inference import predict_for_date
            res = predict_for_date(cur_inst, str(target_dt))
            if "error" in res:
                st.error(res["error"])
            else:
                if res.get("warning"):
                    st.warning(res["warning"])
                st.markdown(f"""
<div class="instant-card">
  <div class="instant-lbl">{target_dt} · {res['n_steps']}j · {res['model']}</div>
  <div class="instant-v">{res['value']:.4f}</div>
</div>""", unsafe_allow_html=True)
                st.caption(f"Référence : {res['last_known_date']} = "
                          f"{res['last_known_value']:.4f}")
        except ImportError:
            st.error("src/inference.py introuvable — lancez d'abord "
                    "`python -m src.train_final_models`")

    # ── Test Août 2026 ─────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📅 Test en conditions réelles — Août 2026")
    aug_p = os.path.join(LIVE, "predictions_aout2026.json")
    if os.path.exists(aug_p):
        with open(aug_p) as f:
            aug = json.load(f)
        fp, gen = aug.get("forecast_period", {}), aug.get("generated_on","")[:10]
        st.info(f"Généré le **{gen}** · Retrain : "
               f"**{aug.get('retrain_period','')}** · "
               f"{fp.get('n_days','')} jours ouvrés")

        aug_cur = st.selectbox("Devise", ["USD","EUR"],
                               format_func=lambda x: f"TND/{x}",
                               key="aug_cur")
        forecasts = aug.get("forecasts",{}).get(aug_cur, {})
        if forecasts:
            all_dates = fp.get("days", [])
            table_rows = []
            for d_str in all_dates:
                row = {"Date": d_str}
                actual_val = None
                for mod, res in forecasts.items():
                    if isinstance(res, dict):
                        entry = res.get("J1_daily",{}).get(d_str,{})
                        p, a = entry.get("prediction"), entry.get("actual")
                        row[mod] = f"{p:.5f}" if p is not None else "—"
                        if a is not None: actual_val = a
                row["Réel BCT"] = (f"{actual_val:.5f}"
                                   if actual_val is not None else "⏳")
                table_rows.append(row)
            if table_rows:
                st.dataframe(pd.DataFrame(table_rows),
                            use_container_width=True, hide_index=True)

            errors = []
            for mod, res in forecasts.items():
                if not isinstance(res, dict): continue
                j1d = res.get("J1_daily",{})
                pv, av = [], []
                for d in all_dates:
                    e = j1d.get(d,{})
                    if e.get("prediction") is not None and e.get("actual") is not None:
                        pv.append(e["prediction"]); av.append(e["actual"])
                if pv and av:
                    p_a, a_a = np.array(pv), np.array(av)
                    rmse = float(np.sqrt(np.mean((p_a-a_a)**2)))
                    errors.append({"Modèle": mod, "N": len(pv),
                                  "RMSE": f"{rmse:.5f}"})
            if errors:
                st.markdown("**✅ Validation en temps réel**")
                st.dataframe(pd.DataFrame(errors).sort_values("RMSE"),
                            use_container_width=True, hide_index=True)
    else:
        st.warning("`python -m src.predict_august`")

# ═══════════════════════════════════════════════════════════
#  TAB 2 — Comparaison
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"### Comparaison — TND/{currency}")
    models = ["Naïf","ARIMA","LSTM","MLP"]
    cmap = {"Naïf":DIM,"ARIMA":ORANGE,"LSTM":GREEN,"MLP":"#8B5CF6"}
    sub = metrics_df[
        (metrics_df["Devise"]==f"TND/{currency}") &
        (metrics_df["Horizon"]==f"J+{horizon}")
    ] if not metrics_df.empty else pd.DataFrame()

    da = {}
    for mod in models:
        cname = f"{'Naif' if mod=='Naïf' else mod}_J{horizon}"
        if (not preds.empty and cname in preds.columns
                and actual is not None and len(actual)>0):
            d = dir_acc(actual, preds[cname])
            if d is not None: da[mod] = d

    kc = st.columns(len(models))
    for col_k, mod in zip(kc, models):
        with col_k:
            row = sub[sub["Modèle"]==mod]
            rmse = row["RMSE"].values[0] if not row.empty else None
            is_b = (mod == get_best_model(metrics_df, currency, horizon))
            naive_row = sub[sub["Modèle"]=="Naïf"]
            naive_r = naive_row["RMSE"].values[0] if not naive_row.empty else None
            dlt = (f"{(naive_r-rmse)/naive_r*100:+.1f}%"
                  if rmse and naive_r and mod!="Naïf" else None)
            st.metric(f"{'★ ' if is_b else ''}{mod}",
                     f"{rmse:.5f}" if rmse else "—",
                     dlt, delta_color="inverse")
            if mod in da:
                icon = "🟢" if da[mod]>55 else "🟡" if da[mod]>50 else "🔴"
                st.markdown(f'<span class="m-chip">Dir.Acc '
                           f'<b>{da[mod]:.1f}%</b> {icon}</span>',
                           unsafe_allow_html=True)

    if not metrics_df.empty:
        fig_b = go.Figure()
        cur_metrics = metrics_df[metrics_df["Devise"]==f"TND/{currency}"]
        for mod in models:
            vals = []
            for h in [1,7,30]:
                r = cur_metrics[(cur_metrics["Modèle"]==mod) &
                               (cur_metrics["Horizon"]==f"J+{h}")]
                vals.append(r["RMSE"].values[0] if not r.empty else 0)
            fig_b.add_trace(go.Bar(name=mod, x=["J+1","J+7","J+30"], y=vals,
                                   marker_color=cmap[mod], marker_opacity=.85))
        layout_b = dark_layout()
        layout_b.update(barmode="group", yaxis_title="RMSE (TND)", height=330,
                        legend=dict(orientation="h",bgcolor="rgba(0,0,0,0)"))
        fig_b.update_layout(**layout_b)
        axes(fig_b)
        st.plotly_chart(fig_b, use_container_width=True)

    if da:
        best_da = max(da, key=da.get)
        st.markdown(f'<div class="box">🧭 Meilleure Directional Accuracy : '
                   f'<b>{best_da}</b> → <b>{da[best_da]:.1f}%</b> '
                   f'(&gt;50% = mieux que le hasard).</div>',
                   unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  TAB 3 — Diebold-Mariano
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔬 Tests de Diebold-Mariano")
    st.caption("p < 0.05 = différence statistiquement significative")

    if not dm_df.empty:
        dc = dm_df[dm_df["Devise"]==f"TND/{currency}"].copy()
        if not dc.empty:
            pairs = (dc["Modèle 1"]+" vs "+dc["Modèle 2"]).unique()
            horiz = ["J+1","J+7","J+30"]
            pm = np.ones((len(pairs),len(horiz)))
            for ip,pr in enumerate(pairs):
                m1,m2 = pr.split(" vs ")
                for ih,h in enumerate(horiz):
                    row = dc[(dc["Modèle 1"]==m1)&(dc["Modèle 2"]==m2)&
                            (dc["Horizon"]==h)]
                    if not row.empty:
                        pm[ip,ih] = row["p-value"].values[0]
            fig_dm = go.Figure(data=go.Heatmap(
                z=pm, x=horiz, y=list(pairs), colorscale="RdYlGn_r",
                zmin=0, zmax=.15,
                text=[[f"p={pm[i,j]:.3f}" for j in range(len(horiz))]
                     for i in range(len(pairs))],
                texttemplate="%{text}", textfont=dict(size=9,color="white")))
            fig_dm.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#EDF1F7"),
                                margin=dict(l=120,r=18,t=25,b=35))
            st.plotly_chart(fig_dm, use_container_width=True)

            if "Significatif" in dc.columns:
                n_sig = int(dc["Significatif"].sum())
                st.markdown(f'<div class="box">{n_sig}/{len(dc)} '
                           f'comparaisons significatives (p&lt;0.05).</div>',
                           unsafe_allow_html=True)
            st.dataframe(dc, use_container_width=True, hide_index=True)
    else:
        st.info("`python -m src.s2_arima_garch.diebold_mariano`")

# ═══════════════════════════════════════════════════════════
#  TAB 4 — GARCH
# ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"### 📉 GARCH(1,1) — TND/{currency}")
    if sigma is not None and len(sigma)>0:
        df_sig = sigma.to_frame("sigma")
        df_sp  = period_bar("garch", df_sig)
        s_ann  = df_sp["sigma"]*np.sqrt(252)*100

        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=s_ann.index, y=s_ann.values,
                                   name="σ_t ann.", line=dict(color=RED,width=.9),
                                   fill="tozeroy",
                                   fillcolor="rgba(229,72,77,.08)"))
        layout_g = dark_layout()
        layout_g.update(yaxis_title="Vol. ann. (%)", height=340,
                        hovermode="x unified")
        fig_g.update_layout(**layout_g)
        axes(fig_g)
        st.plotly_chart(fig_g, use_container_width=True)

        l_sig, var99 = float(sigma.iloc[-1]), 2.326*float(sigma.iloc[-1])*100
        kc = st.columns(4)
        with kc[0]: st.metric("Vol. moy.", f"{s_ann.mean():.2f}%/an")
        with kc[1]: st.metric("Vol. max.", f"{s_ann.max():.2f}%/an")
        with kc[2]: st.metric("Vol. actuelle", f"{s_ann.iloc[-1]:.2f}%/an")
        with kc[3]: st.metric("VaR 99% (1j)", f"{var99:.3f}%")
    else:
        st.info("`python -m src.s2_arima_garch.run_s2`")

st.divider()
st.caption("ARIMA · GARCH · LSTM · MLP · "
          "Youssef Neji — MINDS ENIT 2026")