# =============================================================
#  dashboard/utils/ai_interpret.py
#  Module d'interprétation — Groq API + Fallback offline
#  Auteur : Youssef Neji | MINDS ENIT | 2026
# =============================================================

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
USE_AI_GLOBAL = False


# =============================================================
#  Interprétations — Module Prévisions (S2/S3)
# =============================================================

def interpret_ai_groq(pred, current, horizon, currency,
                      model, garch_vol, vix, brent,
                      taux_fed, taux_bct) -> str | None:
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        chg    = (pred - current) / current * 100
        client = Groq(api_key=GROQ_API_KEY)
        prompt = (
            f"You are a senior FX analyst at BIAT "
            f"(Banque Internationale Arabe de Tunisie). "
            f"Write a professional analysis in French "
            f"(3-4 sentences) for a risk manager.\n\n"
            f"Data:\n"
            f"- Currency: TND/{currency}\n"
            f"- Current rate: {current:.4f}\n"
            f"- {model} forecast J+{horizon}: "
            f"{pred:.4f} ({chg:+.3f}%)\n"
            f"- GARCH volatility: {garch_vol:.2f}%/yr\n"
            f"- VIX: {vix:.1f}\n"
            f"- Brent: {brent:.1f} USD/bbl\n"
            f"- Fed rate: {taux_fed:.2f}%\n"
            f"- BCT rate: {taux_bct:.2f}%\n\n"
            f"Include: (1) direction and magnitude, "
            f"(2) main macro driver, "
            f"(3) preliminary hedging recommendation. "
            f"IMPORTANT: start directly with the content, "
            f"no greeting or introduction phrase. "
            f"Be concise, professional, no math formulas, "
            f"and finish your sentences completely."
        )
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}],
            max_tokens=350,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Groq API error] {e}")
        return None


def interpret_fallback(pred, current, horizon,
                       currency, model, garch_vol,
                       vix=18.0, brent=80.0,
                       taux_fed=None, taux_bct=None) -> str:
    chg    = (pred - current) / current * 100
    direct = "dépréciation" if chg > 0 else "appréciation"
    mag    = ("légère"        if abs(chg) < 0.2 else
              "modérée"       if abs(chg) < 0.5 else
              "significative" if abs(chg) < 1.0 else
              "forte")
    urgence = "⚠️" if abs(chg) > 0.5 else "ℹ️"

    macro_ctx = []
    if vix > 25:
        macro_ctx.append(
            "Le VIX > 25 signale un stress financier mondial élevé"
        )
    elif vix > 20:
        macro_ctx.append(
            "Le VIX > 20 indique une aversion au risque modérée"
        )
    if brent > 95:
        macro_ctx.append(
            "le Brent élevé pèse sur la facture énergétique tunisienne"
        )
    elif brent < 60:
        macro_ctx.append(
            "le Brent bas soulage les importations énergétiques"
        )
    macro_str = (
        " — " + " et ".join(macro_ctx) + "."
        if macro_ctx else "."
    )

    if chg > 0.5:
        rec = ("→ **Couverture urgente recommandée** : "
               "Forward pour fixer le cours immédiatement.")
    elif chg > 0.2:
        rec = ("→ Envisager une **Option PUT** pour "
               "se protéger en conservant l'upside.")
    elif chg < -0.2:
        rec = ("→ **Pas de couverture urgente** : "
               "appréciation prévue, surveiller l'évolution.")
    else:
        rec = ("→ Mouvement limité, surveillance recommandée.")

    horizon_ctx = {1: "demain", 7: "la semaine prochaine",
                  30: "le mois prochain"}

    return (
        f"{urgence} **{model}** prédit une {mag} "
        f"{direct} de **{abs(chg):.3f}%** pour "
        f"{horizon_ctx.get(horizon, f'J+{horizon}')} "
        f"(TND/{currency} : "
        f"{current:.4f} → **{pred:.4f}**)"
        f"{macro_str} "
        f"Volatilité GARCH actuelle : "
        f"**{garch_vol:.2f}%/an**. {rec}"
    )


def get_interpretation(use_ai: bool, **kwargs) -> str:
    if use_ai and GROQ_API_KEY:
        result = interpret_ai_groq(**kwargs)
        if result:
            return result
    return interpret_fallback(**kwargs)


def hedging_recommendation(chg_pct: float,
                            horizon: int,
                            currency: str) -> dict:
    h_days = horizon
    if chg_pct > 0.5:
        return {
            "type": "FORWARD", "icon": "🔴", "color": "#C0392B",
            "title": "Couverture URGENTE — Forward recommandé",
            "text": (
                f"Dépréciation prévue de **{chg_pct:.3f}%** "
                f"à J+{horizon}. Un contrat Forward à {h_days}j "
                f"vous permet de fixer le cours actuel et "
                f"d'éliminer le risque de change. "
                f"→ Calculez le prix Forward dans la page "
                f"**Couverture**."
            )
        }
    elif chg_pct > 0.2:
        return {
            "type": "OPTION PUT", "icon": "🟡", "color": "#E65100",
            "title": "Option PUT recommandée",
            "text": (
                f"Dépréciation modérée prévue "
                f"(**{chg_pct:.3f}%** à J+{horizon}). "
                f"Une Option PUT protège contre la baisse tout "
                f"en conservant le bénéfice d'une appréciation "
                f"éventuelle. → Pricing Garman-Kohlhagen "
                f"disponible dans la page **Couverture**."
            )
        }
    elif chg_pct < -0.3:
        return {
            "type": "ATTENDRE", "icon": "🟢", "color": "#1B7F4F",
            "title": "Pas de couverture — Appréciation prévue",
            "text": (
                f"Appréciation prévue de "
                f"**{abs(chg_pct):.3f}%** à J+{horizon}. "
                f"Une couverture immédiate pénaliserait votre "
                f"position. Surveillez l'évolution et attendez "
                f"une opportunité plus favorable."
            )
        }
    else:
        return {
            "type": "SURVEILLER", "icon": "⚪", "color": "#8B949E",
            "title": "Mouvement limité — Surveillance",
            "text": (
                f"Variation prévue faible "
                f"(**{abs(chg_pct):.3f}%**). Pas d'action urgente "
                f"requise. Réévaluez la position à J+1."
            )
        }


# =============================================================
#  Interprétations — Module Risque (S4)
# =============================================================

def interpret_var_fallback(profile: str, currency: str, method: str,
                           var_value: float, cvar_value: float,
                           confidence: float, exposure: float,
                           kupiec_valid: bool) -> str:
    var_pct = var_value / exposure * 100 if exposure else 0
    severity = "⚠️" if var_pct > 1.5 else "ℹ️"

    validation = (
        f"Cette méthode est **validée** par le backtesting de Kupiec "
        f"à {confidence*100:.0f}% — fiable pour le pilotage opérationnel."
        if kupiec_valid else
        f"⚠️ Cette méthode **n'est pas validée** par Kupiec à "
        f"{confidence*100:.0f}% — à utiliser avec prudence, préférer "
        f"la VaR Historique si disponible."
    )

    return (
        f"{severity} Pour **{profile}**, la VaR **{method}** à "
        f"{confidence*100:.0f}% est de **{var_value:,.0f} TND** "
        f"({var_pct:.2f}% de l'exposition). En cas de dépassement "
        f"de ce seuil, la perte moyenne attendue (CVaR) est de "
        f"**{cvar_value:,.0f} TND**. {validation}"
    )


def interpret_var_ai(profile: str, currency: str, method: str,
                     var_value: float, cvar_value: float,
                     confidence: float, exposure: float,
                     kupiec_valid: bool) -> str | None:
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        prompt = (
            f"Tu es un risk manager senior à la BIAT. Analyse ce "
            f"résultat de VaR en français, 3-4 phrases, pour un "
            f"comité des risques.\n\n"
            f"Profil : {profile} (exposition {exposure:,.0f} TND "
            f"en {currency})\n"
            f"Méthode : {method}, confiance {confidence*100:.0f}%\n"
            f"VaR : {var_value:,.0f} TND\n"
            f"CVaR : {cvar_value:,.0f} TND\n"
            f"Validée par Kupiec : {'oui' if kupiec_valid else 'non'}\n\n"
            f"Explique le niveau de risque, la fiabilité de la "
            f"méthode, et une recommandation opérationnelle. "
            f"IMPORTANT : commence directement par le contenu, "
            f"sans salutation ni formule d'introduction "
            f"('Bonjour', 'Je vous présente', etc.). Sois direct, "
            f"concis, et termine tes phrases complètement."
        )
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}],
            max_tokens=350, temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Groq VaR error] {e}")
        return None


def get_var_interpretation(use_ai: bool, **kwargs) -> str:
    if use_ai and GROQ_API_KEY:
        result = interpret_var_ai(**kwargs)
        if result:
            return result
    return interpret_var_fallback(**kwargs)


# =============================================================
#  Interprétations — Module Couverture (S5)
# =============================================================

def interpret_hedging_fallback(profile: str, currency: str,
                               instrument_type: str, horizon: int,
                               forward_cost: float, option_premium: float,
                               breakeven: float, forecast: float,
                               recommendation: str) -> str:
    ecart = abs(forecast - breakeven) / breakeven * 100 if breakeven else 0
    confiance = "forte" if ecart > 0.5 else "modérée"

    if recommendation == "Forward":
        raison = (
            f"le cours prédit ({forecast:.5f}) n'atteint pas le seuil "
            f"de rentabilité de l'Option ({breakeven:.5f}) — la prime "
            f"payée pour l'Option ({option_premium:,.0f} TND) ne serait "
            f"pas compensée par le mouvement de marché anticipé."
        )
    else:
        raison = (
            f"le cours prédit ({forecast:.5f}) dépasse le seuil de "
            f"rentabilité de l'Option ({breakeven:.5f}) — l'Option "
            f"devient économiquement avantageuse selon le modèle."
        )

    return (
        f"🎯 Pour **{profile}** à J+{horizon}, le modèle recommande "
        f"le **{recommendation}** avec une confiance {confiance} "
        f"(écart de {ecart:.2f}% entre prévision et seuil). "
        f"Concrètement, {raison} Coût Forward estimé : "
        f"**{forward_cost:,.0f} TND** vs prime Option : "
        f"**{option_premium:,.0f} TND**."
    )


def interpret_hedging_ai(profile: str, currency: str,
                         instrument_type: str, horizon: int,
                         forward_cost: float, option_premium: float,
                         breakeven: float, forecast: float,
                         recommendation: str) -> str | None:
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        prompt = (
            f"Tu es un trésorier senior à la BIAT conseillant un "
            f"client sur sa stratégie de couverture de change. "
            f"Rédige 3-4 phrases en français.\n\n"
            f"Client : {profile}, devise {currency}, horizon J+{horizon}\n"
            f"Type d'option : {instrument_type}\n"
            f"Coût Forward : {forward_cost:,.0f} TND\n"
            f"Prime Option : {option_premium:,.0f} TND\n"
            f"Seuil de rentabilité Option : {breakeven:.5f}\n"
            f"Cours prédit par le modèle : {forecast:.5f}\n"
            f"Recommandation du système : {recommendation}\n\n"
            f"Explique pourquoi cet instrument est recommandé et "
            f"ce qui pourrait faire changer cette recommandation. "
            f"IMPORTANT : commence directement par le contenu, "
            f"sans salutation ni formule d'introduction. Sois "
            f"direct, concis, et termine tes phrases complètement."
        )
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}],
            max_tokens=350, temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Groq hedging error] {e}")
        return None


def get_hedging_interpretation(use_ai: bool, **kwargs) -> str:
    if use_ai and GROQ_API_KEY:
        result = interpret_hedging_ai(**kwargs)
        if result:
            return result
    return interpret_hedging_fallback(**kwargs)


# =============================================================
#  Interprétations — Module Portefeuille / Markowitz (S5)
# =============================================================

def interpret_markowitz_fallback(current_eur_pct: float, current_sharpe: float,
                                 minvar_eur_pct: float, minvar_sharpe: float,
                                 maxsharpe_eur_pct: float, maxsharpe_sharpe: float,
                                 rf_pct: float, tangent_return_pct: float,
                                 ecart_pts: float, tnd_domine: bool) -> str:
    ecart_alloc = abs(current_eur_pct - minvar_eur_pct)
    rebalance = "significatif" if ecart_alloc > 10 else "mineur"

    if tnd_domine:
        conclusion = (
            f"le taux TND (**{rf_pct:.2f}%**) domine intégralement la "
            f"frontière EUR/USD — aucune combinaison de devises n'égale "
            f"ce rendement à ce niveau de risque (écart de "
            f"**{ecart_pts:.2f} points**). Détenir du TND pur reste "
            f"économiquement supérieur à toute diversification EUR/USD."
        )
    else:
        conclusion = (
            f"le portefeuille tangent EUR/USD (rendement "
            f"**{tangent_return_pct:.2f}%**) dépasse le taux TND "
            f"(**{rf_pct:.2f}%**) — une allocation en devises peut "
            f"améliorer le couple rendement/risque par rapport au cash TND."
        )

    return (
        f"📐 Position actuelle **{current_eur_pct:.0f}% EUR / "
        f"{100-current_eur_pct:.0f}% USD** (Sharpe {current_sharpe:.3f}) "
        f"vs allocation à variance minimale **{minvar_eur_pct:.0f}% EUR / "
        f"{100-minvar_eur_pct:.0f}% USD** (Sharpe {minvar_sharpe:.3f}) — "
        f"écart {rebalance} ({ecart_alloc:.0f} points). "
        f"Sur la Capital Market Line, {conclusion}"
    )


def interpret_markowitz_ai(current_eur_pct: float, current_sharpe: float,
                           minvar_eur_pct: float, minvar_sharpe: float,
                           maxsharpe_eur_pct: float, maxsharpe_sharpe: float,
                           rf_pct: float, tangent_return_pct: float,
                           ecart_pts: float, tnd_domine: bool) -> str | None:
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        prompt = (
            f"Tu es un gérant de portefeuille senior à la BIAT analysant "
            f"l'allocation EUR/USD du Desk BIAT. Rédige 3-4 phrases en "
            f"français pour un comité d'investissement.\n\n"
            f"Position actuelle : {current_eur_pct:.0f}% EUR / "
            f"{100-current_eur_pct:.0f}% USD, Sharpe {current_sharpe:.3f}\n"
            f"Variance minimale : {minvar_eur_pct:.0f}% EUR / "
            f"{100-minvar_eur_pct:.0f}% USD, Sharpe {minvar_sharpe:.3f}\n"
            f"Sharpe maximal (tangent) : {maxsharpe_eur_pct:.0f}% EUR / "
            f"{100-maxsharpe_eur_pct:.0f}% USD, Sharpe {maxsharpe_sharpe:.3f}\n"
            f"Taux sans risque TND : {rf_pct:.2f}%\n"
            f"Rendement tangent FX : {tangent_return_pct:.2f}%\n"
            f"Le TND domine la frontière FX : "
            f"{'oui' if tnd_domine else 'non'}\n\n"
            f"Explique si un rééquilibrage vers l'allocation optimale "
            f"est justifié, et pourquoi le TND est ou n'est pas "
            f"compétitif face au risque de change. IMPORTANT : commence "
            f"directement par le contenu, sans salutation ni formule "
            f"d'introduction. Sois direct, concis, et termine tes "
            f"phrases complètement."
        )
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}],
            max_tokens=350, temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Groq Markowitz error] {e}")
        return None


def get_markowitz_interpretation(use_ai: bool, **kwargs) -> str:
    if use_ai and GROQ_API_KEY:
        result = interpret_markowitz_ai(**kwargs)
        if result:
            return result
    return interpret_markowitz_fallback(**kwargs)