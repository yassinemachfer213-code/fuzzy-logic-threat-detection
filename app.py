import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Détection de menace réseau - Logique floue", layout="wide")


# ------------------------------------------------------------------
# 1) Construction du système flou (Fuzzification + règles Mamdani)
# ------------------------------------------------------------------
@st.cache_resource
def construire_systeme():
    tentatives = ctrl.Antecedent(np.arange(0, 21, 1), 'tentatives')
    trafic = ctrl.Antecedent(np.arange(0, 101, 1), 'trafic')
    ports = ctrl.Antecedent(np.arange(0, 51, 1), 'ports')
    menace = ctrl.Consequent(np.arange(0, 101, 1), 'menace')

    tentatives['faible'] = fuzz.trapmf(tentatives.universe, [0, 0, 3, 7])
    tentatives['moyen'] = fuzz.trimf(tentatives.universe, [3, 8, 13])
    tentatives['eleve'] = fuzz.trapmf(tentatives.universe, [9, 14, 20, 20])

    trafic['normal'] = fuzz.trapmf(trafic.universe, [0, 0, 30, 50])
    trafic['eleve'] = fuzz.trimf(trafic.universe, [30, 55, 80])
    trafic['tres_eleve'] = fuzz.trapmf(trafic.universe, [60, 80, 100, 100])

    ports['peu'] = fuzz.trapmf(ports.universe, [0, 0, 5, 15])
    ports['beaucoup'] = fuzz.trapmf(ports.universe, [8, 25, 50, 50])

    menace['faible'] = fuzz.trapmf(menace.universe, [0, 0, 10, 40])
    menace['moyenne'] = fuzz.trimf(menace.universe, [25, 50, 75])
    menace['critique'] = fuzz.trapmf(menace.universe, [60, 90, 100, 100])

    r1 = ctrl.Rule(tentatives['eleve'] & ports['beaucoup'], menace['critique'])
    r2 = ctrl.Rule(tentatives['eleve'] & trafic['tres_eleve'], menace['critique'])
    r3 = ctrl.Rule(trafic['tres_eleve'] & ports['beaucoup'], menace['critique'])
    r4 = ctrl.Rule(tentatives['eleve'] & ports['peu'] & (trafic['normal'] | trafic['eleve']), menace['moyenne'])
    r5 = ctrl.Rule(tentatives['moyen'] & ports['beaucoup'], menace['moyenne'])
    r6 = ctrl.Rule(tentatives['moyen'] & trafic['eleve'], menace['moyenne'])
    r7 = ctrl.Rule(tentatives['faible'] & (trafic['tres_eleve'] | ports['beaucoup']), menace['moyenne'])
    r8 = ctrl.Rule(tentatives['faible'] & trafic['normal'], menace['faible'])
    r9 = ctrl.Rule(tentatives['faible'] & ports['peu'], menace['faible'])
    r10 = ctrl.Rule(tentatives['moyen'] & trafic['normal'] & ports['peu'], menace['faible'])
    r11 = ctrl.Rule(tentatives['moyen'] & trafic['tres_eleve'] & ports['peu'], menace['moyenne'])

    systeme = ctrl.ControlSystem([r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11])
    return tentatives, trafic, ports, menace, systeme


tentatives, trafic, ports, menace, systeme = construire_systeme()

CONSTANTES_SUGENO = {'faible': 10, 'moyenne': 50, 'critique': 90}


def degre(var, terme, valeur):
    return float(fuzz.interp_membership(var.universe, var[terme].mf, valeur))


def poids_regles(t, tr, p):
    """Poids (degré d'activation) de chaque règle : ET = min, OU = max."""
    T = {k: degre(tentatives, k, t) for k in ['faible', 'moyen', 'eleve']}
    TR = {k: degre(trafic, k, tr) for k in ['normal', 'eleve', 'tres_eleve']}
    P = {k: degre(ports, k, p) for k in ['peu', 'beaucoup']}
    return [
        ("R1", "tentatives élevé ET ports beaucoup", min(T['eleve'], P['beaucoup']), 'critique'),
        ("R2", "tentatives élevé ET trafic très élevé", min(T['eleve'], TR['tres_eleve']), 'critique'),
        ("R3", "trafic très élevé ET ports beaucoup", min(TR['tres_eleve'], P['beaucoup']), 'critique'),
        ("R4", "tentatives élevé ET ports peu ET (trafic normal OU élevé)",
         min(T['eleve'], P['peu'], max(TR['normal'], TR['eleve'])), 'moyenne'),
        ("R5", "tentatives moyen ET ports beaucoup", min(T['moyen'], P['beaucoup']), 'moyenne'),
        ("R6", "tentatives moyen ET trafic élevé", min(T['moyen'], TR['eleve']), 'moyenne'),
        ("R7", "tentatives faible ET (trafic très élevé OU ports beaucoup)",
         min(T['faible'], max(TR['tres_eleve'], P['beaucoup'])), 'moyenne'),
        ("R8", "tentatives faible ET trafic normal", min(T['faible'], TR['normal']), 'faible'),
        ("R9", "tentatives faible ET ports peu", min(T['faible'], P['peu']), 'faible'),
        ("R10", "tentatives moyen ET trafic normal ET ports peu",
         min(T['moyen'], TR['normal'], P['peu']), 'faible'),
        ("R11", "tentatives moyen ET trafic très élevé ET ports peu",
         min(T['moyen'], TR['tres_eleve'], P['peu']), 'moyenne'),
    ], T, TR, P


def calcul_mamdani(t, tr, p):
    sim = ctrl.ControlSystemSimulation(systeme)
    sim.input['tentatives'] = t
    sim.input['trafic'] = tr
    sim.input['ports'] = p
    sim.compute()
    return float(sim.output['menace'])


def calcul_sugeno(regles):
    somme = sum(w for _, _, w, _ in regles)
    if somme == 0:
        return None
    return sum(w * CONSTANTES_SUGENO[c] for _, _, w, c in regles) / somme


def niveau(valeur):
    if valeur < 30:
        return "FAIBLE", "#16A34A"
    if valeur < 70:
        return "MOYENNE", "#D97706"
    return "CRITIQUE", "#DC2626"


def courbes(var, valeur, titre, unite):
    fig, ax = plt.subplots(figsize=(5, 3))
    for terme in var.terms:
        ax.plot(var.universe, var[terme].mf, linewidth=2, label=terme)
    ax.axvline(valeur, color='black', linestyle='--', linewidth=1.2)
    ax.set_title(titre, fontsize=11)
    ax.set_xlabel(unite, fontsize=9)
    ax.set_ylabel("degré µ", fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# 2) Interface
# ------------------------------------------------------------------
SCENARIOS = {
    "Personnalisé": None,
    "Situation normale": (1, 20, 2),
    "Scan de ports": (5, 40, 35),
    "Brute force": (18, 50, 3),
    "Attaque massive": (18, 90, 40),
}

st.session_state.setdefault('t', 5)
st.session_state.setdefault('tr', 40)
st.session_state.setdefault('p', 35)


def appliquer_scenario():
    valeurs = SCENARIOS.get(st.session_state.scen)
    if valeurs:
        st.session_state.t, st.session_state.tr, st.session_state.p = valeurs


st.title("Détection de menace réseau par logique floue")
st.caption("Systèmes flous de Mamdani et de Sugeno  |  LISI3, Faculté des Sciences de Sfax")

with st.sidebar:
    st.header("Mesures du réseau")
    st.selectbox("Scénario prédéfini", list(SCENARIOS.keys()), key="scen", on_change=appliquer_scenario)
    t = st.slider("Tentatives de connexion échouées / min", 0, 20, key="t")
    tr = st.slider("Trafic réseau (% de la capacité)", 0, 100, key="tr")
    p = st.slider("Ports scannés / min", 0, 50, key="p")
    st.markdown("---")
    st.caption("Modifie les curseurs : les deux systèmes recalculent la menace en direct.")

regles, T, TR, P = poids_regles(t, tr, p)

try:
    m = calcul_mamdani(t, tr, p)
except Exception:
    m = None
s = calcul_sugeno(regles)

if m is None or s is None:
    st.error("Aucune règle ne s'active pour ces valeurs : le système ne peut pas décider.")
    st.stop()

nom_niveau, couleur = niveau(m)

c1, c2, c3 = st.columns(3)
c1.metric("Menace (Mamdani)", f"{m:.1f} / 100")
c2.metric("Menace (Sugeno)", f"{s:.1f} / 100")
c3.markdown(
    f"<div style='padding:14px 18px;border-radius:10px;background:{couleur};color:white;"
    f"text-align:center;font-size:22px;font-weight:700'>Niveau : {nom_niveau}</div>",
    unsafe_allow_html=True)
st.progress(int(round(m)))

tab1, tab2, tab3, tab4 = st.tabs(["1. Fuzzification", "2. Règles activées", "3. Défuzzification (Mamdani)",
                                  "4. Sugeno"])

with tab1:
    st.subheader("Degrés d'appartenance des valeurs saisies")
    d1, d2, d3 = st.columns(3)
    d1.markdown(f"**tentatives = {t}**")
    d1.dataframe(pd.DataFrame({"terme": list(T.keys()), "degré": [round(v, 2) for v in T.values()]}),
                 hide_index=True, width='stretch')
    d2.markdown(f"**trafic = {tr} %**")
    d2.dataframe(pd.DataFrame({"terme": list(TR.keys()), "degré": [round(v, 2) for v in TR.values()]}),
                 hide_index=True, width='stretch')
    d3.markdown(f"**ports = {p}**")
    d3.dataframe(pd.DataFrame({"terme": list(P.keys()), "degré": [round(v, 2) for v in P.values()]}),
                 hide_index=True, width='stretch')
    g1, g2, g3 = st.columns(3)
    g1.pyplot(courbes(tentatives, t, "tentatives", "échecs / min"))
    g2.pyplot(courbes(trafic, tr, "trafic", "% de la capacité"))
    g3.pyplot(courbes(ports, p, "ports", "ports scannés / min"))

with tab2:
    st.subheader("Poids de chaque règle (ET = min, OU = max)")
    df = pd.DataFrame([{"Règle": n, "Condition": cond, "Poids": round(w, 2), "Conclusion": concl}
                       for n, cond, w, concl in regles])
    st.dataframe(df, hide_index=True, width='stretch')
    st.caption("Une règle avec un poids de 0 ne participe pas à la décision.")

with tab3:
    st.subheader("Agrégation (max) et centre de gravité")
    agg = np.zeros_like(menace.universe, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for terme in menace.terms:
        ax.plot(menace.universe, menace[terme].mf, linestyle=':', linewidth=1.3, label=terme)
    for _, _, w, concl in regles:
        agg = np.fmax(agg, np.fmin(w, menace[concl].mf))
    ax.fill_between(menace.universe, agg, color='gray', alpha=0.45, label="sortie floue agrégée")
    ax.axvline(m, color='red', linewidth=2, label=f"centre de gravité = {m:.1f}")
    ax.set_xlabel("niveau de menace")
    ax.set_ylabel("degré µ")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Chaque conclusion est écrêtée au poids de sa règle (implication min), puis toutes sont fusionnées (max).")

with tab4:
    st.subheader("Moyenne pondérée des constantes")
    lignes = [{"Règle": n, "Poids wi": round(w, 2), "Constante zi": CONSTANTES_SUGENO[c],
               "wi × zi": round(w * CONSTANTES_SUGENO[c], 2)} for n, _, w, c in regles if w > 0]
    st.dataframe(pd.DataFrame(lignes), hide_index=True, width='stretch')
    sw = sum(w for _, _, w, _ in regles)
    swz = sum(w * CONSTANTES_SUGENO[c] for _, _, w, c in regles)
    st.latex(r"z = \frac{\sum w_i \, z_i}{\sum w_i} = \frac{%.2f}{%.2f} = %.1f" % (swz, sw, s))
    st.caption("Constantes : faible = 10, moyenne = 50, critique = 90.")
