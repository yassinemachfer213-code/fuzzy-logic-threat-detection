import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# --- Univers de discours ---
tentatives = ctrl.Antecedent(np.arange(0, 21, 1), 'tentatives')
trafic     = ctrl.Antecedent(np.arange(0, 101, 1), 'trafic')
ports      = ctrl.Antecedent(np.arange(0, 51, 1), 'ports')
menace     = ctrl.Consequent(np.arange(0, 101, 1), 'menace')

# --- Fonctions d'appartenance ---
tentatives['faible'] = fuzz.trapmf(tentatives.universe, [0, 0, 3, 7])
tentatives['moyen']  = fuzz.trimf(tentatives.universe, [3, 8, 13])
tentatives['eleve']  = fuzz.trapmf(tentatives.universe, [9, 14, 20, 20])

trafic['normal']     = fuzz.trapmf(trafic.universe, [0, 0, 30, 50])
trafic['eleve']      = fuzz.trimf(trafic.universe, [30, 55, 80])
trafic['tres_eleve'] = fuzz.trapmf(trafic.universe, [60, 80, 100, 100])

ports['peu']      = fuzz.trapmf(ports.universe, [0, 0, 5, 15])
ports['beaucoup'] = fuzz.trapmf(ports.universe, [8, 25, 50, 50])

menace['faible']   = fuzz.trapmf(menace.universe, [0, 0, 10, 40])
menace['moyenne']  = fuzz.trimf(menace.universe, [25, 50, 75])
menace['critique'] = fuzz.trapmf(menace.universe, [60, 90, 100, 100])

# --- Base de règles floues (Mamdani) ---
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

regles = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11]

# --- Le système de contrôle ---
systeme_ctrl = ctrl.ControlSystem(regles)
print("Nombre de règles :", len(regles))

# --- Affichage des courbes (décommente pour les voir) ---
# tentatives.view()
# trafic.view()
# ports.view()
# menace.view()
# plt.show()

# --- Test du système (Mamdani) ---
def calculer_menace(t, tr, p):
    sim = ctrl.ControlSystemSimulation(systeme_ctrl)
    sim.input['tentatives'] = t
    sim.input['trafic'] = tr
    sim.input['ports'] = p
    sim.compute()
    return sim.output['menace'], sim

scenarios = [
    ("Situation normale", 1, 20, 2),
    ("Scan de ports",     5, 40, 35),
    ("Brute force",       18, 50, 3),
    ("Attaque massive",   18, 90, 40),
]

for nom, t, tr, p in scenarios:
    resultat, _ = calculer_menace(t, tr, p)
    print(f"{nom:20s} -> tentatives={t}, trafic={tr}, ports={p} => menace = {resultat:.1f}")
    # Vérification manuelle du scan de ports (5, 40, 35)
faible_coupe  = np.fmin(0.5, menace['faible'].mf)
moyenne_coupe = np.fmin(0.5, menace['moyenne'].mf)
agrege = np.fmax(faible_coupe, moyenne_coupe)
print("Centre de gravité manuel :", fuzz.defuzz(menace.universe, agrege, 'centroid'))

# ================= SUGENO =================
def degre(var, terme, valeur):
    return fuzz.interp_membership(var.universe, var[terme].mf, valeur)

def calculer_menace_sugeno(t, tr, p):
    # Fuzzification : degrés d'appartenance
    T  = {k: degre(tentatives, k, t)  for k in ['faible', 'moyen', 'eleve']}
    TR = {k: degre(trafic, k, tr)     for k in ['normal', 'eleve', 'tres_eleve']}
    P  = {k: degre(ports, k, p)       for k in ['peu', 'beaucoup']}

    FAIBLE, MOYENNE, CRITIQUE = 10, 50, 90

    # Chaque règle = (poids, constante de sortie)
    regles_sugeno = [
        (min(T['eleve'], P['beaucoup']), CRITIQUE),                              # R1
        (min(T['eleve'], TR['tres_eleve']), CRITIQUE),                           # R2
        (min(TR['tres_eleve'], P['beaucoup']), CRITIQUE),                        # R3
        (min(T['eleve'], P['peu'], max(TR['normal'], TR['eleve'])), MOYENNE),    # R4
        (min(T['moyen'], P['beaucoup']), MOYENNE),                               # R5
        (min(T['moyen'], TR['eleve']), MOYENNE),                                 # R6
        (min(T['faible'], max(TR['tres_eleve'], P['beaucoup'])), MOYENNE),       # R7
        (min(T['faible'], TR['normal']), FAIBLE),                                # R8
        (min(T['faible'], P['peu']), FAIBLE),                                    # R9
        (min(T['moyen'], TR['normal'], P['peu']), FAIBLE),                       # R10
        (min(T['moyen'], TR['tres_eleve'], P['peu']), MOYENNE),                  # R11
    ]

    somme_poids = sum(w for w, _ in regles_sugeno)
    if somme_poids == 0:
        return None
    return sum(w * z for w, z in regles_sugeno) / somme_poids

# --- Comparaison Mamdani / Sugeno ---
print("\n--- Comparaison Mamdani / Sugeno ---")
for nom, t, tr, p in scenarios:
    m, _ = calculer_menace(t, tr, p)
    s = calculer_menace_sugeno(t, tr, p)
    print(f"{nom:20s} Mamdani = {m:5.1f} | Sugeno = {s:5.1f}")

    # ================= ÉTAPE 7 : GRAPHIQUES =================

    # --- 1) Comparaison Mamdani / Sugeno ---
    noms = [s[0] for s in scenarios]
    val_m = [calculer_menace(t, tr, p)[0] for _, t, tr, p in scenarios]
    val_s = [calculer_menace_sugeno(t, tr, p) for _, t, tr, p in scenarios]

    x = np.arange(len(noms))
    plt.figure(figsize=(8, 5))
    plt.bar(x - 0.2, val_m, 0.4, label='Mamdani')
    plt.bar(x + 0.2, val_s, 0.4, label='Sugeno')
    plt.xticks(x, noms, rotation=15)
    plt.ylabel('Niveau de menace (0 à 100)')
    plt.title('Comparaison Mamdani / Sugeno')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 2) Surface de contrôle 3D (trafic fixé à 50) ---
    tent_vals = np.arange(0, 21, 1)
    port_vals = np.arange(0, 51, 2)
    X, Y = np.meshgrid(tent_vals, port_vals)
    Z = np.zeros_like(X, dtype=float)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i, j] = calculer_menace(X[i, j], 50, Y[i, j])[0]
            except Exception:
                Z[i, j] = np.nan

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Tentatives')
    ax.set_ylabel('Ports scannés')
    ax.set_zlabel('Menace')
    ax.set_title('Surface de contrôle (trafic = 50)')
    fig.colorbar(surf, shrink=0.5)
    plt.show()