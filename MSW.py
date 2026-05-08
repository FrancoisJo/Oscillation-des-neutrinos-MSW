# -*- coding: utf-8 -*-
"""
2025/2026
FRANÇOIS Jonas
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
from scipy.constants import m_p

# On rappel que dans notre cas e = 0, µ = 1, τ = 2

# Angle de mélange en radian, estimé experimentalement
theta12 = np.deg2rad(33.68)
theta13 = np.deg2rad(8.52)
theta23 = np.deg2rad(48.5)

c12 = np.cos(theta12)
s12 = np.sin(theta12)

c13 = np.cos(theta13)
s13 = np.sin(theta13)

c23 = np.cos(theta23)
s23 = np.sin(theta23)

E_GeV = 0.4         #GeV
E = E_GeV * 1e9     #eV

L_km = 735          #km
km_to_eV_inv = 5.0677e9
L = L_km * km_to_eV_inv   # eV^-1

E0 = 1e9      #eV
Ef = 20e9       #eV
    
m_p_gram = m_p *1e3     # Masse du proton en g
hbar_c  = 1.973e-7
    
# Densité électronique en cm⁻³
rho    = 2.85               # g/cm**3 sur la moyenne terrestre = 5.5 pour minos 2.85
Ye     = 0.49               # Sans unité
G_F    = 1.166e-23          # eV**-2
Ne_cm3 = Ye * rho / m_p_gram  # cm**-3
    
# Conversion cm**3 to eV**3
cm_to_eV_inv = 1e-2 / hbar_c       # 1 cm en eV**-1
Ne_eV3 = Ne_cm3 / cm_to_eV_inv**3  # eV**3
    
# Potentiel MSW
V0 = np.sqrt(2) * G_F * Ne_eV3      # eV 

def PMNS(anti):
    
    delta = np.deg2rad(177) 
    
    if anti : 
        delta = - delta

    # Matrices rotation, partie par partie
    U23 = np.array([
        [1,     0,     0],
        [0,   c23,   s23],
        [0,  -s23,   c23]
    ], dtype=complex)
    
    U13 = np.array([
        [c13, 0, s13*np.exp(-1j*delta)],
        [0,   1, 0],
        [-s13*np.exp(1j*delta), 0, c13]
    ], dtype=complex)
    
    U12 = np.array([
        [c12,  s12, 0],
        [-s12, c12, 0],
        [0,    0,   1]
    ], dtype=complex)
    
    # Matrice totale PMNS
    U = U23 @ U13 @ U12 #ici @ est un produit matriciel
    return U

def p(alpha, beta, L_km, E, anti, InvMasse, V0):    
    
    # Hiérarchie de masse inversé en eV**2
    dm21 = 7.49e-5
    dm31 = 2.534e-3 
    if InvMasse : 
        dm31 = -2.534e-3

    V = V0
    delta = np.deg2rad(177)
    
    # Antiparticule
    if anti == True:
        V = -V
        delta = - delta

    U = PMNS(anti)
    U_adj = U.conj().T
    
    Hvacmasse = np.array([[0, 0, 0],
                          [0, dm21/(2*E), 0],
                          [0, 0, dm31/(2*E)]
                          ], dtype=complex )
    
    Hv = np.array([
                 [V, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0] 
                 ], dtype=complex)
    
    Hvacsav = U @ Hvacmasse @ U_adj
    H_tot = Hvacsav + Hv
    
    vals, U_tilde = scipy.linalg.eigh(H_tot)
    km_to_eV_inv = 5.0677e9   
    
    L = L_km * km_to_eV_inv   # eV^-1
    
    amplitude = np.sum(U_tilde[beta, :] * np.conj(U_tilde[alpha, :]) * np.exp(-1j * vals * L))
    return np.abs(amplitude)**2

E_vals = np.logspace(np.log10(E0), np.log10(Ef), 1000)

A = L_km / (E_vals * 1e-9) #km/Gev
                
anti_vals = [False, True] # On définit des doublets afin de facilement passer de l'un à l'autre dans notre cas
InvMasse_vals = [False, True]

data = {}

for i in range(3):
    for j in range(3):
        for anti in anti_vals:
            for InvMasse in InvMasse_vals:
                key = (i, j, anti, InvMasse)
                   
                y = np.array([ p(i, j, L_km, E, anti, InvMasse, V0) for E in E_vals])
                   
                data[key] = y           
            
compteur = True

if compteur == True:
    incert = 10**-12
    
    for anti in anti_vals:
        for InvMasse in InvMasse_vals:
            
            # Somme sur j pour chaque alpha fixé
            for i in range(3):
                Ptot = sum(data[(i, j, anti, InvMasse)] for j in range(3))
                compteur_erreur = np.sum(np.abs(Ptot - 1) > incert)
                print(f"α={i}, anti={anti}, InvMasse={InvMasse} : {compteur_erreur} erreurs à {incert}")
                

courbes = [
    (1, 1, False, False),  
    (1, 1, True, False), 
    (1, 1, False, True),  
    (1, 1, True, True),
]

flavour = {0: r'e', 1: r'\mu', 2: r'\tau'}

fig, ax = plt.subplots(figsize=(10, 6))

for (i, j, anti, InvMasse) in courbes:
    barre = r'\bar' if anti else ''
    label = rf'$P({barre}{{\nu}}_{flavour[i]} \to {barre}{{\nu}}_{flavour[j]})$'
    ordre = 'Ordre de Masse Inversé' if InvMasse else 'Ordre de Masse Normal'
    ax.plot(A, data[(i, j, anti, InvMasse)], label=f'{label} ({ordre})')

ax.set_xlabel(r'$L/E$ (km/GeV)')
ax.set_ylabel(r'Probabilité')
ax.set_title(r"Oscillations des neutrinos avec effet de matière (MSW)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

