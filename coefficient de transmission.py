
import numpy as np
import matplotlib.pyplot as plt
import math


# PARAMÈTRES NUMÉRIQUES


dt = 1e-7              # Pas de temps
dx = 0.001             # Pas d’espace
nx = int(2 / dx)       # Nombre de points spatiaux
nt = 20000             # Nombre de pas de temps (réduit pour rapidité)
s = dt / dx**2         # Coefficient numérique

# Axe spatial
x_array = np.linspace(0, (nx - 1) * dx, nx)

# Potentiel : puits localisé entre 0.8 et 0.9
V = np.zeros(nx)
V[(x_array >= 0.8) & (x_array <= 0.9)] = -4000

# Zone où on mesure la transmission (à droite du puits)
transmitted_zone = x_array > 1.2



# FONCTION DE SIMULATION


def simulate_transmission(e_ratio):
    """Simule la propagation d’un paquet d’ondes pour un rapport E/V0 donné et calcule la transmission."""
    v0 = -4000
    E = e_ratio * v0
    k = math.sqrt(2 * abs(E))
    xc = 0.6
    sigma = 0.05
    A = 1 / (math.sqrt(sigma * math.sqrt(math.pi)))
    
    # Paquet d’ondes initial (gaussien)
    psi_init = A * np.exp(1j * k * x_array - ((x_array - xc) ** 2) / (2 * sigma**2))
    re = np.real(psi_init).copy()
    im = np.imag(psi_init).copy()
    b = np.zeros(nx)

    # Propagation temporelle
    for i in range(1, nt):
        if i % 2 != 0:
            b[1:-1] = im[1:-1]
            im[1:-1] += s * (re[2:] + re[:-2]) - 2 * re[1:-1] * (s + V[1:-1] * dt)
        else:
            re[1:-1] -= s * (im[2:] + im[:-2]) - 2 * im[1:-1] * (s + V[1:-1] * dt)

    # Densité de probabilité finale
    densite = re**2 + im**2
    transmission = np.sum(densite[transmitted_zone]) * dx
    return transmission



# CALCUL POUR PLUSIEURS ÉNERGIES


e_values = np.linspace(0.1, 10, 20)   # Valeurs de E/V0
T_values = [simulate_transmission(e) for e in e_values]  # Transmission pour chaque énergie



# TRACÉ DU GRAPHIQUE T(E)


plt.figure(figsize=(10, 5))
plt.plot(e_values, T_values, marker='o')
plt.title("Coefficient de transmission T(E) en fonction de E/V0")
plt.xlabel("E / V0")
plt.ylabel("T(E)")
plt.grid(True)
plt.tight_layout()
plt.show()
