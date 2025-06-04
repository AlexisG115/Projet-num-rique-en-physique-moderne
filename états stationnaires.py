
# Recherche numérique des états stationnaires dans un puits de potentiel fini

# Recherche des états propres de l'équation de Schrödinger dans un puits de potentiel en 1D
# Diagonalisation du Hamiltonien discret

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# Paramètres numériques

hbar = 1
m = 1
dx = 0.005
x_min, x_max = 0, 2
x = np.arange(x_min, x_max, dx)
nx = len(x)


# Potentiel : puits de potentiel entre 0.8 et 1.2

V0 = -4000
V = np.zeros(nx)
V[(x >= 0.8) & (x <= 1.2)] = V0


# Construction de la matrice tridiagonale du Hamiltonien

main_diag = hbar**2 / (m * dx**2) + V
off_diag = np.full(nx - 1, -hbar**2 / (2 * m * dx**2))


# Diagonalisation pour obtenir les états propres

energies, states = eigh_tridiagonal(main_diag, off_diag)


# Sélection des 3 premiers états liés

nb_states = 3
plt.figure(figsize=(10, 6))
for i in range(nb_states):
    psi = states[:, i]
    psi_norm = psi / np.sqrt(np.sum(psi**2) * dx)
    plt.plot(x, psi_norm**2, label=f"Etat {i+1}, E = {energies[i]:.2f}")

plt.plot(x, V / abs(V0), label="Potentiel (rescale)", color='k', linestyle='--')
plt.title("États stationnaires dans un puits de potentiel fini")
plt.xlabel("x")
plt.ylabel("|psi(x)|^2")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
