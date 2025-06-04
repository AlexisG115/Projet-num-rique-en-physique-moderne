
# Simulation de l'effet Ramsauer-Townsend

# Ce code simule la propagation d'un paquet d'ondes gaussien
# interagissant avec un puits de potentiel 1D, afin de visualiser
# la densite de probabilite de presence d'une particule quantique.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math


# Parametres physiques et grille


dt = 1e-7                  # Pas de temps
dx = 0.001                # Pas d'espace
nx = int(2 / dx)          # Nombre de points spatiaux sur [0, 2]
nt = 90000                # Nombre total de pas de temps
n_frames = int(nt / 1000) + 1  # Nombre d'images pour l'animation
s = dt / dx**2            # Coefficient pour le schema numerique



# Parametres du potentiel


v0 = -4000                # Profondeur du puits (négative)
e = 5                     # Rapport E/V0, donc E = e * V0
E = e * v0
k = math.sqrt(2 * abs(E)) # Norme du vecteur d'onde associe a l'energie

# Axe spatial
x_array = np.linspace(0, (nx - 1) * dx, nx)

# Potentiel : puits localisé entre 0.8 et 0.9
V = np.zeros(nx)
V[(x_array >= 0.8) & (x_array <= 0.9)] = v0



# Paquet d'ondes gaussien initial


xc = 0.6                  # Centre initial du paquet
sigma = 0.05              # Largeur du paquet
A = 1 / (math.sqrt(sigma * math.sqrt(math.pi)))  # Facteur de normalisation

# Fonction d'onde initiale complexe : gaussienne * phase plane
psi_init = A * np.exp(1j * k * x_array - ((x_array - xc) ** 2) / (2 * sigma**2))

# Parties reelle et imaginaire de la fonction d'onde
re = np.real(psi_init).copy()
im = np.imag(psi_init).copy()
b = np.zeros(nx)          # Tampon pour le schema



# Tableau pour stocker la densite de probabilite


density = np.zeros((nt, nx))
density[0, :] = np.abs(psi_init)**2

# Tableau pour stocker les densites echantillonnees pour l'animation
final_density = np.zeros((n_frames, nx))
final_density[0, :] = density[0, :]



# Boucle principale de simulation


it = 0
for i in range(1, nt):
    if i % 2 != 0:
        b[1:-1] = im[1:-1]
        im[1:-1] += s * (re[2:] + re[:-2]) - 2 * re[1:-1] * (s + V[1:-1] * dt)
        density[i, 1:-1] = re[1:-1]**2 + b[1:-1]**2
    else:
        re[1:-1] -= s * (im[2:] + im[:-2]) - 2 * im[1:-1] * (s + V[1:-1] * dt)

    if (i - 1) % 1000 == 0:
        it += 1
        final_density[it, :] = density[i, :]



# Animation matplotlib


fig = plt.figure(figsize=(10, 5))
line, = plt.plot([], [], lw=2)
plt.plot(x_array, V / abs(v0), label="Potentiel (echelle reduite)")
plt.ylim(0, 1.2)
plt.xlim(0, 2)
plt.title(f"Propagation du paquet d'ondes - E/V0 = {e}")
plt.xlabel("x")
plt.ylabel("Densite de probabilite")
plt.legend()

def init():
    line.set_data([], [])
    return line,

def animate(j):
    line.set_data(x_array, final_density[j, :])
    return line,

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                              interval=80, blit=False, repeat=False)
plt.show()
