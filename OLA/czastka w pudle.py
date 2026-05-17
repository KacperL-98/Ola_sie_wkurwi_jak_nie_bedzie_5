import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse import diags, kron, identity
from scipy.special import erfc
from scipy.optimize import minimize_scalar
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["font.family"] = "Times New Roman"

def solve_schrodinger_2d(
    L=10,
    N=50,
    me=0.24, #mo=1
    mh=0.45, #hh 0.45 lh=0.36
    hbar=1.0, #=1
    potential_func=None,
    num_states=5
):

    # Reduced mass
    m = (me * mh) / (me + mh)

    # 2D grid
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)

    dx = x[1] - x[0]

    X, Y = np.meshgrid(x, y)

    # Potential
    if potential_func is None:
        V = np.zeros((N, N))
    else:
        V = potential_func(X, Y)

    # Flatten potential into vector
    V_flat = V.flatten()

    # 1D finite difference Laplacian
    main_diag = -2.0 * np.ones(N)
    off_diag = np.ones(N - 1)

    D = diags(
        [off_diag, main_diag, off_diag],
        [-1, 0, 1]
    ) / dx**2

    # 2D Laplacian using Kronecker products
    I = identity(N)
    Laplacian_2D = kron(D, I) + kron(I, D)

    # Kinetic energy operator
    T = -(hbar**2 / (2 * m)) * Laplacian_2D

    # Potential energy operator
    V_matrix = diags(V_flat)

    # Hamiltonian
    H = T + V_matrix

    # Convert to dense matrix
    H = H.toarray()

    # Solve eigenvalue problem
    energies, wavefunctions = eigh(H)

    # Plot probability densities
    for i in range(num_states):

        psi = wavefunctions[:, i].reshape((N, N))

        # Normalize
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dx)
        psi /= norm

        mid = N // 2

        # przekrój ψ(x)
        psi_x = np.real(psi[mid, :])

        # przekrój |ψ|²
        prob_x = np.abs(psi[mid, :]) ** 2

        # --- wykres ψ(x) ---
        plt.figure(figsize=(8, 5))

        plt.plot(x, psi_x, linewidth=2)

        plt.title(f"ψ(x), stan {i + 1}")
        plt.xlabel("x")
        plt.ylabel("ψ")

        plt.grid(True)

        plt.show()

        # --- wykres |ψ|² ---
        plt.figure(figsize=(8, 5))

        plt.plot(x, prob_x, color='red', linewidth=2)

        plt.title(f"|ψ(x)|², stan {i + 1}")
        plt.xlabel("x")
        plt.ylabel("|ψ|²")

        plt.grid(True)

        plt.show()

        plt.figure(figsize=(6, 5))

        plt.imshow(
            np.abs(psi)**2,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin='lower'
        )

        plt.colorbar(label='Probability Density')

        plt.title(f"Stany własne {i+1}, energie = {energies[i]:.4f}")
        plt.xlabel("Oś x")
        plt.ylabel("Oś y")
        fig = plt.figure(figsize=(14, 6))

        X, Y = np.meshgrid(x, y)

        # --- ψ ---
        ax1 = fig.add_subplot(121, projection='3d')

        surf1 = ax1.plot_surface(
            X, Y,
            np.real(psi),
            cmap='coolwarm'
        )

        ax1.set_title("Funkcja falowa ψ")

        # --- |ψ|² ---
        ax2 = fig.add_subplot(122, projection='3d')

        surf2 = ax2.plot_surface(
            X, Y,
            np.abs(psi) ** 2,
            cmap='viridis'
        )

        ax2.set_title("Probability Density |ψ|²")

        plt.show()
        plt.show()

    return x, y, energies, wavefunctions, H

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(x, y)

psi = wavefunctions[:, i].reshape((N, N))

# część rzeczywista funkcji falowej
Z = np.real(psi)

surf = ax.plot_surface(
    X, Y, Z,
    cmap='coolwarm'
)

ax.set_title("Funkcja falowa")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("ψ(x,y)")

fig.colorbar(surf, shrink=0.5)

plt.show()