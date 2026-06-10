#WERSJA WSTEPNA KODU CZ2 (NIE DO OCENY/TESTOWA)
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
    num_states=4
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



# Coulomb-like potential
def potencjal(x, y):
    d = 1.134 #units in bohr ao = 0.53 A d = 6 A
    epsilon_0 = 1
    epsilon_r = 5.89 #epsilon_r to ma byc suma statych eps dlakonkretnego materialu dorobić funcje ktora liczy epsilon_r

    k = 1 / (4 * np.pi * epsilon_0 * epsilon_r) # k=1/epsilon_r usuniete 4 pie

    return -k / np.sqrt(x**2 + y**2 + d**2)


# Trial ground-state wavefunction
def test_function(x, y, beta_0=1):
    return np.sqrt(beta_0 / np.pi) * np.exp(-beta_0 * (x**2 + y**2) / 2) #nadmiarowa normalizacja


# Numerical mean Hamiltonian
def mean_hamiltonian(psi, H, dx):
    """
    Computes numerical expectation value:
        <H> = ∫∫ ψ* H ψ dx dy
    """

    psi_flat = psi.flatten()

    # Normalize in 2D
    norm = np.sqrt(np.sum(np.abs(psi_flat)**2) * dx * dx)
    psi_flat = psi_flat / norm

    expectation = np.vdot(psi_flat, H @ psi_flat) * dx * dx

    return np.real(expectation)


# Analytical variational energy E0(beta0)
def epsilon_0(beta_0, me=0.24, mh=0.45, hbar=1.0, d=1.134, epsilon_0=1, epsilon_r=5.89):
    """
    Computes:
        E0 = hbar^2 beta0 / 2m
             - k sqrt(pi beta0) exp(d^2 beta0) erfc(d sqrt(beta0))
    """

    m = (me * mh) / (me + mh)

    k = 1 / (4 * np.pi * epsilon_0 * epsilon_r)

    term1 = (hbar**2 * beta_0) / (2 * m)

    term2 = (
        k
        * np.sqrt(np.pi * beta_0)
        * np.exp(d**2 * beta_0)
        * erfc(d * np.sqrt(beta_0))
    )

    return term1 - term2


# Derivative dE0 / dβ0
def d_E_0_dbeta(beta_0, me=0.25, mh=0.45, hbar=1.0, d=1.134, epsilon_0=1, epsilon_r=5.89):
    """
    Computes derivative:
        dE0/dβ0
    """

    m = (me * mh) / (me + mh)

    k = 1 / (4 * np.pi * epsilon_0 * epsilon_r)

    sqrt_beta = np.sqrt(beta_0)

    term1 = hbar**2 / (2 * m)

    term2 = k * d * (
        1
        - (
            np.sqrt(np.pi)
            * (1 + 2 * d**2 * beta_0)
            * np.exp(d**2 * beta_0)
            * erfc(d * sqrt_beta)
        )
        / (2 * d * sqrt_beta)
    )

    return term1 + term2

# Find beta_0 that minimizes E_0
def find_beta_min(beta_min=1e-6, beta_max=10, me=1.0, mh=1.0, hbar=1.0, d=5, epsilon_0=1, epsilon_r=5.89):
    result = minimize_scalar(
        lambda beta: E_0(beta, me=me, mh=mh, hbar=hbar, d=d, epsilon_0_const=epsilon_0_const, epsilon_r=epsilon_r),
        bounds=(beta_min, beta_max),
        method="bounded"
    )

    return result.x, result.fun

# Check if energy changes less than tolerance
def energy_converged(E_old, E_new, tolerance=1e-6):
    return abs(E_new - E_old) < tolerance

def plot_epsilon_vs_beta(
    beta_min=1e-6,
    beta_max=10,
    points=500,
    me=0.25,
    mh=0.45,
    hbar=1.0,
    d=1.134,
    epsilon_0_const=1,
    epsilon_r=5.89
):
    betas = np.linspace(beta_min, beta_max, points)

    energies = np.array([
        epsilon_0(
            beta,
            me=me,
            mh=mh,
            hbar=hbar,
            d=d,
            epsilon_0_const=epsilon_0_const,
            epsilon_r=epsilon_r
        )
        for beta in betas
    ])

    beta_best, E_best = find_beta_min(
        beta_min=beta_min,
        beta_max=beta_max,
        me=me,
        mh=mh,
        hbar=hbar,
        d=d,
        epsilon_0_const=epsilon_0_const,
        epsilon_r=epsilon_r
    )

    plt.figure(figsize=(8, 5))
    plt.plot(betas, energies)
    plt.scatter(beta_best, E_best, label=f"min: beta={beta_best:.4f}")
    plt.xlabel(r"$\beta_0$")
    plt.ylabel(r"$\epsilon_0(\beta_0)$")
    plt.title("Variational Energy vs Beta")
    plt.legend()
    plt.grid(True)
    plt.show()

    return betas, energies

# Dipole moment
def dipole_moment(psi, x, y, charge=1.0, d=5.0):
    """
    Computes dipole moment.

    For symmetric wavefunctions:
        <x> = 0
        <y> = 0

    If electron and hole are separated vertically by d,
    then p_z = q d.
    """

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y)

    probability = np.abs(psi)**2

    norm = np.sum(probability) * dx * dy
    probability = probability / norm

    x_mean = np.sum(X * probability) * dx * dy
    y_mean = np.sum(Y * probability) * dx * dy

    p_x = charge * x_mean
    p_y = charge * y_mean
    p_z = charge * d

    p_total = np.sqrt(p_x**2 + p_y**2 + p_z**2)

    return p_x, p_y, p_z, p_total


# Run numerical solver
L = 10
N = 50

x, y, energies, wavefunctions, H = solve_schrodinger_2d(
    L=L,
    N=N,
    potential_func=potencjal
)

print("First 5 numerical energy levels:")

for i in range(5):
    print(f"E{i+1} = {energies[i]:.4f}")


# Build 2D grid for trial wavefunction
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]

beta_0 = 1.0

psi_trial = test_function(X, Y, beta_0)

E_mean_numeric = mean_hamiltonian(psi_trial, H, dx)
E_mean_analytic = epsilon_0(beta_0)
dE_dbeta = d_epsilon_0_dbeta(beta_0)

print("\nVariational / mean Hamiltonian results:")
print(f"beta_0 = {beta_0}")
print(f"Numerical <H> = {E_mean_numeric:.6f}")
print(f"Analytical epsilon_0(beta_0) = {E_mean_analytic:.6f}")
print(f"d epsilon_0 / d beta_0 = {dE_dbeta:.6f}")

# Parameters
me = 1.0
mh = 1.0
hbar = 1.0
d = 5
epsilon_0_const = 1
epsilon_r = 5.89

# Minimize energy
beta_best, E_best = find_beta_min(
    beta_min=1e-6,
    beta_max=10,
    me=me,
    mh=mh,
    hbar=hbar,
    d=d,
    epsilon_0_const=epsilon_0_const,
    epsilon_r=epsilon_r
)

print("\nMinimum variational energy:")
print(f"beta_0_min = {beta_best:.6f}")
print(f"epsilon_0_min = {E_best:.6f}")

# Check derivative at minimum
derivative_at_min = d_epsilon_0_dbeta(
    beta_best,
    me=me,
    mh=mh,
    hbar=hbar,
    d=d,
    epsilon_0_const=epsilon_0_const,
    epsilon_r=epsilon_r
)

print(f"d epsilon_0 / d beta_0 at minimum = {derivative_at_min:.6e}")

if abs(derivative_at_min) < 1e-6:
    print("Derivative is approximately zero.")
else:
    print("Derivative is not zero yet.")

# Plot energy as function of beta
plot_epsilon_vs_beta(
    beta_min=1e-6,
    beta_max=10,
    me=me,
    mh=mh,
    hbar=hbar,
    d=d,
    epsilon_0_const=epsilon_0_const,
    epsilon_r=epsilon_r
)

# Build trial wavefunction with best beta
X, Y = np.meshgrid(x, y)
psi_best = test_function(X, Y, beta_best)

# Calculate mean Hamiltonian with best beta
dx = x[1] - x[0]
E_mean_best = mean_hamiltonian(psi_best, H, dx)

print("\nMean Hamiltonian using optimized beta:")
print(f"<H> = {E_mean_best:.6f}")

# Check whether energy decreased
E_old = epsilon_0(1.0, me, mh, hbar, d, epsilon_0_const, epsilon_r)
E_new = E_best

print("\nEnergy comparison:")
print(f"Old energy beta_0=1: {E_old:.6f}")
print(f"New energy beta_0={beta_best:.6f}: {E_new:.6f}")

if E_new < E_old:
    print("Energy decreased.")
elif energy_converged(E_old, E_new):
    print("Energy change is smaller than 1e-6.")
else:
    print("Energy increased.")

# Dipole moment
p_x, p_y, p_z, p_total = dipole_moment(
    psi_best,
    x,
    y,
    charge=1.0,
    d=d
)

print("\nDipole moment:")
print(f"p_x = {p_x:.6e}")
print(f"p_y = {p_y:.6e}")
print(f"p_z = {p_z:.6e}")
print(f"|p| = {p_total:.6e}")

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
