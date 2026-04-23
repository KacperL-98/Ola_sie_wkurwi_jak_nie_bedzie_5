import numpy as np
from scipy.constants import e, epsilon_0, hbar, m_e
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

# Physical parameters (example values for WSe2)
m_eff_e = 0.35 * m_e  # electron effective mass
m_eff_h = 0.45 * m_e  # hole effective mass
mu = (m_eff_e * m_eff_h) / (m_eff_e + m_eff_h)  # reduced mass
eps_r = 5.15           # effective dielectric constant
d = 0.6e-10           # interlayer distance (m)

# Grid parameters
rho_max = 10e-9       # max radial distance (m)
N = 500               # number of grid points
rho = np.linspace(1e-12, rho_max, N)
dr = rho[1] - rho[0]

# Softened Coulomb potential
V = - (e**2) / (4 * np.pi * epsilon_0 * eps_r * np.sqrt(rho**2 + d**2))

# Kinetic energy operator (radial Laplacian in 2D)
diag_main = -2.0 * np.ones(N) / dr**2
diag_off = 1.0 * np.ones(N-1) / dr**2
laplacian = diags([diag_off, diag_main, diag_off], [-1, 0, 1]) / (2 * mu / hbar**2)

# Hamiltonian
H = -laplacian + diags(V, 0)

# Solve for lowest eigenvalue (binding energy)
E_vals, E_vecs = eigsh(H, k=1, which='SA')
E_binding = E_vals[0]

print(f"Interlayer exciton binding energy ≈ {abs(E_binding)/e:.3f} eV")
