import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import struve, y0
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from matplotlib.gridspec import GridSpec
plt.rcParams["font.family"] = "Times New Roman"
#parametry ws2

chi2d = 77.5 #podatnosc warstwy 2d angstremach
 #w bohrach
ε_l = 13.6 #podłuzna przenikalnosc eps||
t = 6.15 #angrems srednia grubosc wartsty w bohrach wartosc podziekilem przez promien bohra 0.5e-10
kappa = 1 #
#r0 = ((chi2d*2*n.pi⋅)/kappa)
r0 = 487.325 #a angsremach dla monomwarstwy w powietrzu/prozni
me_K_WS2  = 0.27
mh_K_WS2  = 0.36
mh_G_WS2  = 3.0
mu_K_WS2 = 0.154

mu = mu_K_WS2
eps = kappa

HARTREE_TO_EV = 27.211386245988
#siatka_biegunowa
r_max = 400.0
Nr = 400
Ntheta = 300

r = np.linspace(1e-12, r_max, Nr)
theta = np.linspace(0.0, 2*np.pi, Ntheta)

dr = r[1] - r[0]
dtheta = theta[1] - theta[0]

R, Theta = np.meshgrid(r, theta, indexing="ij")

X = R * np.cos(Theta)
Y = R * np.sin(Theta)

#potencjal_rk
def V_RK(r):
    z = r / r0
    return -(np.pi/(2.0*eps*r0)) * (struve(0, z) - y0(z))

Vgrid = V_RK(R)

#normalizacja
def integrate2d(f):
    tmp = simpson(f, theta, axis=1)
    return simpson(tmp, r)

def normalize_wavefunction(psi):

    psi_sq = np.abs(psi)**2 * R

    integral_theta = simpson(psi_sq,theta,axis=1)

    integral_total = simpson(integral_theta,r)

    A = 1.0 / np.sqrt(integral_total)
    return A * psi, A

#laplasjan_biegunowy
def laplacian(psi):

    lap = np.zeros_like(psi)

    dpsi_dr = np.zeros_like(psi)
    dpsi_dr[1:-1,:] = (
        psi[2:,:] - psi[:-2,:]
    )/(2*dr)

    term_r = np.zeros_like(psi)
    term_r[1:-1,:] = (
        (R[2:,:]*dpsi_dr[2:,:] -
         R[:-2,:]*dpsi_dr[:-2,:])
        /(2*dr)
    )/R[1:-1,:]

    term_theta = np.zeros_like(psi)

    term_theta[:,1:-1] = (
        psi[:,2:] - 2*psi[:,1:-1] + psi[:,:-2]
    )/dtheta**2

    term_theta[:,0] = (
        psi[:,1] - 2*psi[:,0] + psi[:,-1]
    )/dtheta**2

    term_theta[:,-1] = (
        psi[:,0] - 2*psi[:,-1] + psi[:,-2]
    )/dtheta**2

    lap = term_r + term_theta/(R**2)

    return lap
#f_probna
def psi_1s(b):

    psi = np.exp(-R/b)
    psi_norm, A = normalize_wavefunction(psi)
    return psi_norm

def psi_2s(b2,b1):

    d = (b1+b2)/(2*b1)
    psi = (1.0 - d*R/b2) * np.exp(-R/b2)
    psi_norm, A = normalize_wavefunction(psi)
    return psi_norm

def psi_2px(b):

    psi = (R*np.cos(Theta)* np.exp(-R/b))
    psi_norm, A = normalize_wavefunction(psi)
    return psi_norm

def psi_2py(b):

    psi = ( R*np.sin(Theta)* np.exp(-R/b))
    psi_norm, A = normalize_wavefunction(psi)
    return psi_norm
#energia
def energy_expectation(psi):

    lap = laplacian(psi)
    Tpsi = -(1.0/(2.0*mu))*lap
    Hpsi = Tpsi + Vgrid*psi
    E = integrate2d( np.conjugate(psi)*Hpsi*R)
    return E.real

def energy_1s(b):
    return energy_expectation(psi_1s(b))

def energy_2s(b2, b1):
    return energy_expectation(psi_2s(b2, b1))

def energy_2px(b):
    return energy_expectation(psi_2px(b))

def energy_2py(b):
    return energy_expectation(psi_2py(b))

#optymalizacja
res1 = minimize_scalar(energy_1s,bounds=(0.1,500.0),method="bounded"
)

b1 = res1.x
E1 = res1.fun

res2 = minimize_scalar(lambda b: energy_2s(b,b1),bounds=(0.1,500.0),method="bounded"
)

b2 = res2.x
E2s = res2.fun

respx = minimize_scalar(
    energy_2px,bounds=(0.1,300.0), method="bounded"
)

bpx = respx.x
Epx = respx.fun

respy = minimize_scalar(energy_2py,bounds=(0.1,300.0),method="bounded")

bpy = respy.x
Epy = respy.fun

print("1s :", E1*HARTREE_TO_EV, "eV")
print("2s :", E2s*HARTREE_TO_EV, "eV")
print("2px:", Epx*HARTREE_TO_EV, "eV")
print("2py:", Epy*HARTREE_TO_EV, "eV")

# TEST NORMALIZACJI 2px dla b=1

psi_test = R*np.cos(Theta)*np.exp(-R)

I = integrate2d(np.abs(psi_test)**2 * R)

print("\\nTest:")
print("Integral =", I)
print("A_num =", 1/np.sqrt(I))
print("A_exact =", np.sqrt(8/(3*np.pi)))

def check_normalization(name, psi):

    psi_sq = np.abs(psi)**2 * R
    integral_theta = simpson(psi_sq,theta,axis=1)

    integral_total = simpson(integral_theta,r)

    print(
        f"{name}: {integral_total:.12f}"
    )
#dipole

def dipole_expectation(psi):

    mux = integrate2d(np.conjugate(psi) * X * psi * R)
    muy = integrate2d( np.conjugate(psi) * Y * psi * R )
    return mux.real, muy.real

def transition_dipole(psi_i, psi_j):

    mux = integrate2d( np.conjugate(psi_i) * X * psi_j * R)
    muy = integrate2d(np.conjugate(psi_i) * Y * psi_j * R)
    return mux.real, muy.real

psi1s_opt = psi_1s(b1)
psi2s_opt = psi_2s(b2,b1)
psi2px_opt = psi_2px(bpx)
psi2py_opt = psi_2py(bpy)

mu1s = dipole_expectation(psi1s_opt)
mu2s = dipole_expectation(psi2s_opt)
mu2px = dipole_expectation(psi2px_opt)
mu2py = dipole_expectation(psi2py_opt)

mu_1s_2s = transition_dipole(psi1s_opt,psi2s_opt)
mu_1s_2px = transition_dipole(psi1s_opt,psi2px_opt)
mu_1s_2py = transition_dipole( psi1s_opt,psi2py_opt)

print("\n====================================")
print("Stany ekscytonowe")
print("====================================")

print(
    f"1s : b={b1:.4f} "
    f"E={E1*HARTREE_TO_EV:.6f} eV")
print(
    f"Energia wiązania = {-E1*HARTREE_TO_EV:.6f} eV")
psi1s_center = psi1s_opt[0,0]
print(
    f"psi_1s(0) = {psi1s_center:.8e}")
print(
    f"2s : b={b2:.4f} "
    f"E={E2s*HARTREE_TO_EV:.6f} eV")
print(
    f"2px : b={bpx:.4f} "
    f"E={Epx*HARTREE_TO_EV:.6f} eV")
print(
    f"2py : b={bpy:.4f} "
    f"E={Epy*HARTREE_TO_EV:.6f} eV")
print("\nOczekiwana wartość dipola")
print(
    f"<1s|r|1s> = ({mu1s[0]:.6e},{mu1s[1]:.6e})")
print(
    f"<2s|r|2s> = ({mu2s[0]:.6e},{mu2s[1]:.6e})")
print(
    f"<2px|r|2px> = ({mu2px[0]:.6e},{mu2px[1]:.6e})")
print(
    f"<2py|r|2py> = ({mu2py[0]:.6e},{mu2py[1]:.6e})")
print("\nMoment dipolowy przejścia")
print(
    f"<1s|r|2s> = ({mu_1s_2s[0]:.6e}, "
    f"{mu_1s_2s[1]:.6e})")
print(
    f"<1s|r|2px> = ({mu_1s_2px[0]:.6e}, "
    f"{mu_1s_2px[1]:.6e})")
print(
    f"<1s|r|2py> = ({mu_1s_2py[0]:.6e}, "
    f"{mu_1s_2py[1]:.6e})")


psi1 = psi_1s(b1)
psi2s = psi_2s(b2,b1)
psi2px = psi_2px(bpx)
psi2py = psi_2py(bpy)

print("\nNORMALIZACJA")

check_normalization("1s",psi1)

check_normalization("2s",psi2s)

check_normalization("2px", psi2px)

check_normalization("2py",psi2py)

#################################################################
#rysowanie_wykresow
# radialna gęstość prawdopodobieństwa

P1s  = 2*np.pi * r * np.mean(np.abs(psi1s_opt)**2, axis=1)
P2s  = 2*np.pi * r * np.mean(np.abs(psi2s_opt)**2, axis=1)
P2px = 2*np.pi * r * np.mean(np.abs(psi2px_opt)**2, axis=1)
P2py = 2*np.pi * r * np.mean(np.abs(psi2py_opt)**2, axis=1)
rplot = r
import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))

plt.plot(rplot, P1s, linewidth=2.5, label="1s")
plt.plot(rplot, P2s, linewidth=2.5, label="2s")
plt.plot(rplot, P2px, linewidth=2.5, label="2px")

plt.plot(rplot, P2py,linestyle="--",linewidth=4,label="2py")

plt.xlabel("Promień r (a₀)", fontsize=16)
plt.ylabel(r"$2\pi r |\psi(r)|^2$", fontsize=16)
plt.title("Znormalizowana radialna gęstość prawdopodobieństwa ekscytonu w WS₂", fontsize=18)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.minorticks_on()

plt.grid(which="major",linestyle="-",linewidth=0.8,alpha=0.8)

plt.grid( which="minor",linestyle="--",linewidth=0.5,alpha=0.5)

plt.legend(
    [
        "Stan 1s","Stan 2s","Stan 2px","Stan 2py"
    ],
    fontsize=14
)
plt.xlim(0, 400)
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()
####################################3


#################################################################
# ==========================================================
# Gęstość prawdopodobieństwa |psi|² dla każdego stanu
# ==========================================================
states = [
    (np.abs(psi1s_opt)**2, "1s"),(np.abs(psi2s_opt)**2, "2s"),(np.abs(psi2px_opt)**2, "2px"),(np.abs(psi2py_opt)**2, "2py")
]

for rho, state in states:
    plt.figure(figsize=(8,7))
    plt.contourf(X,Y, rho,levels=100, cmap="viridis")
    plt.axis("equal")
    plt.colorbar(label=r"$|\psi|^2$")

    plt.xlabel("x (Bohr)", fontsize=14)
    plt.ylabel("y (Bohr)", fontsize=14)

    plt.title(
        f"Gęstość prawdopodobieństwa stanu {state}",
        fontsize=16
    )
    plt.xlim(-120, 120)
    plt.ylim(-120, 120)
    plt.tight_layout()
    plt.show()

#######################################################
#porownanie wszytskich 4

fig, axes = plt.subplots(2, 2,figsize=(11,8),constrained_layout=True)

states = [ (np.abs(psi1s_opt)**2, "1s"),(np.abs(psi2s_opt)**2, "2s"),(np.abs(psi2px_opt)**2, "2px"), (np.abs(psi2py_opt)**2, "2py")]

for ax, (rho, title) in zip(axes.flat, states):
    # ciemne tło poza obszarem danych
    ax.set_facecolor("black")
    contour = ax.contourf(X,  Y, rho, levels=100, cmap="plasma")

    ax.set_title( f"Stan {title}", fontsize=16)

    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)

    ax.set_aspect("equal")
# wspólne osie
fig.supxlabel("x (Bohr)", fontsize=14)
fig.supylabel("y (Bohr)", fontsize=14)

cbar = fig.colorbar(contour, ax=axes.ravel().tolist(),location="right", fraction=0.03, pad=0.04)
cbar.set_label( r"$|\psi|^2$", fontsize=14)

fig.suptitle(
    "Gęstość prawdopodobieństwa ekscytonu w WS₂",
    fontsize=20)

plt.show()

########################################
#WYKRESY 3D

states = [
    (np.abs(psi1s_opt)**2, "1s"),(np.abs(psi2s_opt)**2, "2s"),(np.abs(psi2px_opt)**2, "2px"),(np.abs(psi2py_opt)**2, "2py")]

for rho, state in states:
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot( 111, projection="3d")
    step = 1
    ax.plot_surface( X[::step,::step], Y[::step,::step], rho[::step,::step], cmap="viridis", edgecolor="none")
    ax.set_xlabel("x (Bohr)")
    ax.set_ylabel("y (Bohr)")
    ax.set_zlabel(r"$|\psi|^2$")
    ax.set_title(
        f"Gęstość prawdopodobieństwa stanu {state}"
    )
    plt.tight_layout()
    plt.show()
    ############################

# PORÓWNANIE 3D

fig = plt.figure(figsize=(18,5))
states = [(np.abs(psi1s_opt)**2, "1s"),(np.abs(psi2s_opt)**2, "2s"),(np.abs(psi2px_opt)**2, "2px"),(np.abs(psi2py_opt)**2, "2py")]

for i, (rho, title) in enumerate(states, start=1):

    ax = fig.add_subplot(1, 4, i, projection="3d")
    ax.plot_surface( X,Y,rho, cmap="plasma",edgecolor="none",antialiased=True)

    ax.set_title(f"Stan {title}",fontsize=16,pad=5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.view_init(
        elev=25,
        azim=-60)
    ax.set_box_aspect([1,1,0.6])

fig.suptitle(
    "Trójwymiarowa gęstość prawdopodobieństwa ekscytonu w WS$_2$",
    fontsize=20,
    y=0.95
)
plt.subplots_adjust( left=0.01, right=0.99,bottom=0.02,top=0.88,wspace=0.02)
plt.show()

############### 3d sama f falowa
fig = plt.figure(figsize=(18,5))
states = [ (psi1s_opt, "1s"), (psi2s_opt, "2s"), (psi2px_opt, "2px"), (psi2py_opt, "2py")]

for i, (psi, title) in enumerate(states, start=1):
    ax = fig.add_subplot(1, 4, i,  projection="3d")
    vmax = np.max(np.abs(psi))
    ax.plot_surface(X, Y, psi,cmap="plasma",    edgecolor="none",antialiased=True, vmin=-vmax, vmax=vmax)
    ax.set_title( rf"$\psi_{{{title}}}(x,y)$",fontsize=16, pad=5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.view_init(  elev=25, azim=-60 )
    ax.set_box_aspect([1,1,0.6])

fig.suptitle(
    "Funkcje falowe ekscytonu w WS$_2$",fontsize=20,y=0.95)

plt.subplots_adjust(left=0.01,right=0.99,bottom=0.02,top=0.88,wspace=0.02)

plt.show()

#########################################################
#podsumowanie dashboard


fig = plt.figure(figsize=(18,13))
gs = GridSpec(
    4, 5,
    figure=fig,
    width_ratios=[1,1,1,1,1.4],
    height_ratios=[1.0,1.0,1.0,0.35]
)


states_rho = [ (np.abs(psi1s_opt)**2, "1s"),(np.abs(psi2s_opt)**2, "2s"),(np.abs(psi2px_opt)**2, "2px"),(np.abs(psi2py_opt)**2, "2py")]

states_psi = [ (psi1s_opt, "1s"),(psi2s_opt, "2s"),(psi2px_opt, "2px"),(psi2py_opt, "2py")]
#r 1
for i, (rho, title) in enumerate(states_rho):
    ax = fig.add_subplot( gs[0,i], projection="3d")

    ax.plot_surface(  X, Y,  rho, cmap="plasma", edgecolor="none")

    ax.set_title(  rf"$|\psi_{{{title}}}|^2$", fontsize=14)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.view_init(elev=25, azim=-60)
    ax.set_box_aspect([1,1,0.5])
#r_2
for i, (psi, title) in enumerate(states_psi):

    ax = fig.add_subplot(gs[1,i],projection="3d")
    vmax = np.max(np.abs(psi))
    ax.plot_surface( X, Y, psi,cmap="plasma",edgecolor="none", vmin=-vmax, vmax=vmax)

    ax.set_title(rf"$\psi_{{{title}}}$", fontsize=14)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.view_init(elev=25,azim=-60)
    ax.set_box_aspect([1,1,0.5])
#r3b
for i, (rho, title) in enumerate(states_rho):
    ax = fig.add_subplot(gs[2,i])
    contour = ax.contourf( X, Y, rho, levels=100, cmap="plasma")
    ax.set_title(  rf"$|\psi_{{{title}}}|^2$", fontsize=13
    )
    ax.set_xlim(-120,120)
    ax.set_ylim(-120,120)
    ax.set_aspect("equal")
    ax.set_xlabel("x (Bohr)")
    ax.set_ylabel("y (Bohr)")
#_p kolumna
axr = fig.add_subplot(gs[0:3,4])
axr.plot( rplot,P1s, linewidth=2.5, label="1s")
axr.plot(rplot,P2s,linewidth=2.5,label="2s")

axr.plot(rplot,P2px,linewidth=2.5,label="2px"
)

axr.plot(rplot,P2py,"--",linewidth=2.5,label="2py")

axr.set_title( "Radialna gęstość\nprawdopodobieństwa", fontsize=16
)

axr.set_xlabel("r (Bohr)")

axr.set_ylabel( r"$2\pi r |\psi(r)|^2$")

axr.grid( alpha=0.3)

axr.legend()
#tabela

ax_table = fig.add_subplot(gs[3,:])
ax_table.axis("off")

table_data = [[ f"{E1*HARTREE_TO_EV:.4f}",f"{E2s*HARTREE_TO_EV:.4f}", f"{Epx*HARTREE_TO_EV:.4f}", f"{Epy*HARTREE_TO_EV:.4f}", f"{-E1*HARTREE_TO_EV:.4f}"]]

table = ax_table.table( cellText=table_data, colLabels=[  "1s (eV)",  "2s (eV)",  "2px (eV)",  "2py (eV)",  "Energia wiązania (eV)" ],cellLoc="center", loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1,1.5)
#tytuk
fig.suptitle("WS$_2$ – stany ekscytonowe wyznaczone metodą wariacyjną",fontsize=12)
plt.tight_layout()
plt.show()
