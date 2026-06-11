import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import simpson
from scipy.optimize import minimize_scalar
from scipy.special import j0

plt.rcParams["font.family"] = "Times New Roman"

HA_TO_EV=27.2114
BOHR_TO_ANG=0.529177
#parametry
eps_hbn=4.9
eps_ws2=6.12
eps_wse2=6.48
#z poscar
d_ws2_ang  = 3.098
d_wse2_ang = 3.414
#srodek warstw
z_ws2  = 3.463
z_wse2 = 10.393

# odległość międzywarstwowa
d_intra_ws2=abs(z_ws2-z_ws2)
d_intra_wse2=abs(z_wse2-z_wse2)
d_inter_ang=abs(z_wse2-z_ws2)
#masy
me_ws2_K=0.27
mh_ws2_K=0.32
me_wse2_K=0.29
mh_wse2_K=0.36

def mu(me,mh):
    return me*mh/(me+mh)

mu_ws2 =mu(me_ws2_K,mh_ws2_K)
mu_wse2=mu(me_wse2_K,mh_wse2_K)
mu_inter=mu(me_wse2_K,mh_ws2_K)
#siatka
r_max=400.0
Nr=400
Ntheta= 360
r = np.linspace(1e-12,r_max,Nr)
theta = np.linspace(0,2*np.pi,Ntheta)

R,Theta = np.meshgrid(r,theta,indexing="ij")

X=R*np.cos(Theta)
Y=R*np.sin(Theta)

dr =r[1]-r[0]
dtheta= theta[1]-theta[0]

# MODEL εeff(q)

def gamma_plus(q):

    return ((1+np.exp(-2*d_wse2_ang*q))*eps_wse2+(1-np.exp(-2*d_wse2_ang*q))*eps_hbn)

def gamma_minus(q):

    return ((1-np.exp(-2*d_wse2_ang*q))*eps_wse2+(1+np.exp(-2*d_wse2_ang*q))*eps_hbn)

def A_q(q):

    gp = gamma_plus(q)
    gm = gamma_minus(q)

    return ((1-np.exp(-2*d_ws2_ang*q))*( eps_ws2**2*gp+eps_hbn*eps_wse2*gm)+(1+np.exp(-2*d_ws2_ang*q))
        *( eps_hbn*eps_ws2*gp + eps_ws2*eps_wse2*gm))

def B_q(q):

    term1 = ( (1-np.exp(-d_ws2_ang*q))*eps_hbn+(1+np.exp(-d_ws2_ang*q))*eps_ws2)
    term2 = ( (1+np.exp(-d_wse2_ang*q))*eps_wse2 + (1-np.exp(-d_wse2_ang*q))*eps_hbn)

    return 2*term1*term2

def eps_eff(q):

    return A_q(q)/B_q(q)
#potencjaly

def V_real(rval,d_ang):

    integrand = lambda q: (np.exp(-q*d_ang) * j0(q*rval) / eps_eff(q))

    val,_ = quad(  integrand, 1e-10,  20.0,  limit=300 )
    return -val

def build_potential(d_ang):

    V = np.zeros_like(r)
    for i,rr in enumerate(r):
        V[i] = V_real(rr,d_ang)
    return V

print("Liczenie potencjałów...")

V_ws2_r   = build_potential(d_intra_ws2)
V_wse2_r  = build_potential(d_intra_wse2)
V_inter_r = build_potential(d_inter_ang)
#interpolacja

def Vgrid_from_radial(Vr):
    return np.interp(R,r,Vr)
# narzedzia numeryczne

def integrate2d(f):

    tmp = simpson(  f, theta,  axis=1 )
    return simpson( tmp,  r)

def normalize(psi):

    rho = np.abs(psi)**2 * R

    I_theta = simpson( rho, theta, axis=1 )
    I = simpson(  I_theta,  r)
    return psi/np.sqrt(I)

#laplasjan

def laplacian(psi):

    dpsi_dr = np.zeros_like(psi)

    dpsi_dr[1:-1,:] = ( psi[2:,:]-psi[:-2,:] )/(2*dr)

    term_r = np.zeros_like(psi)

    term_r[1:-1,:] = ( (R[2:,:]*dpsi_dr[2:,:] - R[:-2,:]*dpsi_dr[:-2,:] )/(2*dr))/R[1:-1,:]

    term_theta = np.zeros_like(psi)

    term_theta[:,1:-1] = (psi[:,2:]- 2*psi[:,1:-1] + psi[:,:-2])/dtheta**2

    lap = term_r + term_theta/R**2
    return lap
#f_probne

def psi_1s(b):

    return normalize(  np.exp(-R/b))

def psi_2s(b2,b1):

    d = (b1+b2)/(2*b1)

    return normalize((1-d*R/b2)*np.exp(-R/b2))

def psi_2px(b):

    return normalize(  R*np.cos(Theta)  *np.exp(-R/b))

def psi_2py(b):

    return normalize(R*np.sin(Theta)*np.exp(-R/b))

#solver wariacyjny
class ExcitonSolver:

    def __init__(self,mu_eff,Vr):
        self.mu = mu_eff
        self.Vgrid = Vgrid_from_radial(Vr)

    def energy(self,psi):

        lap= laplacian(psi)
        Tpsi= -(1/(2*self.mu))*lap
        Hpsi = Tpsi + self.Vgrid*psi

        return (integrate2d(np.conjugate(psi) *Hpsi*  R)/integrate2d(np.conjugate(psi)*psi*R)).real

    def solve(self):
        # 1s
        res1 = minimize_scalar(lambda b: self.energy(   psi_1s(b)),bounds=(0.001, 300),method="bounded")

        b1 = res1.x
        E1 = res1.fun
        # 2s
        res2s = minimize_scalar(lambda b2:self.energy( psi_2s(b2, b1)),bounds=(0.001, 300),method="bounded")

        b2 = res2s.x
        E2s = res2s.fun
        # 2px

        respx = minimize_scalar(lambda b:self.energy( psi_2px(b)), bounds=(0.001, 300), method="bounded")

        bpx = respx.x
        Epx = respx.fun
        # 2py
        respy = minimize_scalar(lambda b:self.energy(psi_2py(b) ), bounds=(0.001, 300), method="bounded")

        bpy = respy.x
        Epy = respy.fun

        # energia wiązania
        Es_px = Epx - E1
        Es_py = Epy - E1

        Ebind = 9.0 * (0.5 * Es_px + 0.5 * Es_py) / 8.0

        # funkcje falowe

        psi1s_opt = psi_1s(b1)
        psi2s_opt = psi_2s(b2, b1)

        psi2px_opt = psi_2px(bpx)
        psi2py_opt = psi_2py(bpy)

        return { "E1s": E1 * HA_TO_EV, "E2s": E2s * HA_TO_EV, "E2px": Epx * HA_TO_EV, "E2py": Epy * HA_TO_EV,"Ebind": Ebind * HA_TO_EV,
"b1": b1,"b2": b2, "bpx": bpx,"bpy": bpy, "psi1s": psi1s_opt, "psi2s": psi2s_opt, "psi2px": psi2px_opt, "psi2py": psi2py_opt}

# URUCHOMIENIE

solver_ws2 = ExcitonSolver( mu_ws2, V_ws2_r)

solver_wse2 = ExcitonSolver( mu_wse2, V_wse2_r)

solver_inter = ExcitonSolver(mu_inter,V_inter_r)

res_ws2 = solver_ws2.solve()
res_wse2 = solver_wse2.solve()
res_inter = solver_inter.solve()

from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

systems = [

    ("WS$_2$", res_ws2),("WSe$_2$", res_wse2),("Interlayer Exciton", res_inter)
]

for system_name, res in systems:

    psi1s_opt = res["psi1s"]
    psi2s_opt = res["psi2s"]
    psi2px_opt = res["psi2px"]
    psi2py_opt = res["psi2py"]

    E1s = res["E1s"]
    E2s = res["E2s"]
    Epx = res["E2px"]
    Epy = res["E2py"]
    Ebind = res["Ebind"]

    b1=res["b1"]
    b2=res["b2"]
    bpx=res["bpx"]
    bpy=res["bpy"]

    rho1s=np.abs(psi1s_opt)**2
    rho2s=np.abs(psi2s_opt)**2
    rho2px=np.abs(psi2px_opt)**2
    rho2py=np.abs(psi2py_opt)**2

    P1s = r*simpson(rho1s, theta, axis=1)
    P2s = r*simpson(rho2s, theta, axis=1)
    P2px = r*simpson(rho2px, theta, axis=1)
    P2py = r*simpson(rho2py, theta, axis=1)

    states_rho = [
        (rho1s, "1s"),(rho2s, "2s"), (rho2px, "2px"), (rho2py, "2py")
    ]

    states_psi = [
        (psi1s_opt, "1s"),(psi2s_opt, "2s"),(psi2px_opt, "2px"),(psi2py_opt, "2py")
    ]

    fig = plt.figure(figsize=(18,13))

    gs = GridSpec(4,5,figure=fig,width_ratios=[1,1,1,1,1.4],height_ratios=[1,1,1,0.4])

    # |psi|² 3D

    for i,(rho,title) in enumerate(states_rho):

        ax = fig.add_subplot(gs[0,i], projection="3d")

        ax.plot_surface( X,Y,rho, cmap="plasma", edgecolor="none")

        ax.set_title(rf"$|\psi_{{{title}}}|^2$")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.view_init(25,-60)

    # psi 3D

    for i,(psi,title) in enumerate(states_psi):

        vmax = np.max(np.abs(psi))

        ax = fig.add_subplot(gs[1,i],projection="3d" )

        ax.plot_surface( X,Y,psi,cmap="plasma",edgecolor="none",vmin=-vmax,vmax=vmax)

        ax.set_title(  rf"$\psi_{{{title}}}$" )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.view_init(25,-60)

    # mapy 2D

    for i,(rho,title) in enumerate(states_rho):

        ax = fig.add_subplot(gs[2,i])

        contour = ax.contourf(X, Y, rho, levels=100, cmap="plasma")

        ax.set_title(rf"$|\psi_{{{title}}}|^2$")

        ax.set_xlim(-120,120)
        ax.set_ylim(-120,120)

        ax.set_aspect("equal")

    # radialna gęstość

    axr = fig.add_subplot(gs[0:3,4])

    axr.plot(r,P1s,lw=3,label="1s")
    axr.plot(r,P2s,lw=3,label="2s")
    axr.plot(r,P2px,lw=3,label="2px")
    axr.plot(r,P2py,"--",lw=3,label="2py")
    axr.set_ylim(bottom=0)
    axr.set_title(
        "Radialna gęstość\nprawdopodobieństwa",
        fontsize=16
    )

    axr.set_xlabel("r (Bohr)")
    axr.set_ylabel(r"$r\int |\psi|^2 d\theta$")

    axr.grid(alpha=0.3)
    axr.legend()
    # tabela

    ax_table = fig.add_subplot(gs[3,:])

    ax_table.axis("off")

    table_data = [[

        f"{E1s:.4f}", f"{E2s:.4f}",f"{Epx:.4f}",f"{Epy:.4f}",f"{Ebind:.4f}",f"{b1:.2f}",f"{b2:.2f}",f"{bpx:.2f}",f"{bpy:.2f}"
    ]]

    table = ax_table.table(

        cellText=table_data,
        colLabels=[

            "E1s (eV)", "E2s (eV)", "E2px (eV)", "E2py (eV)", "Ebind (eV)", "b1", "b2", "bpx", "bpy"],
        cellLoc="center",
        loc="center"

    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1,1.5)

    fig.suptitle(

        f"{system_name}\n",
        fontsize=18

    )
    plt.tight_layout()
    plt.show()
