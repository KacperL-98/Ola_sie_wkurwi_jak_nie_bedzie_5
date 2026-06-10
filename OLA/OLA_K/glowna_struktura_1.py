import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.rcParams["font.family"] = "Times New Roman"
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
def lorentzian(x, x0, gamma, A):
    return A * (gamma**2 / ((x - x0)**2 + gamma**2))


def triple_lorentzian(x,
                     x01, g1, A1,
                     x02, g2, A2,
                     x03, g3, A3,
                     a, b):

    return (
        lorentzian(x,x01,g1,A1)+
        lorentzian(x,x02,g2,A2)+
        lorentzian(x,x03,g3,A3)+
        a*x+b
    )

with open("/workspaces/Ola/OLA/xg.txt", "r") as fx:
    x_data = np.array([float(line.strip()) for line in fx])

with open("/workspaces/Ola/OLA/yg.txt", "r") as fy:
    y_data = np.array([float(line.strip()) for line in fy])


initial_guess = [
    1.37, 0.07, 1000,   # IX
    1.72, 0.05, 700,    # WS0.5Se0.5
    2.003, 0.02, 900,    # WS2
    0, 100              # tło
]
lower_bounds = [
    1.45, 0.01, 500,
    1.80, 0.005, 200,
    1.98, 0.005, 5000,
    -10000, -10000
]

upper_bounds = [
    1.55, 0.10, 5000,
    1.90, 0.08, 5000,
    2.03, 0.05, 30000,
    10000, 10000
]
mask = (x_data > 1.75) & (x_data < 1.92)

x_fit = x_data[mask]
y_fit_data = y_data[mask]
popt, pcov = curve_fit(
    triple_lorentzian,
    x_data,
    y_data,
    p0=initial_guess,
    maxfev=20000
)


print("\n=== PARAMETRY DOPASOWANIA ===\n")

print("Lorentz 1:")
print(f"x01    = {popt[0]}")
print(f"gamma1 = {popt[1]}")
print(f"A1     = {popt[2]}")

print("\nLorentz 2:")
print(f"x02    = {popt[3]}")
print(f"gamma2 = {popt[4]}")
print(f"A2     = {popt[5]}")

print("\nLorentz 3:")
print(f"x03    = {popt[6]}")
print(f"gamma3 = {popt[7]}")
print(f"A3     = {popt[8]}")

print("\nTło:")
print(f"a = {popt[9]}")
print(f"b = {popt[10]}")
gamma1 = popt[1]
gamma2 = popt[4]
gamma3 = popt[7]
print(f"FWHM1 = {2*popt[1]}")
print(f"FWHM2 = {2*popt[4]}")
print(f"FWHM3 = {2*popt[7]}")

y_fit = triple_lorentzian(x_data, *popt)

y1 = lorentzian(x_data, popt[0], popt[1], popt[2])
y2 = lorentzian(x_data, popt[3], popt[4], popt[5])
y3 = lorentzian(x_data, popt[6], popt[7], popt[8])
background = popt[9] * x_data + popt[10]

plt.figure(figsize=(7,5))

fig, ax = plt.subplots(figsize=(7,5))


ax.scatter(x_data, y_data, s=15, label='Dane pomiarowe')
ax.plot(x_data, y_fit, 'r', linewidth=2, label='Dopasowanie Lorentzem')


ax.set_xlim(1.3, 2.2)
ax.set_ylim(-10, 40000)


ax.minorticks_on()
ax.grid(which='major', linestyle='-', linewidth=0.5, color='black')
ax.grid(which='minor', linestyle='--', linewidth=0.5, color='grey')

# opisy
ax.set_xlabel("Energia (eV)", fontsize=14)
ax.set_ylabel("Intensywność fotoluminescencji (a.u.)", fontsize=14)

ax.legend(fontsize=12)

#WYKRES W WYKRESIE


axins = inset_axes(
    ax,
    width="50%",
    height="50%",
    bbox_to_anchor=(0.1, 0.27, 1, 1),
    bbox_transform=ax.transAxes,
    loc='lower left'
)

axins.scatter(x_data, y_data, s=10)
axins.plot(x_data, y_fit, 'r', linewidth=1.5)


x1, x2 = 1.3, 1.50
y1_zoom, y2_zoom = 0, 2500

axins.set_xlim(x1, x2)
axins.set_ylim(y1_zoom, y2_zoom)


axins.minorticks_on()
axins.grid(True, linestyle='--', linewidth=0.4)


mark_inset(ax, axins, loc1=2, loc2=4,
           fc="none", ec="black", linestyle="--")
plt.show()
plt.savefig("wykres7.png")
print("Zapisano wykres")
