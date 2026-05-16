import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



def lorentzian(x, x0, gamma, A):
    return A * (gamma**2 / ((x - x0)**2 + gamma**2))


def double_lorentzian(x,
                      x01, g1, A1,
                      x02, g2, A2,
                      a, b):

    return (
        lorentzian(x, x01, g1, A1) +
        lorentzian(x, x02, g2, A2) +
        a * x + b
    )

with open("/workspaces/analiza/OLA/x1_2.txt", "r") as fx:
    x_data = np.array([float(line.strip()) for line in fx])

with open("/workspaces/analiza/OLA/x2_2.txt", "r") as fy:
    y_data = np.array([float(line.strip()) for line in fy])


initial_guess = [
    1.72, 0.03, 90000,
    1.99, 0.025, 95000,
    0, 0
]


popt, pcov = curve_fit(
    double_lorentzian,
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
print(f"FWHM1  = {2*popt[1]}")

print("\nLorentz 2:")
print(f"x02    = {popt[3]}")
print(f"gamma2 = {popt[4]}")
print(f"A2     = {popt[5]}")
print(f"FWHM2  = {2*popt[4]}")

print("\nTło:")
print(f"a = {popt[6]}")
print(f"b = {popt[7]}")

# =========================
# KRZYWE
# =========================

y_fit = double_lorentzian(x_data, *popt)

y1 = lorentzian(x_data, popt[0], popt[1], popt[2])
y2 = lorentzian(x_data, popt[3], popt[4], popt[5])

background = popt[6] * x_data + popt[7]

plt.figure(figsize=(8,5))

plt.scatter(
    x_data,
    y_data,
    s=5,
    color='blue',
    label='Dane pomiarowe'
)

plt.plot(
    x_data,
    y_fit,
    'r',
    linewidth=2,
    label='Dopasowanie Lorentzem'
)


plt.plot(x_data, y1, '--', linewidth=2, label='Dopasowanie dla WS$_{0.5}$Se$_{1.5}$', color="green")
plt.plot(x_data, y2, '--', linewidth=2, label='Dopasowanie dla WS$_2$', color="brown")


plt.xlim(1.3, 2.2)
plt.ylim(-100, 110000)


plt.xlabel("Energia (eV)", fontsize=16)
plt.ylabel("Intensywność fotoluminescencji (a. u.)", fontsize=16)

plt.title(
    "Widmo fotoluminescencji wraz z dopasowaniem funkcją Lorenza",
    fontsize=14
)

# ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


plt.minorticks_on()

plt.grid(
    which='major',
    linestyle='-',
    linewidth=0.5,
    color='black'
)

plt.grid(
    which='minor',
    linestyle='--',
    linewidth=0.5,
    color='grey'
)

plt.legend(fontsize=12)

plt.show()
plt.savefig("wykres2.png")
print("Zapisano wykres")
