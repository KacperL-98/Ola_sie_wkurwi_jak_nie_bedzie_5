import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def lorentzian(x, x0, gamma, A):
    return A * (gamma**2 / ((x - x0)**2 + gamma**2))


def triple_lorentzian(x,
                     x01, g1, A1,
                     x02, g2, A2,
                     x03, g3, A3,
                     a, b):

    return (
        lorentzian(x, x01, g1, A1) +
        lorentzian(x, x02, g2, A2) +
        lorentzian(x, x03, g3, A3) +
        a * x + b
    )


with open("/workspaces/Ola_sie_wkurwi_jak_nie_bedzie_5/OLA/x1.txt", "r") as fx:
    x_data = np.array([float(line.strip()) for line in fx])

with open("/workspaces/Ola_sie_wkurwi_jak_nie_bedzie_5/OLA/y1.txt", "r") as fy:
    y_data = np.array([float(line.strip()) for line in fy])


initial_guess = [
    1.82, 0.03, 40000,
    1.90, 0.03, 20000,
    2.01, 0.02, 80000,
    0, 0
]


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

plt.figure(figsize=(8, 5))

plt.scatter(x_data, y_data, s=15, label='Dane')
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(x_data, y_fit, 'r', linewidth=2, label='Dopasowanie')
xmin=1.25
xmax = 2.2
ymin = -100
ymax=80000
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='black')
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth=0.5, color='black')
plt.grid(which='minor', linestyle='--', linewidth=0.5, color='grey')
plt.legend(fontsize=14)

plt.xlabel("Energia (eV)",fontsize=14)
plt.ylabel("Intensywność fotoluminescencji (a. u.)",fontsize=14)
plt.title("Widmo fotoluminescencji struktury nr 1 wraz z dopasowanie funkcją Lorentza",fontsize=14)


plt.grid(True)

plt.show()
plt.savefig("wykres.png")
print("Zapisano wykres")