import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

plt.rcParams["font.family"] = "Times New Roman"


def lorentzian(x, x0, gamma, A):
    return A * (gamma**2 / ((x - x0)**2 + gamma**2))

def wiele_lorentzow(x, *params):

    n = (len(params)-2)//3
    y = np.zeros_like(x)

    for i in range(n):

        x0=params[3*i]
        gamma=params[3*i + 1]
        A=params[3*i + 2]

        y += lorentzian(x, x0, gamma, A)
    a =params[-2]
    b= params[-1]

    return y + a*x + b

#funkcja liczaca pole pod pikami
def pole_lorentza(A, gamma):
    return np.pi*A*gamma

with open("/workspaces/analiza/OLA/MOC/xc.txt", "r") as fx:
    x_data = np.array([float(line.strip()) for line in fx])

pliki = [
    "/workspaces/analiza/OLA/MOC/1a.txt", "/workspaces/analiza/OLA/MOC/2a.txt", "/workspaces/analiza/OLA/MOC/3a.txt", "/workspaces/analiza/OLA/MOC/4a.txt", "/workspaces/analiza/OLA/MOC/5a.txt", "/workspaces/analiza/OLA/MOC/6a.txt", "/workspaces/analiza/OLA/MOC/7a.txt", "/workspaces/analiza/OLA/MOC/8a.txt"
]

#os x (moce laserow)
moc = np.array([
    1270,
    930,
    560,
    400,
    260,
    142,
    11.5,
    2
])

pole_ix1 = []
pole_ix2 = []

#to bylo w programach wczesniejszych, że szukamy ile pików
for plik in pliki:
    print("\n===================================")
    print("Analiza:", plik)

    with open(plik, "r") as fy:
        y_data = np.array([float(line.strip()) for line in fy])

    y_smooth = savgol_filter(y_data, 31, 3)

    #szukanie pików
    peaks, properties = find_peaks(
        y_smooth,
        prominence=np.max(y_smooth)*0.08,
        distance=120,
        width=15
    )

    #sa 2 to tu sobie mozemy uzyc takiego warunku
    if len(peaks) > 2:

        peak_heights = y_smooth[peaks]
        idx_sorted = np.argsort(peak_heights)[-2:]
        peaks = peaks[idx_sorted]
    peaks = peaks[np.argsort(x_data[peaks])]

    print("Liczba pików:", len(peaks))
    #dalsza czesc_charakterystyka_tych_pikow_tez bylo

    initial_guess = []

    for peak in peaks:

        x0 = x_data[peak]
        A = y_data[peak]
        gamma = 0.02
        initial_guess.extend([x0, gamma, A])

    # tło
    initial_guess.extend([0, 0])

    xmin = np.min(x_data)
    xmax = np.max(x_data)

    lower_bounds = []
    upper_bounds = []

    for peak in peaks:

        lower_bounds.extend([xmin, 0.001, 0])
        upper_bounds.extend([xmax, 0.1, np.max(y_data)*10])

    lower_bounds.extend([-np.inf, -np.inf])
    upper_bounds.extend([ np.inf,  np.inf])

    popt, pcov = curve_fit(
        wiele_lorentzow,
        x_data,
        y_data,
        p0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        maxfev=50000
    )

    n_peaks = (len(popt)-2)//3
   #dane_wyswietlane
    pola = []

    for i in range(n_peaks):

        x0 = popt[3*i]
        gamma = popt[3*i + 1]
        A = popt[3*i + 2]
        pole = pole_lorentza(A, gamma)
        pola.append(pole)

        print(f"\nPIK {i+1}")
        print(f"x0 = {x0:.5f}")
        print(f"gamma = {gamma:.5f}")
        print(f"A = {A:.2f}")
        print(f"Pole = {pole:.2f}")

    pole_ix1.append(pola[0])
    pole_ix2.append(pola[1])

#interesuje nas zaleznosc liniowa, wiec tylko do 3 ostatnich (pierwszych),
# ale mamy odwrotna kolejnsoc
#robimy regresje liniowa, aby wyznaczyc alfa dla IX1 i IX2 + SKALA LOG10
N = 3
x_fit = np.log10(moc[-N:])

y_fit_ix1 = np.log10(pole_ix1[-N:])
y_fit_ix2 = np.log10(pole_ix2[-N:])

wspolczynnik1 = np.polyfit(x_fit, y_fit_ix1, 1) #gotowa_funkcja
wspolczynnik2 = np.polyfit(x_fit, y_fit_ix2, 1)

alpha1 = wspolczynnik1[0]
alpha2 = wspolczynnik2[0]
#zadanie_zaleznosci_liniowej
x_line = np.linspace(moc[-N], moc[-1], 200)

y_line1 = 10**(wspolczynnik1[0]*np.log10(x_line) + wspolczynnik1[1])
y_line2 = 10**(wspolczynnik2[0]*np.log10(x_line) + wspolczynnik2[1])

plt.figure(figsize=(5,7))
plt.scatter(
    moc, pole_ix1, color='blue', s=70,label=f'IX1   α1={alpha1:.2f}'
)
plt.scatter(
    moc, pole_ix2, color='red', s=70,label=f'IX2   α2={alpha2:.2f}'
)
plt.plot(
    x_line, y_line1, '--', color='blue', linewidth=2
)
plt.plot(
    x_line, y_line2, '--', color='red', linewidth=2
)
plt.tick_params(axis='both', labelsize=16)
# SKALA LOG10
plt.xscale('log')
plt.yscale('log')

plt.xlabel(r"Gęstość mocy ($\mu W/cm^2$)", fontsize=16)
plt.ylabel("Intensywność PL (a.u.)", fontsize=16)
plt.grid(which='major', linestyle='-', linewidth=0.5, color='black')
plt.grid(which='minor', linestyle='--', linewidth=0.5, color='grey')

plt.legend(fontsize=18)
plt.tight_layout()
plt.show()
#wyswietlanie danych
print("\n===================================")
print("WYNIKI")
print("===================================")

print("\nIX1:")
for p in pole_ix1:
    print(p)

print("\nIX2:")
for p in pole_ix2:
    print(p)

print(f"\nalpha IX1 = {alpha1:.3f}")
print(f"alpha IX2 = {alpha2:.3f}")

plt.savefig("wykres11.png")
print("Zapisano wykres")
