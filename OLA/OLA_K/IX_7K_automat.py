import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter #automatyczne znajdowanie pików
#wygładzanie danych

#plt.rcParams["font.family"] = "Times New Roman"

def lorentzian(x, x0, gamma, A):
    return A * (gamma**2 / ((x - x0)**2 + gamma**2))

#funkcja buduje sumę dowolnej liczby funkcji Lorentza + liniowe tło
def wiele_lorentzow(x, *params):

    n =(len(params)-2)//3 #liczy_ile_pikow

    y = np.zeros_like(x)
    for i in range(n):

        x0= params[3*i]
        gamma=params[3*i+1]
        A=params[3*i+2]

        y += lorentzian(x, x0, gamma, A)
    a=params[-2]
    b=params[-1]

    return y+a*x+b

with open("/workspaces/Ola_sie_wkurwi_jak_nie_bedzie_5/OLA/x.txt", "r") as fx:
    x_data = np.array([float(line.strip()) for line in fx])

with open("/workspaces/Ola_sie_wkurwi_jak_nie_bedzie_5/OLA/6.txt", "r") as fy:
    y_data = np.array([float(line.strip()) for line in fy])

mask = (x_data > 1.35) & (x_data < 1.70)

x_data = x_data[mask]
y_data = y_data[mask]

y_smooth = savgol_filter(y_data, 31, 3)

#służy do znajdowania maksimów lokalnych w sygnale
peaks, properties = find_peaks(
    y_smooth,
    prominence=np.max(y_smooth)*0.12, #jak bardzo pik wystaje ponad otoczenie
    distance=120,
    width=15
)

print("\nLiczba znalezionych pików:", len(peaks))

initial_guess = []

for peak in peaks:

    x0 = x_data[peak]
    A = y_data[peak]
    gamma = 0.02
    initial_guess += [x0, gamma, A]

initial_guess += [0, 0]
initial_guess = []

for peak in peaks:

    x0 = x_data[peak]
    A = y_data[peak]
    gamma = 0.02
    initial_guess.extend([x0, gamma, A])

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


print("initial_guess =", len(initial_guess))
print("lower_bounds  =", len(lower_bounds))
print("upper_bounds  =", len(upper_bounds))


popt, pcov = curve_fit(
    wiele_lorentzow,
    x_data,
    y_data,
    p0=initial_guess,
    bounds=(lower_bounds, upper_bounds),
    maxfev=50000
)

n_peaks = (len(popt)-2)//3

print("\n=== PARAMETRY DOPASOWANIA ===\n")

for i in range(n_peaks):

    x0 = popt[3*i]
    gamma = popt[3*i + 1]
    A = popt[3*i + 2]

    print(f"PIK {i+1}")
    print(f"Pozycja = {x0:.6f} eV")
    print(f"Gamma   = {gamma:.6f}")
    print(f"FWHM    = {2*gamma:.6f} eV")
    print(f"Amplituda = {A:.2f}")
    print()


y_fit = wiele_lorentzow(x_data, *popt)

#tworzeie wykresu
plt.figure(figsize=(9,6))

plt.scatter(
    x_data,
    y_data,
    s=12,
    label='Dane pomiarowe'
)

# fit
plt.plot(
    x_data,
    y_fit,
    'r',
    linewidth=2.5,
    label='Dopasowanie funkcją Lorentza'
)
colors = ["magenta", "orange", "blue", "orange", "purple"]
for i in range(n_peaks):

    y_single = lorentzian(
        x_data, popt[3*i], popt[3*i + 1], popt[3*i + 2]
    )
    plt.plot(
        x_data, y_single, '--', linewidth=2,
        label=f'Pik {i+1}',
        color=colors[i]

    )

#dodatek_pokazuje_gdzie_sa
plt.plot(
    x_data[peaks],
    y_smooth[peaks],
    ".",
    color='black',
    markersize=18,
    label='Wykryte piki'
)
plt.vlines(
    x_data[peaks],
    ymin=0,
    ymax=y_smooth[peaks],
    colors='red',
    linestyles='dashed',
    linewidth=1.5
)

plt.xlabel("Energia (eV)", fontsize=16)
plt.ylabel("Intensywność PL (a.u.)", fontsize=16)

plt.title(
    "Dopasowanie pików funkcją Lorentza bez podawania liczby maksimów (automat)",
    fontsize=16
)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.minorticks_on()

plt.grid(which='major', linestyle='-', linewidth=0.5, color='black')
plt.grid(which='minor', linestyle='--', linewidth=0.5, color='grey')
plt.legend(fontsize=14)
#xmin=1.36
#xmax = 1.7
#ymin = -10
#ymax=900


#plt.xlim(xmin, xmax)
#plt.ylim(ymin, ymax)


plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
plt.savefig("wykres9.png")
print("Zapisano wykres")