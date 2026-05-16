import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

plt.rcParams["font.family"] = "Times New Roman"
#znow Lorentz_do_fitowania
def lorentzian(x, x0, gamma, A):
    return A*(gamma**2/((x-x0)**2+gamma**2))
#i ponownie funkcja, ktora podaje wiele Lorenztow
def wiele_lorentzow(x, *params):
    n = (len(params)-2)//3
    y = np.zeros_like(x)

    for i in range(n):
        x0=params[3*i]
        gamma=params[3*i+1]
        A=params[3*i+2]
        y+=lorentzian(x, x0, gamma, A)

    a = params[-2]
    b = params[-1]
    return y + a*x + b

#i znow liczymy pole pod wykresem
def pole_lorentza(A, gamma):
    return np.pi * A * gamma
#czesc termiczna (funkcja)
def model_termiczny(Tinv, I0, A, Ea):

    kB=8.617e-5  # eV/K
    T=1.0/Tinv # Tinv  - odwrotnosc temperatury
    return I0/(1+A*np.exp(-Ea/(kB*T)))

with open("xc.txt", "r") as fx:
    x_data = np.array([float(line.strip()) for line in fx])

pliki = [
    "t1.txt",   # 7 K
    "t2.txt",   # 20 K
    "t3.txt",   # 40 K
    "t4.txt",   # 60 K
    "t5.txt",   # 80 K
    "t6.txt",   # 100 K
    "t7.txt"    # 120 K
]

temperatury = np.array([7, 20, 40, 60, 80, 100, 120])
pole_ix1 = []
pole_ix2 = []
blad_ix1 = []
blad_ix2 = []

for plik in pliki:
    print("Analiza:", plik)
    with open(plik, "r") as fy: #fy - os y
        y_data = np.array([float(line.strip()) for line in fy])
    y_smooth = savgol_filter(y_data, 31, 3)

#znow szukanie piczkow
    peaks, properties = find_peaks(
        y_smooth,
        prominence=np.max(y_smooth)*0.08, # to bylo wczesniej, że jak wystaje wzgledem powierzchni
        distance=120,
        width=15
    )
#ograniczenie, bo wiemy, że sa 2
    # jeśli więcej niż 2 piki -> bierzemy 2 największe
    if len(peaks)>2:
        peak_heights = y_smooth[peaks]
        idx_sorted = np.argsort(peak_heights)[-2:]
        peaks = peaks[idx_sorted]
    peaks = peaks[np.argsort(x_data[peaks])]
    print("Liczba pików:", len(peaks))

#klasycznie_wartosci_poczatkowe
    initial_guess = []
    for peak in peaks:
        x0 = x_data[peak]
        A = y_data[peak]
        gamma = 0.02
        initial_guess.extend([x0, gamma, A])
# tło
    initial_guess.extend([0, 0])
    xmin = np.min(x_data)
    xmax = np.max(x_data)  #zakresy
    lower_bounds = []
    upper_bounds = []

    for peak in peaks:
        lower_bounds.extend([xmin, 0.001, 0])
        upper_bounds.extend([xmax, 0.1, np.max(y_data)*10]) #ograniczniki
    lower_bounds.extend([-np.inf, -np.inf])
    upper_bounds.extend([ np.inf,  np.inf])
#nie dawał ujemnych amplitud
#nie robił gigantycznych gamma
#nie przesuwał pików poza widmo

    popt, pcov = curve_fit(
        wiele_lorentzow, x_data, y_data, p0=initial_guess, bounds=(lower_bounds, upper_bounds), maxfev=50000
    )
#z scipy szuka takich parametrów funkcji, żeby jak najlepiej pasowała do danych
#liczymy pola
    n_peaks=(len(popt)-2)//3
    pola=[]
    bledy=[]

    for i in range(n_peaks):
        x0 = popt[3*i]
        gamma = popt[3*i+1]
        A = popt[3*i+2]
        pole=pole_lorentza(A,gamma)

#i też bledy
        sigma_gamma=np.sqrt(pcov[3*i+1,3*i+1]) #blad gamma
        sigma_A=np.sqrt(pcov[3*i+2,3*i+2]) #blad A
        blad_stat=pole*np.sqrt((sigma_A/A)**2 +(sigma_gamma/gamma)**2)
        blad_systematyczny=0.08*pole
        blad_pola=np.sqrt(
            blad_stat**2 +
            blad_systematyczny**2
        )
        pola.append(pole)
        bledy.append(blad_pola)
        print(f"\nPIK {i+1}")
        print(f"x0 = {x0:.5f}")
        print(f"gamma = {gamma:.5f}")
        print(f"A = {A:.2f}")
        print(f"Pole = {pole:.2f} ± {blad_pola:.2f}")
#To zabezpieczenie na sytuację, gdy w widmie znajdzie się tylko 1 pik
# IX1
    if len(pola) >= 1:
        pole_ix1.append(pola[0])
        blad_ix1.append(bledy[0]) #dodaje niepewnosci
    else:
        pole_ix1.append(np.nan) #nie ma piczku to nan
        blad_ix1.append(np.nan)
# IX2
    if len(pola) >= 2:
        pole_ix2.append(pola[1])
        blad_ix2.append(bledy[1])
    else:
        pole_ix2.append(np.nan)
        blad_ix2.append(np.nan)

pole_ix1=np.array(pole_ix1)
pole_ix2=np.array(pole_ix2)
blad_ix1=np.array(blad_ix1)
blad_ix2=np.array(blad_ix2)

Tinv=1/temperatury

mask1 = ~np.isnan(pole_ix1) #ciagle sprawdzamy czy nan
mask2 = ~np.isnan(pole_ix2)
Tinv1 = Tinv[mask1]
Tinv2 = Tinv[mask2]

pole_ix1_fit=pole_ix1[mask1]
pole_ix2_fit=pole_ix2[mask2]
blad_ix1_fit=blad_ix1[mask1]
blad_ix2_fit=blad_ix2[mask2]

#fit temperatura
p0_ix1 = [
np.max(pole_ix1_fit),1,0.04 #jakies startowe wartosci
]

p0_ix2 = [np.max(pole_ix2_fit),1,0.03 #Io, A, Ea
]

# IX1
popt1, pcov1 = curve_fit(
model_termiczny,Tinv1,pole_ix1_fit,p0=p0_ix1,sigma=blad_ix1_fit,absolute_sigma=True,

    bounds=(
        [0, 0, 0],
        [np.inf, np.inf, 0.2]
    ),
    maxfev=50000
)

# IX2
popt2, pcov2 = curve_fit(model_termiczny,Tinv2,pole_ix2_fit,p0=p0_ix2,sigma=blad_ix2_fit,absolute_sigma=True,
bounds=(
        [0, 0, 0],
        [np.inf, np.inf, 0.2]
    ),
maxfev=50000
)

I01,A1,Ea1 = popt1
I02,A2,Ea2 = popt2
dEa1 = np.sqrt(pcov1[2,2])
dEa2 = np.sqrt(pcov2[2,2])

print("\n===================================")
print("ENERGIE AKTYWACJI")
print("===================================")

print(f"IX1: Ea = {Ea1*1000:.2f} ± {dEa1*1000:.2f} meV")
print(f"IX2: Ea = {Ea2*1000:.2f} ± {dEa2*1000:.2f} meV")

#wykresiki
xfit1 = np.linspace(
np.min(Tinv1),np.max(Tinv1),500
)
xfit2 = np.linspace(
np.min(Tinv2),np.max(Tinv2),500 #punkty, aby bylo gladko
)

yfit1 = model_termiczny(xfit1, *popt1)
yfit2 = model_termiczny(xfit2, *popt2)

plt.figure(figsize=(6,7))
plt.errorbar(
Tinv1,pole_ix1_fit,yerr=blad_ix1_fit,fmt='o',color='blue',capsize=6,label='Ekscyton IX1'
)

plt.errorbar(
    Tinv2,pole_ix2_fit,yerr=blad_ix2_fit,fmt='o',color='red',capsize=6,label='Ekscyton IX2'
)

#fity na wykresie
plt.plot(
    xfit1,yfit1,color='blue'
)
plt.plot(
    xfit2,yfit2,color='red'
)

plt.xlabel(r"1/T (1/K)", fontsize=16)
plt.ylabel("Intensywność PL (a.u.)", fontsize=16)
plt.tick_params(axis='both', labelsize=14)
#tekscik na ekranie
if np.isinf(dEa1):
    tekst1=rf"$E_{{a1}} = {Ea1*1000:.1f}$ meV"
else:
    tekst1=rf"$E_{{a1}} = ({Ea1*1000:.1f}\pm{dEa1*1000:.1f})$ meV"
if np.isinf(dEa2):
    tekst2=rf"$E_{{a2}} = {Ea2*1000:.1f}$ meV"
else:
    tekst2=rf"$E_{{a2}} = ({Ea2*1000:.1f}\pm{dEa2*1000:.1f})$ meV"  #wraz z zabezpieczeniem przed bledem
plt.text(
    0.04,np.max(pole_ix1_fit)*0.80,tekst1,fontsize=16,fontname="Times New Roman" #tekst na wykresie o energii aktywacji_lokalizacja
)
plt.text(
    0.03,np.max(pole_ix2_fit)*0.55,tekst2,fontsize=16,fontname="Times New Roman" #tekst na wykresie o energii aktywacji_lokalizacja

)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth=0.5, color='black')
plt.grid(which='minor', linestyle='--', linewidth=0.5, color='grey')
plt.legend(fontsize=13)
plt.tight_layout()
plt.show()

