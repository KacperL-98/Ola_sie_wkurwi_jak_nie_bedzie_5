ja vidze BH
cos
print("gunwo")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os
from warnings import catch_warnings, simplefilter

source_folder = r"C:\Users\Bartosz\Downloads\zamienione2-20251218T144717Z-1-001\zamienione2\map12"
save_folder = r"C:\Users\Bartosz\Downloads\zamienione2-20251218T144717Z-1-001\zamienione2\zrobione map12\podejscie0501"
output_summary_path = os.path.join(save_folder, "all_peaks_summary_reff_att5.txt")
x_ranges = [(0, 1.650), (1.640, 1.667), (1.65, 1.679),
            (1.675, 1.71), (1.69, 1.725), (1.725, 1.755), (1.74, 1.79)]
#x_ranges = [(1.7, 1.8)]
os.makedirs(save_folder, exist_ok=True)

def gaussian(x, A, x0, sigma, y0):
    return y0 + A * np.exp(-((x - x0)**2) / (2 * sigma**2))

def fwhm_gaussian(sigma):
    return 2.3548 * sigma



all_results = []

with open(output_summary_path, "w", encoding="utf-8") as out_file:
    out_file.write("File\tRange_ID\tA\tX0\tSigma\tFWHM\tY0\tR_squared\tArea\n")

    for file_idx, filename in enumerate(os.listdir(source_folder)):
        if not (filename.endswith(".txt") or filename.endswith(".csv")):
            continue

        file_path = os.path.join(source_folder, filename)
        try:
            data = pd.read_csv(file_path, sep=r"\s+", header=None)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        x = data.iloc[:, 0].to_numpy(dtype=float)
        y = data.iloc[:, 1].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        # --- Plot every 100th file ---
        do_plot = (file_idx % 100 == 0)
        if do_plot:
            plt.figure(figsize=(8, 5))
            plt.plot(x, y, "k-", lw=1, label="Data")

        for range_id, (x_min, x_max) in enumerate(x_ranges):
            # Select data only within this range
            range_mask = (x >= x_min) & (x <= x_max)
            x_range = x[range_mask]
            y_range = y[range_mask]

            if len(x_range) < 5:
                continue  # not enough points for fit

            # Initial guesses
            A_guess = np.max(y_range) - np.min(y_range)
            x0_guess = x_range[np.argmax(y_range)]
            sigma_guess = (x_max - x_min) / 6
            y0_guess = np.min(y_range)
            p0 = [A_guess, x0_guess, sigma_guess, y0_guess]

            try:
                params, _ = curve_fit(gaussian, x_range, y_range, p0=p0, maxfev=10000)
                A, x0, sigma, y0 = params
                FWHM = fwhm_gaussian(sigma)
                y_fit = gaussian(x_range, *params)
                R2 = 1 - np.sum((y_range - y_fit)**2) / np.sum((y_range - np.mean(y_range))**2)
                area = A * sigma * np.sqrt(2 * np.pi)

                # Save results
                out_file.write(f"{filename}\t{range_id}\t{A:.6f}\t{x0:.6f}\t{sigma:.6f}\t"
                               f"{FWHM:.6f}\t{y0:.6f}\t{R2:.6f}\t{area:.6f}\n")

                all_results.append({
                    "file": filename,
                    "range_id": range_id,
                    "A": A,
                    "x0": x0,
                    "sigma": sigma,
                    "FWHM": FWHM,
                    "y0": y0,
                    "R2": R2,
                    "area": area
                })

                # Plot fit if required
                if do_plot:
                    plt.plot(x_range, y_fit, '-', lw=1.5, label=f"Range {range_id+1}")

            except Exception as e:
                print(f"Fit failed for {filename}, range {x_min}-{x_max}: {e}")
                continue

        if do_plot:
            plt.title(f"Gaussian fits (1 per range) for: {filename}")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend(fontsize=7)
            plt.tight_layout()
            save_fig_path = os.path.join(save_folder, f"{os.path.splitext(filename)[0]}_fit_att1.png")
            plt.savefig(save_fig_path, dpi=300)
            plt.close()
            print(f"Saved plot for {filename}")

        #print(f"Processed: {filename}")

print("All spectra processed successfully.")
"""

all_results = []

with open(output_summary_path, "w", encoding="utf-8") as out_file:
    out_file.write("File\tPeak_ID\tA\tX0\tWidth\tFWHM\tY0\tR_squared\n")

    for file_idx, filename in enumerate(os.listdir(source_folder)):
        if not filename.lower().endswith((".txt", ".csv")):
            continue

        file_path = os.path.join(source_folder, filename)
        try:
            data = pd.read_csv(file_path, sep=r"\s+", header=None)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        # --- Load data ---
        x = data.iloc[:, 0].to_numpy(dtype=float)
        y = data.iloc[:, 1].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        # --- Peak detection ---
        peaks, _ = find_peaks(y, height=np.mean(y) + 0.5 * np.std(y), distance=10)
        if len(peaks) == 0:
            continue

        for peak_id, peak_idx in enumerate(peaks):
            # --- Define local window in x-units ---
            x0_guess = x[peak_idx]
            window_width = 0.015  # eV (or whatever unit your x is)
            mask_window = (x >= x0_guess - window_width) & (x <= x0_guess + window_width)
            x_peak = x[mask_window]
            y_peak = y[mask_window]

            if len(x_peak) < 5:
                continue  # too few points for fit

            # --- Initial guesses ---
            A_guess = y[peak_idx] - np.min(y_peak)
            width_guess = (x_peak[-1] - x_peak[0]) / 10
            y0_guess = np.min(y_peak)
            p0 = [A_guess, x0_guess, width_guess, y0_guess]

            with catch_warnings():
                simplefilter("ignore")
                try:
                    popt, pcov = curve_fit(gaussian, x_peak, y_peak, p0=p0, maxfev=10000)
                    A, x0, sigma, y0 = popt
                    FWHM = fwhm_gaussian(sigma)
                    y_fit = gaussian(x_peak, *popt)
                    R2 = 1 - np.sum((y_peak - y_fit)**2) / np.sum((y_peak - np.mean(y_peak))**2)
                except Exception:
                    continue

            # --- Save results ---
            out_file.write(f"{filename}\t{peak_id}\t{A:.4f}\t{x0:.6f}\t{sigma:.6f}\t"
                           f"{FWHM:.6f}\t{y0:.4f}\t{R2:.4f}\n")

            all_results.append({
                "file": filename,
                "peak_id": peak_id,
                "A": A,
                "x0": x0,
                "sigma": sigma,
                "FWHM": FWHM,
                "y0": y0,
                "R2": R2
            })

        if file_idx % 100 == 0:
            print(f"Processed {file_idx} files...")

print(f" Done! Results saved to: {output_summary_path}")
"""

results_df = pd.DataFrame(all_results)
for i, (x_min, x_max) in enumerate(x_ranges):
    range_df = results_df[(results_df["x0"] >= x_min) & (results_df["x0"] < x_max)]
    range_file = os.path.join(save_folder, f"fit_parameters_range_{x_min:.3f}_{x_max:.3f}.txt")
    range_df.to_csv(range_file, sep="\t", index=False)
    print(f"Saved range file: {range_file}")
