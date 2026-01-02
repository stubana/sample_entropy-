# sample_entropy-
#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
import datetime

# ============================================================
M = 2                  # embedding dimension
TAU = 1                # time delay
R_FACTOR = 0.2         # tolerance scaling
MIN_LENGTH = 100       # minimum acceptable time series length
VAR_EPS = 1e-8         # near-zero variance threshold

# ============================================================
# PATHS
# ============================================================
voxelwise_dir = Path('/home/stubanadean/voxelwise_timeseries_nilearn')
output_dir = Path('/home/stubanadean/sample_entropy_results')
output_dir.mkdir(exist_ok=True)

layers = ['Interoception', 'Exteroception', 'Cognition']

# ============================================================
def sampen_matlab_equivalent(x, m=2, r=0.2, tau=1):
    x = np.asarray(x, dtype=float)
    N = len(x)

    if N <= (m + 1) * tau:
        return np.nan, 0, 0

    def embed(seq, order):
        return np.array([
            seq[i : i + order * tau : tau]
            for i in range(N - order * tau + 1)
        ])

    Xm = embed(x, m)
    Xm1 = embed(x, m + 1)

    def count_matches(X):
        count = 0
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                if np.max(np.abs(X[i] - X[j])) < r:
                    count += 1
        return count

    B = count_matches(Xm)
    A = count_matches(Xm1)

    if A == 0 or B == 0:
        return np.nan, A, B

    se = -np.log2(A / B)
    return se, A, B

# ============================================================
# SANITY CHECK 
# ============================================================
def sanity_check():
    N = 1000
    noise = np.random.randn(N)
    sine = np.sin(np.linspace(0, 20*np.pi, N))

    r_noise = R_FACTOR * np.std(noise)
    r_sine = R_FACTOR * np.std(sine)

    se_noise, _, _ = sampen_matlab_equivalent(noise, M, r_noise, TAU)
    se_sine, _, _ = sampen_matlab_equivalent(sine, M, r_sine, TAU)

    print("Sanity check:")
    print("  White noise SampEn:", se_noise)
    print("  Sine wave SampEn :", se_sine)
    print("  PASS =", se_noise > se_sine)
    print("-" * 50)

sanity_check()

# ============================================================
# MAIN ANALYSIS
# ============================================================
results = []

for layer in layers:
    layer_dir = voxelwise_dir / layer
    roi_files = list(layer_dir.glob("*_voxelwise_timeseries.npy"))

    for ts_file in roi_files:
        roi_name = ts_file.name.replace("_voxelwise_timeseries.npy", "")
        ts_data = np.load(ts_file)      # shape: time x voxels

        T, V = ts_data.shape

        # ---------- BASIC QC ----------
        if T < MIN_LENGTH:
            print(f"Skipping {roi_name}: too short (T={T})")
            continue

        voxel_std = np.std(ts_data, axis=0)
        valid_voxels = voxel_std > VAR_EPS

        ts_data = ts_data[:, valid_voxels]
        voxel_std = voxel_std[valid_voxels]

        if ts_data.shape[1] == 0:
            print(f"Skipping {roi_name}: no valid voxels")
            continue

        sampen_vals = []
        A_vals = []
        B_vals = []

        # ---------- PER-VOXEL SampEn ----------
        for v in range(ts_data.shape[1]):
            x = ts_data[:, v]
            r = R_FACTOR * voxel_std[v]

            se, A, B = sampen_matlab_equivalent(x, M, r, TAU)

            sampen_vals.append(se)
            A_vals.append(A)
            B_vals.append(B)

        sampen_vals = np.array(sampen_vals)
        A_vals = np.array(A_vals)
        B_vals = np.array(B_vals)

        # ---------- QC METRICS ----------
        nan_fraction = np.mean(np.isnan(sampen_vals))
        zero_A_fraction = np.mean(A_vals == 0)
        zero_B_fraction = np.mean(B_vals == 0)

        results.append({
            "Layer": layer,
            "ROI": roi_name,
            "Timepoints": T,
            "Num_voxels_used": ts_data.shape[1],
            "Mean_SampEn": np.nanmean(sampen_vals),
            "Median_SampEn": np.nanmedian(sampen_vals),
            "Std_SampEn": np.nanstd(sampen_vals),
            "Min_SampEn": np.nanmin(sampen_vals),
            "Max_SampEn": np.nanmax(sampen_vals),
            "Frac_NaN_SampEn": nan_fraction,
            "Frac_A_zero": zero_A_fraction,
            "Frac_B_zero": zero_B_fraction
        })

        print(f"{layer} | {roi_name}")
        print(f"  Voxels used        : {ts_data.shape[1]}")
        print(f"  NaN SampEn frac    : {nan_fraction:.3f}")
        print(f"  A=0 frac           : {zero_A_fraction:.3f}")
        print(f"  B=0 frac           : {zero_B_fraction:.3f}")
        print("-" * 50)

# ============================================================
# SAVE RESULTS
# ============================================================
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
outpath = output_dir / f"sample_entropy_summary_{timestamp}.csv"
pd.DataFrame(results).to_csv(outpath, index=False)

print(f"\nSaved results to: {outpath}")
