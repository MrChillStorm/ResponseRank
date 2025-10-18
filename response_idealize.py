#!/usr/bin/env python3

"""
response_idealize.py

Performs REW-style variable smoothing in log-frequency space using per-point
normalized Gaussian kernels, with gap handling to prevent bridging artifacts.
Reads input CSV with provided headers or as headerless if none exist. Outputs
CSV with the input headers or 'frequency,raw' for headerless files.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d
import plotly.graph_objects as go
import plotly.io as pio

# === CONFIG ===
DEFAULT_INPUT = "Derived_Target.csv"
DEFAULT_OUTPUT = "Idealized_Target.csv"
REFERENCE_HZ = 1000
ANCHOR_FREQS = []  # optional anchor points
DEFAULT_POINTS_PER_OCTAVE = 48
MAX_WINDOW_SAMPLES = 50  # Cap on kernel window size to prevent over-smoothing
GAP_THRESHOLD_FACTOR = 5.0  # Factor for detecting large gaps

def error_exit(msg):
    """Print error message and exit cleanly without traceback."""
    print(f"Error: {msg}")
    sys.exit(1)

# === Octave fraction N per frequency ===
def octave_fraction_N(freq):
    N = np.empty_like(freq, dtype=float)
    for i, f in enumerate(freq):
        if f <= 100:
            N[i] = 48.0
        elif f < 1000:
            N[i] = 48.0 + (6.0 - 48.0) * np.log10(f / 100.0) / np.log10(1000.0 / 100.0)
        elif f < 10000:
            N[i] = 6.0 + (3.0 - 6.0) * np.log10(f / 1000.0) / np.log10(10000.0 / 1000.0)
        else:
            N[i] = 3.0
    return N

# === Resample to uniform log-frequency grid ===
def resample_log_frequency(f, y, points_per_octave=DEFAULT_POINTS_PER_OCTAVE):
    logf_min, logf_max = np.log10(np.min(f)), np.log10(np.max(f))
    if logf_min == logf_max:
        return f, y  # No resampling needed if single point or same freq
    num_points = max(int((logf_max - logf_min) / np.log10(2) * points_per_octave), 2)  # At least 2 points
    logf_new = np.linspace(logf_min, logf_max, num_points)
    f_new = 10 ** logf_new
    interp_func = interp1d(np.log10(f), y, kind="linear", fill_value="extrapolate")
    y_new = interp_func(logf_new)
    return f_new, y_new

# === Detect frequency gaps ===
def detect_gaps(f, threshold_factor=GAP_THRESHOLD_FACTOR):
    logf = np.log10(f)
    log_diff = np.diff(logf)
    mean_diff = np.mean(log_diff)
    gaps = log_diff > threshold_factor * mean_diff
    gap_indices = np.where(gaps)[0]
    return gap_indices

# === Core: variable log-space smoothing with gap handling ===
def rew_variable_smoothing_logspace_preserve_energy(f, mag_db, strength=1.0, half_width_multiplier=4.0):
    f = np.asarray(f)
    mag = np.asarray(mag_db)

    # Convert to log-frequency
    logf = np.log10(f)
    log_step = np.mean(np.diff(logf))

    # Detect gaps
    gap_indices = detect_gaps(f)
    weights = np.ones_like(mag)
    for idx in gap_indices:
        weights[max(0, idx-1):min(len(mag), idx+2)] = 0.1  # Down-weight near gaps

    # Compute octave fraction N and sigma per point
    N = octave_fraction_N(f)
    bandwidths = f * (2 ** (1.0 / (2.0 * N)) - 2 ** (-1.0 / (2.0 * N)))
    sigma_log = np.log10(1.0 + bandwidths / f) * strength
    sigma_samples = np.maximum(sigma_log / log_step, 0.01)

    # Smooth with endpoint truncation and gap handling
    y_smooth = np.empty_like(mag)
    n = len(f)
    idx = np.arange(n)
    for i in range(n):
        s = sigma_samples[i]
        hw = min(int(np.ceil(half_width_multiplier * s)), MAX_WINDOW_SAMPLES)
        left = max(0, i - hw)
        right = min(n - 1, i + hw)
        window_idx = idx[left:right + 1]
        distances = window_idx - i
        kernel = np.exp(-0.5 * (distances / s) ** 2) * weights[window_idx]
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel /= kernel_sum
            y_smooth[i] = np.dot(kernel, mag[window_idx])
        else:
            y_smooth[i] = mag[i]  # Fallback to raw value if kernel is invalid

    return y_smooth

# === Main script & CLI ===
def main():
    parser = argparse.ArgumentParser(description="Idealize target curve with REW-style variable smoothing (log-space).")
    parser.add_argument("-i", "--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoothing-strength", type=float, default=0.5)
    parser.add_argument("--smoothing-factor", type=float, default=0.001)
    parser.add_argument("--half-width-mult", type=float, default=4.0)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        error_exit(f"Input file '{args.input}' not found.")
    outdir = os.path.dirname(args.output) or "."
    if not os.access(outdir, os.W_OK):
        error_exit(f"Cannot write to output directory '{outdir}'.")

    pio.renderers.default = "browser"

    try:
        # Read CSV with explicit comma delimiter and UTF-8 encoding
        df = pd.read_csv(args.input, header="infer", sep=',', encoding='utf-8')
        # Check if the first row (headers) is non-numeric
        first_row = df.columns[:2].to_list()
        is_numeric = not np.isnan(pd.to_numeric(first_row, errors='coerce')).all()
        if is_numeric:
            # No headers detected, re-read as headerless with default names
            print("Warning: No headers detected in input CSV (first row is numeric). Using default headers 'frequency,raw'.")
            df = pd.read_csv(args.input, header=None, names=["frequency", "raw"], sep=',', encoding='utf-8')
            headers = ["frequency", "raw"]
        else:
            # Headers present, use them
            headers = first_row
            print(f"Using input CSV headers: {headers}")
    except pd.errors.EmptyDataError:
        error_exit("Input CSV file is empty or contains no valid data.")
    except pd.errors.ParserError:
        error_exit("Failed to parse CSV file. Check delimiter (expected comma) and file format.")
    except Exception as e:
        error_exit(f"Failed to read input CSV file: {str(e)}")

    if df.shape[1] < 2:
        error_exit("Input CSV must have at least two columns.")

    try:
        # Debug non-numeric values
        col1 = df.iloc[:, 0].astype(str)
        col2 = df.iloc[:, 1].astype(str)
        non_numeric_col1 = col1[~pd.to_numeric(col1, errors='coerce').notna()]
        non_numeric_col2 = col2[~pd.to_numeric(col2, errors='coerce').notna()]
        if not non_numeric_col1.empty or not non_numeric_col2.empty:
            error_exit(f"Non-numeric values found in columns: {non_numeric_col1.tolist()[:5]}, {non_numeric_col2.tolist()[:5]}")
        f = pd.to_numeric(df.iloc[:, 0], errors='raise').to_numpy()
        y_raw = pd.to_numeric(df.iloc[:, 1], errors='raise').to_numpy()
    except ValueError as e:
        error_exit(f"First two columns of input CSV must contain numeric values. Error: {str(e)}")

    if len(f) < 2:
        error_exit("Input CSV must have at least 2 data points.")

    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(y_raw)):
        error_exit("Frequency and raw values must be finite numbers.")

    if np.any(f <= 0):
        error_exit("Frequency values must be positive.")

    if np.any(np.diff(f) <= 0):
        error_exit("Frequency values must be strictly increasing.")

    logf = np.log10(f)
    log_diff = np.diff(logf)
    mean_diff = np.mean(log_diff)

    has_large_gaps = np.any(log_diff > GAP_THRESHOLD_FACTOR * mean_diff)
    if has_large_gaps:
        print("Warning: Large frequency gaps detected, resampling to uniform grid.")
        f, y_raw = resample_log_frequency(f, y_raw)

    # Apply variable log-space smoothing
    y_smooth = rew_variable_smoothing_logspace_preserve_energy(
        f, y_raw,
        strength=args.smoothing_strength,
        half_width_multiplier=args.half_width_mult
    )

    # Octave-energy preservation and 1kHz anchor
    power = 10 ** (y_smooth / 10.0)
    octave_widths = np.gradient(np.log2(f))
    if np.any(octave_widths <= 0):
        error_exit("Invalid octave widths detected after resampling.")

    power *= octave_widths / np.mean(octave_widths)
    y_smooth = 10 * np.log10(power)

    ref_idx = np.argmin(np.abs(f - REFERENCE_HZ))
    y_smooth -= y_smooth[ref_idx] - y_raw[ref_idx]

    # Spline fit
    logf = np.log10(f)
    weights = np.ones_like(y_smooth)
    if len(ANCHOR_FREQS) > 0:
        anchor_indices = [np.argmin(np.abs(f - af)) for af in ANCHOR_FREQS if min(f) <= af <= max(f)]
        if anchor_indices:
            weights[anchor_indices] = 5.0

    if len(f) < 4:
        print("Warning: Too few points for spline fit, using linear interpolation.")
        interp_func = interp1d(logf, y_smooth, kind="linear", fill_value="extrapolate")
        ideal = interp_func(logf)
    else:
        spline = UnivariateSpline(logf, y_smooth, w=weights, s=len(f) * args.smoothing_factor)
        ideal = spline(logf)

    # Normalize to reference
    ref = np.interp(REFERENCE_HZ, f, ideal)
    ideal -= ref

    # Save with input headers (or defaults), explicitly ensuring headers are written
    print(f"Writing output CSV with headers: {headers}")
    pd.DataFrame({headers[0]: f, headers[1]: ideal}).to_csv(args.output, index=False, header=True)

    # For plot, normalize all curves to 0 at REFERENCE_HZ
    raw_ref = np.interp(REFERENCE_HZ, f, y_raw)
    smooth_ref = np.interp(REFERENCE_HZ, f, y_smooth)
    y_raw_plot = y_raw - raw_ref
    y_smooth_plot = y_smooth - smooth_ref
    y_ideal_plot = ideal  # Already normalized

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=y_raw_plot, mode="lines", name="Raw", line=dict(color="deepskyblue", width=2), opacity=0.4))
    fig.add_trace(go.Scatter(x=f, y=y_smooth_plot, mode="lines", name="Variable-smoothed", line=dict(color="lime", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=f, y=y_ideal_plot, mode="lines", name="Idealized (spline)", line=dict(color="dodgerblue", width=2)))

    if len(ANCHOR_FREQS) > 0:
        anchor_y = np.interp(ANCHOR_FREQS, f, y_smooth) - smooth_ref
        fig.add_trace(go.Scatter(x=ANCHOR_FREQS, y=anchor_y, mode="markers", name="Anchors", marker=dict(color="yellow", size=8)))

    fig.update_layout(title="Idealized Target Curve",
                      xaxis=dict(title="Frequency (Hz)", type="log"),
                      yaxis_title="Level (dB, normalized)",
                      hovermode="x unified",
                      template="plotly_dark")
    fig.show()

if __name__ == "__main__":
    main()
