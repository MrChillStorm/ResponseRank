#!/usr/bin/env python3
"""
response_idealize_rew_variable.py

Performs REW-style variable smoothing in log-frequency space with per-point
normalized Gaussian kernels to preserve local energy and produce endpoint bends.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import plotly.graph_objects as go
import plotly.io as pio

# === CONFIG ===
DEFAULT_INPUT = "Derived_Target.csv"
DEFAULT_OUTPUT = "Idealized_Target.csv"
REFERENCE_HZ = 1000
ANCHOR_FREQS = []  # optional anchor points
DEFAULT_POINTS_PER_OCTAVE = 48

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

# === Core: variable log-space smoothing with per-point sigma and endpoint truncation ===
def rew_variable_smoothing_logspace_preserve_energy(f, mag_db, strength=1.0, half_width_multiplier=4.0):
    f = np.asarray(f)
    mag = np.asarray(mag_db)

    # convert to log-frequency
    logf = np.log10(f)
    log_step = np.mean(np.diff(logf))

    # compute octave fraction N and sigma per point
    N = octave_fraction_N(f)
    bandwidths = f * (2 ** (1.0 / (2.0 * N)) - 2 ** (-1.0 / (2.0 * N)))
    sigma_log = np.log10(1.0 + bandwidths / f) * strength
    sigma_samples = sigma_log / log_step
    sigma_samples = np.maximum(sigma_samples, 0.01)

    # smooth with endpoint truncation
    y_smooth = np.empty_like(mag)
    n = len(f)
    idx = np.arange(n)
    for i in range(n):
        s = sigma_samples[i]
        hw = int(np.ceil(half_width_multiplier * s))
        left = max(0, i - hw)
        right = min(n - 1, i + hw)
        window_idx = idx[left:right + 1]
        distances = window_idx - i
        kernel = np.exp(-0.5 * (distances / s) ** 2)
        kernel /= kernel.sum()
        y_smooth[i] = np.dot(kernel, mag[window_idx])

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
        sys.exit(f"Error: Input file '{args.input}' not found.")
    outdir = os.path.dirname(args.output) or "."
    if not os.access(outdir, os.W_OK):
        sys.exit(f"Error: Cannot write to output directory '{outdir}'.")

    pio.renderers.default = "browser"

    df = pd.read_csv(args.input)
    if "frequency" not in df.columns or "raw" not in df.columns:
        sys.exit("Error: input CSV must contain 'frequency' and 'raw' columns.")
    f = df["frequency"].to_numpy()
    y_raw = df["raw"].to_numpy()

    # apply variable log-space smoothing
    y_smooth = rew_variable_smoothing_logspace_preserve_energy(
        f, y_raw,
        strength=args.smoothing_strength,
        half_width_multiplier=args.half_width_mult
    )

    # --- octave-energy preservation and 1kHz anchor ---
    power = 10 ** (y_smooth / 10.0)
    octave_widths = np.gradient(np.log2(f))
    power *= octave_widths / np.mean(octave_widths)
    y_smooth = 10 * np.log10(power)

    ref_idx = np.argmin(np.abs(f - 1000))
    y_smooth -= y_smooth[ref_idx] - y_raw[ref_idx]

    # spline fit
    logf = np.log10(f)
    weights = np.ones_like(y_smooth)
    if len(ANCHOR_FREQS) > 0:
        anchor_indices = [np.argmin(np.abs(f - af)) for af in ANCHOR_FREQS]
        weights[anchor_indices] = 5.0

    spline = UnivariateSpline(logf, y_smooth, w=weights, s=len(f) * args.smoothing_factor)
    ideal = spline(logf)

    # normalize to reference
    ref = np.interp(REFERENCE_HZ, f, ideal)
    ideal -= ref

    # save
    pd.DataFrame({"frequency": f, "raw": ideal}).to_csv(args.output, index=False)

    # plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=y_raw, mode="lines", name="Raw", line=dict(color="Azure", width=1), opacity=0.4))
    fig.add_trace(go.Scatter(x=f, y=y_smooth, mode="lines", name="Variable-smoothed", line=dict(color="#FFBF00", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=f, y=ideal, mode="lines", name="Idealized (spline)", line=dict(color="red", width=2)))

    if len(ANCHOR_FREQS) > 0:
        anchor_y = np.interp(ANCHOR_FREQS, f, y_smooth) - ref
        fig.add_trace(go.Scatter(x=ANCHOR_FREQS, y=anchor_y, mode="markers", name="Anchors", marker=dict(color="yellow", size=8)))

    fig.update_layout(title="Idealized Target Curve",
                      xaxis=dict(title="Frequency (Hz)", type="log"),
                      yaxis_title="Level (dB, normalized)",
                      hovermode="x unified",
                      template="plotly_dark")
    fig.show()

if __name__ == "__main__":
    main()
