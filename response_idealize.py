#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
import plotly.graph_objects as go
import plotly.io as pio
import argparse
import os
import sys

# === CONFIG ===
default_input_file = "Derived_Target.csv"
default_output_file = "Idealized_Target.csv"

smoothing_factor = 0.007   # spline smoothness (0.05–0.2 = close fit, higher = smoother)
reference_hz = 1000        # normalization reference
anchor_freqs = []          # anchor points in Hz


# === REW-Style Variable Smoothing ===
def rew_variable_smoothing(f, mag_db):
    """
    Approximates REW's 'Variable' smoothing:
    1/48 octave below 100 Hz,
    1/3 octave above 10 kHz,
    transitions between those limits with 1/6 octave around 1 kHz.
    """
    f = np.asarray(f)
    mag = np.asarray(mag_db)
    logf = np.log10(f)

    # determine octave fraction at each frequency
    N = np.zeros_like(f)
    for i, freq in enumerate(f):
        if freq <= 100:
            N[i] = 48
        elif freq < 1000:
            N[i] = 48 + (6 - 48) * np.log10(freq / 100) / np.log10(1000 / 100)
        elif freq < 10000:
            N[i] = 6 + (3 - 6) * np.log10(freq / 1000) / np.log10(10000 / 1000)
        else:
            N[i] = 3

    # convert octave fraction -> Gaussian sigma width (in log frequency space)
    bandwidths = f * (2 ** (1 / (2 * N)) - 2 ** (-1 / (2 * N)))
    sigma = np.log10(1 + bandwidths / f)
    sigma_samples = sigma / np.mean(np.diff(logf))

    # apply smoothing (Gaussian with varying sigma)
    smoothed = np.zeros_like(mag)
    for i, s in enumerate(sigma_samples):
        smoothed[i] = gaussian_filter1d(mag, s)[i]

    return smoothed


# === 0. Parse command-line arguments ===
parser = argparse.ArgumentParser(description="Idealize a derived target curve with anchor points and REW-style smoothing.")
parser.add_argument("-i", "--input", type=str, default=default_input_file,
                    help=f"Input CSV file (default: {default_input_file})")
parser.add_argument("-o", "--output", type=str, default=default_output_file,
                    help=f"Output CSV file (default: {default_output_file})")
args = parser.parse_args()

input_file = args.input
output_file = args.output

# === 0b. Graceful file checks ===
if not os.path.isfile(input_file):
    sys.exit(f"Error: Input file '{input_file}' not found. Please check the path and try again.")

output_dir = os.path.dirname(output_file) or "."
if not os.access(output_dir, os.W_OK):
    sys.exit(f"Error: Cannot write to output directory '{output_dir}'. Check permissions.")

# open plot in full browser window
pio.renderers.default = "browser"

# === 1. Load ===
df = pd.read_csv(input_file)
f = df["frequency"].to_numpy()
y_raw = df["raw"].to_numpy()

# === 2. Apply REW-style variable smoothing ===
y_smooth = rew_variable_smoothing(f, y_raw)

# === 3. Prepare data for spline ===
logf = np.log10(f)
anchor_logf = np.log10(anchor_freqs) if anchor_freqs else []
anchor_y = np.interp(anchor_freqs, f, y_smooth) if anchor_freqs else []

# === 4. Smoothing spline over smoothed curve ===
weights = np.ones_like(y_smooth)
if anchor_freqs:
    anchor_indices = [np.argmin(np.abs(f - af)) for af in anchor_freqs]
    weights[anchor_indices] = 5  # higher weight for anchor points

spline = UnivariateSpline(logf, y_smooth, w=weights, s=len(f) * smoothing_factor)
ideal = spline(logf)

# === 5. Normalize to reference ===
ref = np.interp(reference_hz, f, ideal)
ideal -= ref
anchor_y_normalized = anchor_y - ref if anchor_freqs else []

# === 6. Save ===
try:
    pd.DataFrame({"frequency": f, "raw": ideal}).to_csv(output_file, index=False)
except Exception as e:
    sys.exit(f"Error saving output file '{output_file}': {e}")

# === 7. Interactive Plot (Plotly) ===
fig = go.Figure()

# raw
fig.add_trace(go.Scatter(
    x=f, y=y_raw, mode="lines", name="Raw derived",
    line=dict(color="Azure", width=1), opacity=0.4,
    hovertemplate="%{y:.2f} dB"
))

# idealized
fig.add_trace(go.Scatter(
    x=f, y=ideal, mode="lines", name="Idealized (spline of variable-smooth)",
    line=dict(color="red", width=2),
    hovertemplate="%{y:.2f} dB"
))

# anchor points
if anchor_freqs:
    fig.add_trace(go.Scatter(
        x=anchor_freqs, y=anchor_y_normalized, mode="markers", name="Anchor points",
        marker=dict(color="yellow", size=8, symbol="circle"),
        hovertemplate="%{x:.0f} Hz, %{y:.2f} dB"
    ))

fig.update_layout(
    title="Idealized Target Curve (Variable Smoothing)",
    xaxis=dict(title="Frequency (Hz)", type="log"),
    yaxis_title="Level (dB, normalized)",
    hovermode="x unified",
    template="plotly_dark",
    autosize=True,
    height=None,
)

fig.show()
