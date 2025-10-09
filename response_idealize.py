#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go
import plotly.io as pio
import argparse
import os
import sys

# === CONFIG ===
default_input_file = "Derived_Target.csv"
default_output_file = "Idealized_Target.csv"

sigma_octaves = 0.12       # gentle smoothing
hump_search = (1500, 8000) # where to look for hump
reference_hz = 500         # normalization reference

# === 0. Parse command-line arguments ===
parser = argparse.ArgumentParser(description="Idealize a derived target curve.")
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
y = df["raw"].to_numpy()

# === 2. Smooth ===
logf = np.log10(f)
sigma_points = sigma_octaves / np.mean(np.diff(logf))
y_smooth = gaussian_filter1d(y, sigma_points)

# === 3. Detect hump automatically ===
search_mask = (f >= hump_search[0]) & (f <= hump_search[1])
search_f = f[search_mask]
search_y = y_smooth[search_mask]

peaks, _ = find_peaks(search_y, distance=40)
if len(peaks) == 0:
    raise RuntimeError("No hump found in midrange region.")
peak_idx = peaks[np.argmax(search_y[peaks])]
peak_freq = search_f[peak_idx]
peak_amp = search_y[peak_idx]

# find -3 dB boundaries
half_height = peak_amp - 3
left = search_f[np.where(search_y[:peak_idx] <= half_height)[0][-1]] if np.any(search_y[:peak_idx] <= half_height) else hump_search[0]
right = search_f[peak_idx + np.where(search_y[peak_idx:] <= half_height)[0][0]] if np.any(search_y[peak_idx:] <= half_height) else hump_search[1]

# === 4. Fit cubic spline through region + boundary anchors ===
region_mask = (f >= left) & (f <= right)
region_f = f[region_mask]
region_y = y_smooth[region_mask]

anchor_left_idx = np.where(f < left)[0][-1] if np.any(f < left) else 0
anchor_right_idx = np.where(f > right)[0][0] if np.any(f > right) else len(f) - 1

fit_f = np.concatenate(([f[anchor_left_idx]], region_f, [f[anchor_right_idx]]))
fit_y = np.concatenate(([y_smooth[anchor_left_idx]], region_y, [y_smooth[anchor_right_idx]]))

cs = CubicSpline(np.log10(fit_f), fit_y, bc_type='natural')
ideal_region = cs(np.log10(region_f))

# Normalize to preserve peak height
ideal_region += peak_amp - np.max(ideal_region)

# === 5. Merge ===
ideal = y_smooth.copy()
ideal[region_mask] = ideal_region

# === 6. Keep bass as original below hump start ===
bass_mask = f < left
ideal[bass_mask] = y_smooth[bass_mask]

# === 7. Normalize to reference ===
ref = np.interp(reference_hz, f, ideal)
ideal -= ref

# === 8. Save ===
try:
    pd.DataFrame({"frequency": f, "raw": ideal}).to_csv(output_file, index=False)
except Exception as e:
    sys.exit(f"Error saving output file '{output_file}': {e}")

# === 9. Interactive Plot (Plotly, auto-sized) ===
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=f, y=y, mode="lines", name="Raw derived",
    line=dict(color="Azure", width=1), opacity=0.4,
    hovertemplate="%{y:.2f} dB"
))
fig.add_trace(go.Scatter(
    x=f, y=y_smooth, mode="lines", name="Smoothed",
    line=dict(color="#ff7f0e", width=1.5), opacity=0.6,
    hovertemplate="%{y:.2f} dB"
))
baseline = y_smooth  # the smoothed curve we want as reference
fig.add_trace(go.Scatter(
    x=f, 
    y=ideal,  # visually lifted
    customdata=np.round(ideal - np.interp(reference_hz, f, ideal) + np.interp(reference_hz, f, baseline), 2),
    hovertemplate="%{customdata:.2f} dB",
    name=f"Idealized (peak {peak_freq:.0f} Hz)",
    line=dict(color='red', width=2)
))
fig.add_vline(x=left, line_width=1, line_dash="dash", line_color="black")
fig.add_vline(x=right, line_width=1, line_dash="dash", line_color="black")

fig.update_layout(
    title="Idealized Target Curve",
    xaxis=dict(title="Frequency (Hz)", type="log"),
    yaxis_title="Level (dB, normalized)",
    hovermode="x unified",
    template="plotly_dark",
    autosize=True,
    height=None,
)

fig.show()
