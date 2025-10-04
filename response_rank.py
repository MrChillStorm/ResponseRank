#!/usr/bin/env python3

# ResponseRank
# Rank headphone frequency response measurements against a target curve

import os
import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot top N headphone measurements against a target curve, with optional weighting (interactive)."
    )
    parser.add_argument(
        "measurements_dir",
        help="Directory containing measurement CSV files."
    )
    parser.add_argument(
        "target_path",
        help="Path to the target CSV file (frequency in first column, response in second)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--aweight",
        action="store_true",
        help="Use A-weighting for RMSE calculation."
    )
    group.add_argument(
        "--cweight",
        action="store_true",
        help="Use C-weighting for RMSE calculation."
    )
    parser.add_argument(
        "--top",
        type=int,
        help="Number of top headphones to plot."
    )
    parser.add_argument(
        "--ranking",
        type=str,
        help="Comma-separated list of rank numbers to plot (e.g., 1,3,5)."
    )
    return parser.parse_args()

def a_weight(freq):
    f = np.asarray(freq, dtype=float)
    f2 = f**2
    c = 12194.0
    num = (c**2) * (f2**2)
    den = (f2 + 20.6**2) * (f2 + c**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
    Ra = num / den
    f1k2 = 1000.0**2
    Ra1k = (c**2) * (f1k2**2) / ((f1k2 + 20.6**2) * (f1k2 + c**2) *
                                  np.sqrt((f1k2 + 107.7**2) * (f1k2 + 737.9**2)))
    return Ra / Ra1k

def c_weight(freq):
    f = np.asarray(freq, dtype=float)
    f2 = f**2
    c = 12194.0
    num = (c**2) * f2
    den = (f2 + 20.6**2) * (f2 + c**2)
    Rc = num / den
    f1k2 = 1000.0**2
    Rc1k = (c**2) * f1k2 / ((f1k2 + 20.6**2) * (f1k2 + c**2))
    return Rc / Rc1k

def smooth_response(freq, resp, window=51):
    """Apply 1/3-octave smoothing using Savitzky-Golay filter."""
    if len(resp) < window:
        window = len(resp) // 2 * 2 + 1
    return savgol_filter(resp, window, 3)

def normalize_at_1kHz(freq, resp):
    """Normalize response to 0 dB at 1 kHz by interpolating and subtracting the value at 1 kHz."""
    interp_func = np.interp(1000, freq, resp)
    return resp - interp_func

def compute_rmse(meas_freq, meas_resp, target_freq, target_resp, weights):
    """Compute RMSE on raw, normalized data."""
    interp_resp = np.interp(target_freq, meas_freq, meas_resp)
    interp_resp = normalize_at_1kHz(target_freq, interp_resp)
    target_resp = normalize_at_1kHz(target_freq, target_resp)
    rmse = np.sqrt(np.mean(weights * (interp_resp - target_resp)**2))
    return rmse

def compute_pref_metrics(meas_freq, meas_resp, target_freq, target_resp):
    """Compute preference score on smoothed, normalized data."""
    interp_resp = np.interp(target_freq, meas_freq, meas_resp)
    interp_resp = smooth_response(target_freq, interp_resp)
    interp_resp = normalize_at_1kHz(target_freq, interp_resp)
    target_resp = smooth_response(target_freq, target_resp)
    target_resp = normalize_at_1kHz(target_freq, target_resp)
    error = interp_resp - target_resp

    # Mask for 20 Hz to 10 kHz
    mask_20_10k = (target_freq >= 20) & (target_freq <= 10000)
    error_20_10k = error[mask_20_10k]
    freq_20_10k = target_freq[mask_20_10k]

    # Standard Deviation (SD) over 20 Hz to 10 kHz
    sd = np.std(error_20_10k)

    # Absolute Slope (AS) over 20 Hz to 10 kHz
    x = np.log(freq_20_10k).reshape(-1, 1)
    y = error_20_10k
    model = LinearRegression().fit(x, y)
    as_value = np.abs(model.coef_[0])

    # Preference Score (based on 2018 AES paper)
    pref = 114.49 - (12.62 * sd) - (15.52 * as_value)
    return pref

def main():
    args = parse_args()
    measurements_dir = args.measurements_dir
    target_path = args.target_path
    use_a = args.aweight
    use_c = args.cweight
    top_n = args.top
    ranking = args.ranking

    # Load raw target curve
    target = pd.read_csv(target_path)
    target_freq = target.iloc[:, 0].values
    target_resp = target.iloc[:, 1].values

    # Precompute weights
    if use_a:
        weights = a_weight(target_freq)
        weight_label = "A-weighted"
    elif use_c:
        weights = c_weight(target_freq)
        weight_label = "C-weighted"
    else:
        weights = np.ones_like(target_freq)  # flat weighting
        weight_label = "Flat"

    results = []

    for fname in os.listdir(measurements_dir):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(measurements_dir, fname)
        try:
            meas = pd.read_csv(fpath)
            meas_freq = meas.iloc[:, 0].values
            meas_resp = meas.iloc[:, 1].values
            rmse = compute_rmse(meas_freq, meas_resp, target_freq, target_resp, weights)
            pref = compute_pref_metrics(meas_freq, meas_resp, target_freq, target_resp)
            results.append((fname, rmse, pref))
        except Exception as e:
            print(f"Skipping {fname}, error: {e}")

    # Sort by RMSE
    results.sort(key=lambda x: x[1])

    print("Ranked headphones (closest first):")
    for i, (fname, rmse, pref) in enumerate(results, start=1):
        print(f"{i:3d}. {fname:<50} RMSE={rmse:.4f}  Pref≈{pref:.2f}")

    # Determine which items to plot
    to_plot = []
    title_parts = []

    if top_n:
        to_plot.extend(results[:top_n])
        title_parts.append(f"Top {top_n}")

    if ranking:
        selected_ranks = [int(x) for x in ranking.split(',')]
        ranked_items = [results[i-1] for i in selected_ranks if 0 < i <= len(results)]
        to_plot.extend(ranked_items)
        title_parts.append(f"Selected ranks: {ranking}")

    # Remove duplicates while preserving order
    seen = set()
    unique_to_plot = []
    for item in to_plot:
        if item[0] not in seen:
            unique_to_plot.append(item)
            seen.add(item[0])
    to_plot = unique_to_plot

    title_suffix = " & ".join(title_parts) if title_parts else "All Selected Headphones"

    # Plot interactive Plotly figure
    if to_plot:
        fig = go.Figure()
        # Target curve (raw, normalized)
        target_resp_norm = normalize_at_1kHz(target_freq, target_resp)
        fig.add_trace(go.Scatter(
            x=target_freq, y=target_resp_norm, mode='lines', name='Target Curve',
            line=dict(color='black', width=2)
        ))

        # Headphones (raw, normalized)
        for fname, rmse, pref in to_plot:
            meas = pd.read_csv(os.path.join(measurements_dir, fname))
            meas_freq = meas.iloc[:, 0].values
            meas_resp = meas.iloc[:, 1].values
            raw_interp = np.interp(target_freq, meas_freq, meas_resp)
            raw_interp_norm = normalize_at_1kHz(target_freq, raw_interp)
            fig.add_trace(go.Scatter(
                x=target_freq, y=raw_interp_norm, mode='lines', name=f"{fname} (RMSE={rmse:.2f}, Pref≈{pref:.2f})"
            ))

        fig.update_layout(
            title=f"{title_suffix} vs Target Curve ({weight_label})",
            xaxis_type='log',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Response (dB)',
            template='plotly_white',
            hovermode='x unified'
        )
        fig.show()

if __name__ == "__main__":
    main()