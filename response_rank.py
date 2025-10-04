#!/usr/bin/env python3

# ResponseRank
# Rank headphone frequency response measurements against a target curve

import os
import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot top N headphone measurements against a target curve, with optional weighting (interactive)."
    )
    parser.add_argument("measurements_dir", help="Directory containing measurement CSV files.")
    parser.add_argument("target_path", help="Path to the target CSV file (frequency in first column, response in second).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--aweight", action="store_true", help="Use A-weighting for RMSE calculation.")
    group.add_argument("--cweight", action="store_true", help="Use C-weighting for RMSE calculation.")
    parser.add_argument("--top", type=int, help="Number of top headphones to plot.")
    parser.add_argument("--ranking", type=str, help="Comma-separated list of rank numbers to plot (e.g., 1,3,5).")
    return parser.parse_args()

# ----------------------------------------------------------------------
# Weighting filters
# ----------------------------------------------------------------------

def a_weight(freq):
    f = np.asarray(freq, dtype=float)
    f2 = f**2
    c = 12194.0
    num = (c**2) * (f2**2)
    den = (f2 + 20.6**2) * (f2 + c**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
    Ra = num / den
    f1k2 = 1000.0**2
    Ra1k = (c**2) * (f1k2**2) / ((f1k2 + 20.6**2) * (f1k2 + c**2)
                                  * np.sqrt((f1k2 + 107.7**2) * (f1k2 + 737.9**2)))
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

# ----------------------------------------------------------------------
# Smoothing and normalization
# ----------------------------------------------------------------------

def smooth_one_third_octave_rms(freq, resp_db):
    """
    True 1/3-octave RMS smoothing per AES17-2015 using predefined center frequencies.
    Only measurement is smoothed.
    """
    freq = np.asarray(freq)
    resp_db = np.asarray(resp_db)
    resp_lin = 10 ** (resp_db / 20.0)
    resp_lin = np.clip(resp_lin, 1e-12, None)  # avoid log10(0)

    # 1/3-octave centers from 20 Hz to 20 kHz
    centers = 20 * 2 ** (np.arange(0, int(np.log2(20000/20)*3)+1) / 3)

    # Dense interpolation
    interp_freq = np.logspace(np.log10(freq.min()), np.log10(freq.max()), 10000)
    interp_lin = np.interp(interp_freq, freq, resp_lin)

    # RMS in each 1/3-octave band
    lower = centers * 2**(-1/6)
    upper = centers * 2**(1/6)
    smoothed_mags = np.array([
        20 * np.log10(np.sqrt(np.mean(interp_lin[(interp_freq >= l) & (interp_freq <= u)]**2)))
        if np.any((interp_freq >= l) & (interp_freq <= u)) else np.nan
        for l, u in zip(lower, upper)
    ])

    # Interpolate back to original frequencies
    smoothed_resp = np.interp(freq, centers, smoothed_mags, left=np.nan, right=np.nan)
    smoothed_resp = np.where(np.isnan(smoothed_resp), resp_db, smoothed_resp)
    return smoothed_resp

def normalize_at_1kHz(freq, resp):
    """Normalize response to 0 dB at 1 kHz."""
    val_1k = np.interp(1000, freq, resp)
    return resp - val_1k

# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def compute_rmse(meas_freq, meas_resp, target_freq, target_resp, weights):
    """Compute RMSE on normalized data."""
    interp_resp = np.interp(target_freq, meas_freq, meas_resp)
    interp_resp = normalize_at_1kHz(target_freq, interp_resp)
    target_resp = normalize_at_1kHz(target_freq, target_resp)
    rmse = np.sqrt(np.mean(weights * (interp_resp - target_resp)**2))
    return rmse

def compute_pref_metrics(meas_freq, meas_resp, target_freq, target_resp):
    """
    Compute AES 2018 preference score (Pref) for a headphone measurement.
    Only the measurement is smoothed per AES17/AES2015; target remains raw.
    """
    # Interpolate measurement to target frequencies
    meas_interp = np.interp(target_freq, meas_freq, meas_resp)

    # Smooth the measurement only
    meas_smooth = smooth_one_third_octave_rms(target_freq, meas_interp)

    # Normalize both at 1 kHz
    meas_smooth = normalize_at_1kHz(target_freq, meas_smooth)
    target_norm = normalize_at_1kHz(target_freq, target_resp)

    # Compute error in 20 Hz – 10 kHz
    mask = (target_freq >= 20) & (target_freq <= 10000)
    error_band = meas_smooth[mask] - target_norm[mask]
    freq_band = target_freq[mask]

    # SD: standard deviation of the error
    sd = np.std(error_band)

    # AS: absolute slope of linear regression vs log frequency
    x = np.log10(freq_band).reshape(-1, 1)
    model = LinearRegression().fit(x, error_band)
    as_value = np.abs(model.coef_[0])

    # Preference score (Olive et al. 2018)
    pref = 114.49 - (12.62 * sd) - (15.52 * as_value)
    return pref

# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------

def main():
    args = parse_args()
    measurements_dir = args.measurements_dir
    target_path = args.target_path
    use_a = args.aweight
    use_c = args.cweight
    top_n = args.top
    ranking = args.ranking

    # Load target curve
    target = pd.read_csv(target_path)
    target_freq = target.iloc[:, 0].values
    target_resp = target.iloc[:, 1].values

    # Select weighting
    if use_a:
        weights = a_weight(target_freq)
        weight_label = "A-weighted"
    elif use_c:
        weights = c_weight(target_freq)
        weight_label = "C-weighted"
    else:
        weights = np.ones_like(target_freq)
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

    # Remove duplicates
    seen = set()
    unique_to_plot = []
    for item in to_plot:
        if item[0] not in seen:
            unique_to_plot.append(item)
            seen.add(item[0])
    to_plot = unique_to_plot
    title_suffix = " & ".join(title_parts) if title_parts else "All Selected Headphones"

    # Interactive Plotly figure
    if to_plot:
        fig = go.Figure()

        # Normalize and plot target curve
        target_norm = normalize_at_1kHz(target_freq, target_resp)
        fig.add_trace(go.Scatter(
            x=target_freq, y=target_norm, mode='lines', name='Target Curve',
            line=dict(color='black', width=2)
        ))

        # Plot each headphone
        for fname, rmse, pref in to_plot:
            meas = pd.read_csv(os.path.join(measurements_dir, fname))
            meas_freq = meas.iloc[:, 0].values
            meas_resp = meas.iloc[:, 1].values

            # Plot raw measurement normalized at 1 kHz
            meas_norm = normalize_at_1kHz(meas_freq, meas_resp)

            fig.add_trace(go.Scatter(
                x=meas_freq, y=meas_norm, mode='lines',
                name=f"{fname} (RMSE={rmse:.2f}, Pref≈{pref:.2f})"
            ))

        # Layout with annotation below the plot
        fig.update_layout(
            title=f"{title_suffix} vs Target Curve ({weight_label})",
            xaxis_type='log',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Response (dB)',
            template='plotly_white',
            hovermode='x unified',
            margin=dict(t=80, b=80),
            annotations=[
                dict(
                    x=0, y=0, xref='paper', yref='paper',
                    text='Pref score computed using 1/3-octave RMS smoothed measurement; plotted curves are raw',
                    showarrow=False, font=dict(size=10), xanchor='left'
                )
            ]
        )
        fig.show()

if __name__ == "__main__":
    main()
