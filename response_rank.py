#!/usr/bin/env python3
# ResponseRank — RMSE + Pref with Spearman correlation optimization and Plotly plotting

import os
import sys
import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline, interp1d

def error_exit(msg):
    """Print error message and exit cleanly without traceback."""
    print(f"Error: {msg}")
    sys.exit(1)

# === CONFIG ===
REFERENCE_HZ = 1000
ANCHOR_FREQS = []  # Optional anchor points
DEFAULT_POINTS_PER_OCTAVE = 48
MAX_WINDOW_SAMPLES = 50  # Cap on kernel window size to prevent over-smoothing
GAP_THRESHOLD_FACTOR = 5.0  # Factor for detecting large gaps
SMOOTHING_STRENGTH = 0.5
HALF_WIDTH_MULTIPLIER = 4.0
SMOOTHING_FACTOR = 0.001

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

# === Variable log-space smoothing with gap handling ===
def rew_variable_smoothing_logspace_preserve_energy(f, mag_db, strength=SMOOTHING_STRENGTH, half_width_multiplier=HALF_WIDTH_MULTIPLIER):
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

# ----------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank headphones vs. target using optimized RMSE/Pref weights (Spearman correlation) and Plotly plots."
    )
    parser.add_argument("measurements_dir", help="Directory containing measurement CSVs.")
    parser.add_argument("target_path", help="Target curve CSV (freq, dB).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--aweight", action="store_true", help="Use A-weighting for RMSE.")
    group.add_argument("--bweight", action="store_true", help="Use B-weighting for RMSE (medium levels).")
    group.add_argument("--cweight", action="store_true", help="Use C-weighting for RMSE.")
    parser.add_argument("--all-weightings", action="store_true", help="Run all weightings (Flat, A, B, C) and show tonally balanced top headphones.")
    parser.add_argument("--top", type=int, help="Plot / print top N headphones.")
    parser.add_argument("--ranking", type=str, help="Comma-separated rank numbers to plot (e.g. 1,3,5).")
    parser.add_argument("--sort", type=str, choices=["combined","rmse","pref"], default="combined",
                        help="Sort by this metric for table and plot. Default: combined")
    parser.add_argument("--filter", type=str,
                        help="Regular expression to filter headphone filenames (e.g. 'HD[56]00|DT 880').")
    return parser.parse_args()

# ----------------------------------------------------------------------
# Weighting filters
# ----------------------------------------------------------------------
def a_weight(freq):
    f = np.asarray(freq, dtype=float)
    f2 = f**2
    c = 12194.0
    num = (c**2)*(f2**2)
    den = (f2+20.6**2)*(f2+c**2)*np.sqrt((f2+107.7**2)*(f2+737.9**2))
    Ra = num/den
    f1k2 = 1000.0**2
    Ra1k = (c**2)*(f1k2**2)/((f1k2+20.6**2)*(f1k2+c**2)
                              *np.sqrt((f1k2+107.7**2)*(f1k2+737.9**2)))
    return Ra/Ra1k

def b_weight(freq):
    f = np.asarray(freq, dtype=float)
    f2 = f**2
    c = 12194.0
    num = (c**2)*(f2**1.3)
    den = (f2+20.6**2)*(f2+c**2)*np.sqrt((f2+158.5**2)*(f2+737.9**2))
    Rb = num/den
    f1k2 = 1000.0**2
    Rb1k = (c**2)*(f1k2**1.3)/((f1k2+20.6**2)*(f1k2+c**2)*np.sqrt((f1k2+158.5**2)*(f1k2+737.9**2)))
    return Rb/Rb1k

def c_weight(freq):
    f = np.asarray(freq, dtype=float)
    f2 = f**2
    c = 12194.0
    num = (c**2)*f2
    den = (f2+20.6**2)*(f2+c**2)
    Rc = num/den
    f1k2 = 1000.0**2
    Rc1k = (c**2)*f1k2/((f1k2+20.6**2)*(f1k2+c**2))
    return Rc/Rc1k

# ----------------------------------------------------------------------
# Smoothing & normalization
# ----------------------------------------------------------------------
def smooth_one_third_octave_rms(freq, resp_db):
    """
    Apply 1/3rd octave RMS smoothing for preference metric calculation.
    Used only in compute_pref_metrics, not for derived target smoothing.
    """
    freq = np.asarray(freq)
    resp_db = np.asarray(resp_db)
    resp_lin = 10**(resp_db/20.0)
    resp_lin = np.clip(resp_lin, 1e-12, None)
    centers = 20*2**(np.arange(0, int(np.log2(20000/20)*3)+1)/3)
    interp_freq = np.logspace(np.log10(freq.min()), np.log10(freq.max()), 10000)
    interp_lin = np.interp(interp_freq, freq, resp_lin)
    lower = centers*2**(-1/6)
    upper = centers*2**(1/6)
    smoothed_mags = np.array([
        20*np.log10(np.sqrt(np.mean(interp_lin[(interp_freq>=l)&(interp_freq<=u)]**2)))
        if np.any((interp_freq>=l)&(interp_freq<=u)) else np.nan
        for l,u in zip(lower,upper)
    ])
    smoothed_resp = np.interp(freq, centers, smoothed_mags, left=np.nan, right=np.nan)
    smoothed_resp = np.where(np.isnan(smoothed_resp), resp_db, smoothed_resp)
    return smoothed_resp

def normalize_at_1kHz(freq, resp):
    val_1k = np.interp(1000, freq, resp)
    return resp - val_1k

# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def compute_rmse(meas_freq, meas_resp, target_freq, target_resp, weights):
    interp_resp = np.interp(target_freq, meas_freq, meas_resp)
    interp_resp = normalize_at_1kHz(target_freq, interp_resp)
    target_resp = normalize_at_1kHz(target_freq, target_resp)
    return np.sqrt(np.mean(weights*(interp_resp - target_resp)**2))

def compute_pref_metrics(meas_freq, meas_resp, target_freq, target_resp):
    meas_interp = np.interp(target_freq, meas_freq, meas_resp)
    meas_smooth = smooth_one_third_octave_rms(target_freq, meas_interp)
    meas_smooth = normalize_at_1kHz(target_freq, meas_smooth)
    target_norm = normalize_at_1kHz(target_freq, target_resp)
    mask = (target_freq>=20)&(target_freq<=10000)
    error_band = meas_smooth[mask]-target_norm[mask]
    freq_band = target_freq[mask]
    sd = np.std(error_band)
    x = np.log10(freq_band).reshape(-1,1)
    model = LinearRegression().fit(x, error_band)
    as_value = abs(model.coef_[0])
    return 114.49 - 12.62*sd - 15.52*as_value

# ----------------------------------------------------------------------
# Spearman optimization
# ----------------------------------------------------------------------
def find_optimal_weights(rmses, prefs):
    rmses = np.array(rmses)
    prefs = np.array(prefs)
    if rmses.max() == rmses.min():
        rmses_n = np.zeros_like(rmses)
    else:
        rmses_n = (rmses - rmses.min()) / (rmses.max()-rmses.min())
    if prefs.max() == prefs.min():
        prefs_n = np.zeros_like(prefs)
    else:
        prefs_n = (prefs - prefs.min()) / (prefs.max()-prefs.min())

    best_rho, best_w = -1, None
    for w_rmse in np.linspace(0, 1, 101):
        w_pref = 1 - w_rmse
        combined = w_pref*prefs_n + (1-w_pref)*(1 - rmses_n)
        rho_rmse,_ = spearmanr(combined, -rmses)
        rho_pref,_ = spearmanr(combined, prefs)
        avg_rho = (rho_rmse + rho_pref)/2
        if np.isnan(avg_rho):
            continue
        if avg_rho > best_rho:
            best_rho, best_w = avg_rho, (1-w_pref, w_pref)
    return best_w, best_rho

# ----------------------------------------------------------------------
# Plot helper
# ----------------------------------------------------------------------
def plot_headphones(measurements_dir, target_freq, target_resp, ranking, top=None, ranking_nums=None, weight_label="Flat", sort_metric="combined"):
    to_plot = []
    title_parts = []
    if top:
        to_plot.extend(ranking[:top])
        title_parts.append(f"Top {top}")
    if ranking_nums:
        to_plot.extend([ranking[i-1] for i in ranking_nums if 0<i<=len(ranking)])
        title_parts.append(f"Ranks {','.join(map(str, ranking_nums))}")
    seen = set()
    unique = []
    for r in to_plot:
        if r[0] not in seen:
            unique.append(r)
            seen.add(r[0])
    to_plot = unique
    title_suffix = " & ".join(title_parts) if title_parts else "All Headphones"
    if to_plot:
        fig = go.Figure()
        target_norm = normalize_at_1kHz(target_freq, target_resp)
        fig.add_trace(go.Scatter(x=target_freq, y=target_norm, mode='lines',
                                 name='Target Curve', line=dict(color='black', width=2)))
        for fname, rmse, pref, comb in to_plot:
            meas_path = os.path.join(measurements_dir, fname)
            if os.path.exists(meas_path):
                try:
                    meas = pd.read_csv(meas_path, sep=',', encoding='utf-8')
                    if meas.shape[1] < 2:
                        raise ValueError("Measurement CSV must have at least two columns.")
                    first_row = meas.columns[:2].to_list()
                    is_numeric = not np.isnan(pd.to_numeric(first_row, errors='coerce')).all()
                    if is_numeric:
                        print(f"Warning: No headers detected in {fname} (first row is numeric). Using default headers 'frequency,raw'.")
                        meas = pd.read_csv(meas_path, header=None, names=["frequency", "raw"], sep=',', encoding='utf-8')
                    meas_freq = pd.to_numeric(meas.iloc[:, 0], errors='raise').to_numpy()
                    meas_resp = pd.to_numeric(meas.iloc[:, 1], errors='raise').to_numpy()
                    meas_norm = normalize_at_1kHz(meas_freq, meas_resp)
                except Exception as e:
                    print(f"Skipping {fname} in plot: {e}")
                    continue
                clean_name = os.path.splitext(fname)[0]
                if sort_metric == "AvgNormalizedRank":
                    label = f"{clean_name} (AvgNorm={comb:.3f})"
                else:
                    label = f"{clean_name} (RMSE={rmse:.2f}, Pref={pref:.2f}, Comb={comb:.2f})"
                fig.add_trace(go.Scatter(
                    x=meas_freq, y=meas_norm, mode='lines',
                    name=label
                ))
        fig.update_layout(
            title=f"{title_suffix} vs Target ({weight_label})",
            xaxis_type='log', xaxis_title='Frequency (Hz)', yaxis_title='Response (dB)',
            template='plotly_white', hovermode='x unified',
            margin=dict(t=100, b=80),
            annotations=[
                dict(x=0, y=0, xref='paper', yref='paper',
                     text=f'Sort metric: {sort_metric}; Pref uses 1/3-octave smoothed data; curves are raw',
                     showarrow=False, font=dict(size=10), xanchor='left', yanchor='bottom')
            ]
        )
        fig.show()

# ----------------------------------------------------------------------
# Core ranking function
# ----------------------------------------------------------------------
def rank_headphones(measurements_dir, target_freq, target_resp, weight_func, weight_label,
                    sort_metric="combined", verbose=True, top=None, regex_filter=None):
    """
    Rank headphones in a directory against a target response using RMSE and Pref metrics.
    """
    weights = weight_func(target_freq)
    results = []

    pattern = re.compile(regex_filter, re.IGNORECASE) if regex_filter else None

    for fname in sorted(os.listdir(measurements_dir)):
        if not fname.endswith(".csv"):
            continue
        if pattern and not pattern.search(fname):
            continue
        meas_path = os.path.join(measurements_dir, fname)
        try:
            meas = pd.read_csv(meas_path, sep=',', encoding='utf-8')
            if meas.shape[1] < 2:
                raise ValueError("Measurement CSV must have at least two columns.")
            first_row = meas.columns[:2].to_list()
            is_numeric = not np.isnan(pd.to_numeric(first_row, errors='coerce')).all()
            if is_numeric:
                print(f"Warning: No headers detected in {fname} (first row is numeric). Using default headers 'frequency,raw'.")
                meas = pd.read_csv(meas_path, header=None, names=["frequency", "raw"], sep=',', encoding='utf-8')
            col1 = meas.iloc[:, 0].astype(str)
            col2 = meas.iloc[:, 1].astype(str)
            non_numeric_col1 = col1[~pd.to_numeric(col1, errors='coerce').notna()]
            non_numeric_col2 = col2[~pd.to_numeric(col2, errors='coerce').notna()]
            if not non_numeric_col1.empty or not non_numeric_col2.empty:
                raise ValueError(f"Non-numeric values found in {fname}: {non_numeric_col1.tolist()[:5]}, {non_numeric_col2.tolist()[:5]}")
            meas_freq = pd.to_numeric(meas.iloc[:, 0], errors='raise').to_numpy()
            meas_resp = pd.to_numeric(meas.iloc[:, 1], errors='raise').to_numpy()
            if len(meas_freq) < 2:
                raise ValueError("Measurement CSV must have at least 2 data points.")
            if not np.all(np.isfinite(meas_freq)) or not np.all(np.isfinite(meas_resp)):
                raise ValueError("Measurement frequency and dB values must be finite numbers.")
            if np.any(meas_freq <= 0):
                raise ValueError("Measurement frequency values must be positive.")
            if np.any(np.diff(meas_freq) <= 0):
                raise ValueError("Measurement frequency values must be strictly increasing.")
            rmse = compute_rmse(meas_freq, meas_resp, target_freq, target_resp, weights)
            pref = compute_pref_metrics(meas_freq, meas_resp, target_freq, target_resp)
            results.append((fname, rmse, pref))
        except ValueError as e:
            if verbose:
                print(f"Skipping {fname}: Invalid data - {e}")
        except Exception as e:
            if verbose:
                print(f"Skipping {fname}: {e}")

    if not results:
        return []

    names, rmses, prefs = zip(*results)

    (w_rmse, w_pref), best_rho = find_optimal_weights(rmses, prefs)

    if verbose:
        print(f"\n{weight_label} → Optimal weights → RMSE={w_rmse:.2f}, Pref={w_pref:.2f}, Spearman ρ≈{best_rho:.4f}\n")

    rmses = np.array(rmses)
    prefs = np.array(prefs)
    rmses_n = (rmses - rmses.min())/(rmses.max()-rmses.min()) if rmses.max() != rmses.min() else np.zeros_like(rmses)
    prefs_n = (prefs - prefs.min())/(prefs.max()-prefs.min()) if prefs.max() != prefs.min() else np.zeros_like(prefs)
    combined = w_pref * prefs_n + (1 - w_pref) * (1 - rmses_n)

    sort_idx_map = {"rmse": 1, "pref": 2, "combined": 3}
    sort_idx = sort_idx_map.get(sort_metric, 3)
    reverse = True if sort_metric != "rmse" else False

    ranking = sorted(zip(names, rmses, prefs, combined), key=lambda x: x[sort_idx], reverse=reverse)

    if verbose:
        top_to_show = min(top or len(ranking), len(ranking))
        print(f"Ranked headphones (sorted by {sort_metric}):")
        for i, (fname, rmse, pref, comb) in enumerate(ranking[:top_to_show], 1):
            clean_name = os.path.splitext(fname)[0]
            print(f"{i:3d}. {clean_name:<65} RMSE={rmse:6.3f}  Pref≈{pref:7.2f}  Combined≈{comb:.3f}")

    return ranking

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    try:
        target = pd.read_csv(args.target_path, sep=',', encoding='utf-8')
        first_row = target.columns[:2].to_list()
        is_numeric = not np.isnan(pd.to_numeric(first_row, errors='coerce')).all()
        if is_numeric:
            print("Warning: No headers detected in target CSV (first row is numeric). Using default headers 'frequency,raw'.")
            target = pd.read_csv(args.target_path, header=None, names=["frequency", "raw"], sep=',', encoding='utf-8')
    except pd.errors.EmptyDataError:
        error_exit("Target CSV file is empty or contains no valid data.")
    except FileNotFoundError:
        error_exit(f"Target CSV file not found: {args.target_path}")
    except pd.errors.ParserError:
        error_exit("Failed to parse target CSV file. Check delimiter (expected comma) and file format.")
    except Exception as e:
        error_exit(f"Failed to read target CSV file: {str(e)}")

    if target.shape[1] < 2:
        error_exit("Target CSV must have at least two columns (frequency and dB).")

    try:
        col1 = target.iloc[:, 0].astype(str)
        col2 = target.iloc[:, 1].astype(str)
        non_numeric_col1 = col1[~pd.to_numeric(col1, errors='coerce').notna()]
        non_numeric_col2 = col2[~pd.to_numeric(col2, errors='coerce').notna()]
        if not non_numeric_col1.empty or not non_numeric_col2.empty:
            error_exit(f"Non-numeric values found in target CSV: {non_numeric_col1.tolist()[:5]}, {non_numeric_col2.tolist()[:5]}")
        target_freq = pd.to_numeric(target.iloc[:, 0], errors='raise').to_numpy()
        target_resp = pd.to_numeric(target.iloc[:, 1], errors='raise').to_numpy()
    except ValueError as e:
        error_exit(f"Target CSV first two columns must contain numeric values. Error: {str(e)}")

    if len(target_freq) < 2:
        error_exit("Target CSV must have at least 2 data points.")

    if not np.all(np.isfinite(target_freq)) or not np.all(np.isfinite(target_resp)):
        error_exit("Target frequency and dB values must be finite numbers.")

    if np.any(target_freq <= 0):
        error_exit("Target frequency values must be positive.")

    if np.any(np.diff(target_freq) <= 0):
        error_exit("Target frequency values must be strictly increasing.")

    if args.all_weightings:
        weightings = [
            (lambda f: np.ones_like(f), "Flat"),
            (a_weight, "A-weighted"),
            (b_weight, "B-weighted"),
            (c_weight, "C-weighted")
        ]

        all_rankings = []
        for func, label in weightings:
            ranking = rank_headphones(
                args.measurements_dir,
                target_freq,
                target_resp,
                func,
                label,
                args.sort,
                verbose=True,
                top=args.top,
                regex_filter=args.filter
            )
            all_rankings.append(ranking)

        top_n = args.top or len(all_rankings[0])
        top_sets = [set([r[0] for r in ranking[:top_n]]) for ranking in all_rankings]
        consistent_top = set.intersection(*top_sets)

        if not consistent_top:
            print(f"\nNo headphones appear in the top-{top_n} across all weightings.")
            return

        avg_ranks = []
        for fname in consistent_top:
            ranks = []
            for ranking in all_rankings:
                idx = next((i for i, r in enumerate(ranking[:top_n]) if r[0] == fname), top_n)
                ranks.append(1.0 - (idx / top_n))
            avg_ranks.append((fname, float(np.mean(ranks))))

        df_consistent = pd.DataFrame(avg_ranks, columns=["Headphone", "AvgNormalizedRank"])
        df_consistent = df_consistent.sort_values(by="AvgNormalizedRank", ascending=False).reset_index(drop=True)

        top_to_show = min(args.top, len(df_consistent)) if args.top else len(df_consistent)
        print(f"\nTop-{top_to_show} tonally balanced headphones by average normalized rank:\n")

        for i, row in enumerate(df_consistent.head(top_to_show).itertuples(), 1):
            clean_name = os.path.splitext(row.Headphone)[0]
            print(f"{i:3d}. {clean_name:<50} AvgNormalizedRank={row.AvgNormalizedRank:.3f}")

        rank_map_first = {r[0]: (r[1], r[2], r[3]) for r in all_rankings[0]}
        filtered = []
        for fname, avg in df_consistent.itertuples(index=False):
            if fname in rank_map_first:
                rmse, pref, _ = rank_map_first[fname]
                filtered.append((fname, rmse, pref, avg))
        filtered_sorted = sorted(filtered, key=lambda x: x[3], reverse=True)

        # -----------------------------
        # Compute derived target curve
        # -----------------------------
        derived_responses = []
        for fname, rmse, pref, avg in filtered_sorted[:top_to_show]:
            meas_path = os.path.join(args.measurements_dir, fname)
            try:
                meas = pd.read_csv(meas_path, sep=',', encoding='utf-8')
                if meas.shape[1] < 2:
                    raise ValueError("Measurement CSV must have at least two columns.")
                first_row = meas.columns[:2].to_list()
                is_numeric = not np.isnan(pd.to_numeric(first_row, errors='coerce')).all()
                if is_numeric:
                    print(f"Warning: No headers detected in {fname} (first row is numeric). Using default headers 'frequency,raw'.")
                    meas = pd.read_csv(meas_path, header=None, names=["frequency", "raw"], sep=',', encoding='utf-8')
                col1 = meas.iloc[:, 0].astype(str)
                col2 = meas.iloc[:, 1].astype(str)
                non_numeric_col1 = col1[~pd.to_numeric(col1, errors='coerce').notna()]
                non_numeric_col2 = col2[~pd.to_numeric(col2, errors='coerce').notna()]
                if not non_numeric_col1.empty or not non_numeric_col2.empty:
                    raise ValueError(f"Non-numeric values found in {fname}: {non_numeric_col1.tolist()[:5]}, {non_numeric_col2.tolist()[:5]}")
                freq = pd.to_numeric(meas.iloc[:, 0], errors='raise').to_numpy()
                resp = pd.to_numeric(meas.iloc[:, 1], errors='raise').to_numpy()
                if len(freq) < 2:
                    raise ValueError("Measurement CSV must have at least 2 data points.")
                if not np.all(np.isfinite(freq)) or not np.all(np.isfinite(resp)):
                    raise ValueError("Measurement frequency and dB values must be finite numbers.")
                if np.any(freq <= 0):
                    raise ValueError("Measurement frequency values must be positive.")
                if np.any(np.diff(freq) <= 0):
                    raise ValueError("Measurement frequency values must be strictly increasing.")
                interp_resp = np.interp(target_freq, freq, resp)
                norm_resp = normalize_at_1kHz(target_freq, interp_resp)
                derived_responses.append(norm_resp)
            except Exception as e:
                print(f"Skipping {fname} in derived target computation: {e}")
                continue

        if not derived_responses:
            print("No valid measurement data available for derived target computation.")
            return

        derived_target = np.mean(derived_responses, axis=0)
        f = target_freq
        y_raw = derived_target

        # Apply variable log-space smoothing
        logf = np.log10(f)
        log_diff = np.diff(logf)
        mean_diff = np.mean(log_diff)
        has_large_gaps = np.any(log_diff > GAP_THRESHOLD_FACTOR * mean_diff)
        if has_large_gaps:
            print("Warning: Large frequency gaps detected in derived target, resampling to uniform grid.")
            f, y_raw = resample_log_frequency(f, y_raw)

        y_smooth = rew_variable_smoothing_logspace_preserve_energy(
            f, y_raw, strength=SMOOTHING_STRENGTH, half_width_multiplier=HALF_WIDTH_MULTIPLIER
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
            spline = UnivariateSpline(logf, y_smooth, w=weights, s=len(f) * SMOOTHING_FACTOR)
            ideal = spline(logf)

        # Normalize to reference
        ref = np.interp(REFERENCE_HZ, f, ideal)
        derived_target_smooth = ideal - ref

        # Plot all together
        fig = go.Figure()
        target_norm = normalize_at_1kHz(target_freq, target_resp)
        fig.add_trace(go.Scatter(
            x=target_freq, y=target_norm, mode='lines',
            name='Original Target', line=dict(color='deepskyblue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=f, y=derived_target_smooth, mode='lines',
            name=f'Derived Target (empirical neutral, strength={SMOOTHING_STRENGTH:.2f}, s={SMOOTHING_FACTOR:.3f})',
            line=dict(color='lime', width=3, dash='dot')
        ))

        for fname, rmse, pref, avg in filtered_sorted[:top_to_show]:
            meas_path = os.path.join(args.measurements_dir, fname)
            try:
                meas = pd.read_csv(meas_path, sep=',', encoding='utf-8')
                if meas.shape[1] < 2:
                    raise ValueError("Measurement CSV must have at least two columns.")
                first_row = meas.columns[:2].to_list()
                is_numeric = not np.isnan(pd.to_numeric(first_row, errors='coerce')).all()
                if is_numeric:
                    print(f"Warning: No headers detected in {fname} (first row is numeric). Using default headers 'frequency,raw'.")
                    meas = pd.read_csv(meas_path, header=None, names=["frequency", "raw"], sep=',', encoding='utf-8')
                freq = pd.to_numeric(meas.iloc[:, 0], errors='raise').to_numpy()
                resp = pd.to_numeric(meas.iloc[:, 1], errors='raise').to_numpy()
                meas_norm = normalize_at_1kHz(freq, resp)
                clean_name = os.path.splitext(fname)[0]
                fig.add_trace(go.Scatter(
                    x=freq, y=meas_norm, mode='lines',
                    name=f"{clean_name} (AvgNorm={avg:.3f})"
                ))
            except Exception as e:
                print(f"Skipping {fname} in plot: {e}")
                continue

        fig.update_layout(
            title=f"Tonally Balanced Top-{top_to_show} vs Targets",
            xaxis_type='log', xaxis_title='Frequency (Hz)',
            yaxis_title='Response (dB)',
            template='plotly_dark', hovermode='x unified',
            margin=dict(t=100, b=80)
        )
        fig.show()

        # Export to CSV
        print("Writing output CSV with headers: ['frequency', 'raw']")
        pd.DataFrame({'frequency': f, 'raw': derived_target_smooth}).to_csv(
            "Derived_Target.csv", index=False, header=True
        )

    else:
        if args.aweight:
            weight_func, weight_label = a_weight, "A-weighted"
        elif args.bweight:
            weight_func, weight_label = b_weight, "B-weighted"
        elif args.cweight:
            weight_func, weight_label = c_weight, "C-weighted"
        else:
            weight_func, weight_label = lambda f: np.ones_like(f), "Flat"

        ranking = rank_headphones(
            args.measurements_dir,
            target_freq,
            target_resp,
            weight_func,
            weight_label,
            args.sort,
            verbose=True,
            top=args.top,
            regex_filter=args.filter
        )

        plot_headphones(
            args.measurements_dir,
            target_freq,
            target_resp,
            ranking,
            top=args.top,
            ranking_nums=[int(x) for x in args.ranking.split(',')] if args.ranking else None,
            weight_label=weight_label,
            sort_metric=args.sort
        )

if __name__ == "__main__":
    main()
