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

def error_exit(msg):
    """Print error message and exit cleanly without traceback."""
    print(f"Error: {msg}")
    sys.exit(1)

# === CONFIG ===
REFERENCE_HZ = 1000

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
    parser.add_argument("--all-weightings", action="store_true", help="Run all weightings (Z, A, B, C) and show tonally balanced top headphones.")
    parser.add_argument("--top", type=int, help="Plot / print top N headphones.")
    parser.add_argument("--ranking", type=str, help="Comma-separated rank numbers to plot (e.g. 1,3,5).")
    parser.add_argument("--sort", type=str, choices=["combined","rmse","pref"], default="combined",
                        help="Sort by this metric for table and plot. Default: combined")
    parser.add_argument("--filter", type=str,
                        help="Regular expression to filter headphone filenames (e.g. 'HD[56]00|DT 880').")
    parser.add_argument("--cutoff", type=float, default=None,
                        help="Minimum frequency (Hz) to include in ranking metrics (RMSE and Pref).")
    parser.add_argument("--overlay", type=str, nargs='+', metavar='CSV',
                        help="One or more CSV files (freq, dB) to overlay on the plot. Not used for ranking.")
    parser.add_argument("-np", "--no-plot", action="store_true",
                        help="Skip the interactive Plotly plot.")
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
def compute_rmse(meas_freq, meas_resp, target_freq, target_resp, weights, cutoff=None):
    interp_resp = np.interp(target_freq, meas_freq, meas_resp)
    interp_resp = normalize_at_1kHz(target_freq, interp_resp)
    target_resp = normalize_at_1kHz(target_freq, target_resp)
    
    mask = np.ones_like(target_freq, dtype=bool)
    if cutoff is not None:
        mask &= (target_freq >= cutoff)
        
    if not np.any(mask):
        return np.nan
        
    return np.sqrt(np.mean(weights[mask]*(interp_resp[mask] - target_resp[mask])**2))

def compute_pref_metrics(meas_freq, meas_resp, target_freq, target_resp, cutoff=None):
    meas_interp = np.interp(target_freq, meas_freq, meas_resp)
    meas_smooth = smooth_one_third_octave_rms(target_freq, meas_interp)
    meas_smooth = normalize_at_1kHz(target_freq, meas_smooth)
    target_norm = normalize_at_1kHz(target_freq, target_resp)
    
    min_f = 20 if cutoff is None else max(20, cutoff)
    mask = (target_freq >= min_f) & (target_freq <= 10000)
    
    if not np.any(mask) or len(target_freq[mask]) < 2:
        return np.nan
        
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
# PCA-based core response
# ----------------------------------------------------------------------
def extract_core_response_svd(measurements):
    M = np.array(measurements)          # (n, f)
    mean_response = M.mean(axis=0)
    X = M - mean_response               # centred
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    ev = (S ** 2) / np.sum(S ** 2)     # explained-variance ratios
    n = M.shape[0]
    amplitude = S[0] / np.sqrt(max(n - 1, 1))
    core = mean_response + Vt[0, :] * amplitude
    return core, ev

# ----------------------------------------------------------------------
# Consensus Response
# ----------------------------------------------------------------------
def compute_consensus_response(rankings, top_to_show, measurements_dir, target_freq):
    derived_responses = []
    for item in rankings[:top_to_show]:
        fname = item[0]
        meas_path = os.path.join(measurements_dir, fname)
        try:
            meas = pd.read_csv(meas_path, sep=',', encoding='utf-8')
            if meas.shape[1] < 2:
                raise ValueError("Measurement CSV must have at least two columns.")
            first_row = meas.columns[:2].to_list()
            is_numeric = not np.isnan(pd.to_numeric(first_row, errors='coerce')).all()
            if is_numeric:
                meas = pd.read_csv(meas_path, header=None, names=["frequency", "raw"], sep=',', encoding='utf-8')
            
            freq = pd.to_numeric(meas.iloc[:, 0], errors='raise').to_numpy()
            resp = pd.to_numeric(meas.iloc[:, 1], errors='raise').to_numpy()
            
            if len(freq) < 2:
                continue
            if not np.all(np.isfinite(freq)) or not np.all(np.isfinite(resp)):
                continue
            if np.any(freq <= 0) or np.any(np.diff(freq) <= 0):
                continue
                
            interp_resp = np.interp(target_freq, freq, resp)
            norm_resp = normalize_at_1kHz(target_freq, interp_resp)
            derived_responses.append(norm_resp)
        except Exception as e:
            print(f"Skipping {fname} in consensus response computation: {e}")
            continue

    if not derived_responses:
        print("No valid measurement data available for consensus response computation.")
        return None, None, None

    n_resp = len(derived_responses)
    if n_resp > 4:
        print(f"Using PC1-based core response (n={n_resp} > 4).")
        derived_target, ev = extract_core_response_svd(derived_responses)
        print(f"  PC1 explains {ev[0]*100:.1f}% of variance "
              f"(PC2: {ev[1]*100:.1f}%, PC3: {ev[2]*100:.1f}%)")
        derived_method_label = f"PC1 core, n={n_resp}"
    else:
        print(f"Using mean response (n={n_resp} ≤ 4).")
        derived_target = np.mean(derived_responses, axis=0)
        derived_method_label = f"mean, n={n_resp}"

    return target_freq, derived_target, derived_method_label

# ----------------------------------------------------------------------
# Plot helper
# ----------------------------------------------------------------------
def plot_headphones(measurements_dir, target_freq, target_resp, ranking, top=None, ranking_nums=None, weight_label="Z-weighted", sort_metric="combined", derived_data=None, cutoff=None, overlay_data=None, filter_str=None):
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
    title = f"{title_suffix} vs Target ({weight_label})"
    if filter_str:
        title += f"  ·  filter: {filter_str}"
    if to_plot:
        fig = go.Figure()
        target_norm = normalize_at_1kHz(target_freq, target_resp)
        fig.add_trace(go.Scatter(x=target_freq, y=target_norm, mode='lines',
                                 name='Target Curve', line=dict(color='deepskyblue', width=2)))

        # Overlay curves (additional reference curves, not part of ranking)
        if overlay_data:
            overlay_colors = ['orange', 'gold', 'salmon', 'darkorange', 'khaki']
            for i, (ov_label, ov_freq, ov_resp) in enumerate(overlay_data):
                color = overlay_colors[i % len(overlay_colors)]
                fig.add_trace(go.Scatter(
                    x=ov_freq, y=ov_resp, mode='lines',
                    name=ov_label,
                    line=dict(color=color, width=2, dash='dash')
                ))

        # Add consensus response if computed
        if derived_data is not None:
            f_der, y_der, label_der = derived_data
            if f_der is not None and y_der is not None:
                fig.add_trace(go.Scatter(
                    x=f_der, y=y_der, mode='lines',
                    name=f'Consensus Response ({label_der})',
                    line=dict(color='lime', width=3, dash='dot')
                ))

        for item in to_plot:
            fname = item[0]
            meas_path = os.path.join(measurements_dir, fname)
            if os.path.exists(meas_path):
                try:
                    meas = pd.read_csv(meas_path, sep=',', encoding='utf-8')
                    if meas.shape[1] < 2:
                        raise ValueError("Measurement CSV must have at least two columns.")
                    first_row = meas.columns[:2].to_list()
                    is_numeric = not np.isnan(pd.to_numeric(first_row, errors='coerce')).all()
                    if is_numeric:
                        meas = pd.read_csv(meas_path, header=None, names=["frequency", "raw"], sep=',', encoding='utf-8')
                    meas_freq = pd.to_numeric(meas.iloc[:, 0], errors='raise').to_numpy()
                    meas_resp = pd.to_numeric(meas.iloc[:, 1], errors='raise').to_numpy()
                    meas_norm = normalize_at_1kHz(meas_freq, meas_resp)
                except Exception as e:
                    print(f"Skipping {fname} in plot: {e}")
                    continue
                clean_name = os.path.splitext(fname)[0]
                
                # Handling single weight mode vs all-weightings mode label parsing
                if sort_metric == "AvgNormalizedRank" and len(item) == 4:
                    label = f"{clean_name} (AvgNorm={item[3]:.3f})"
                elif len(item) == 4:
                    rmse, pref, comb = item[1], item[2], item[3]
                    label = f"{clean_name} (RMSE={rmse:.2f}, Pref={pref:.2f}, Comb={comb:.2f})"
                else:
                    label = clean_name
                    
                fig.add_trace(go.Scatter(
                    x=meas_freq, y=meas_norm, mode='lines',
                    name=label
                ))
        
        # Add a visual indicator for the ranking cutoff frequency if provided
        if cutoff is not None:
            fig.add_vline(x=cutoff, line_width=1, line_dash="dash", line_color="gray", 
                          annotation_text=f"Rank Cutoff: {cutoff}Hz", annotation_position="top right")

        fig.update_layout(
            title=title,
            xaxis_type='log', xaxis_title='Frequency (Hz)', yaxis_title='Response (dB)',
            template='plotly_dark', hovermode='x unified',
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
                    sort_metric="combined", verbose=True, top=None, regex_filter=None, cutoff=None):
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
                meas = pd.read_csv(meas_path, header=None, names=["frequency", "raw"], sep=',', encoding='utf-8')
            col1 = meas.iloc[:, 0].astype(str)
            col2 = meas.iloc[:, 1].astype(str)
            non_numeric_col1 = col1[~pd.to_numeric(col1, errors='coerce').notna()]
            non_numeric_col2 = col2[~pd.to_numeric(col2, errors='coerce').notna()]
            if not non_numeric_col1.empty or not non_numeric_col2.empty:
                raise ValueError(f"Non-numeric values found in {fname}")
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
            
            rmse = compute_rmse(meas_freq, meas_resp, target_freq, target_resp, weights, cutoff)
            pref = compute_pref_metrics(meas_freq, meas_resp, target_freq, target_resp, cutoff)
            
            if np.isnan(rmse) or np.isnan(pref):
                raise ValueError(f"Cutoff ({cutoff}Hz) resulted in invalid metric computations.")
                
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
            print("Warning: No headers detected in target CSV. Using default headers 'frequency,raw'.")
            target = pd.read_csv(args.target_path, header=None, names=["frequency", "raw"], sep=',', encoding='utf-8')
    except Exception as e:
        error_exit(f"Failed to read target CSV file: {str(e)}")

    if target.shape[1] < 2:
        error_exit("Target CSV must have at least two columns (frequency and dB).")

    try:
        target_freq = pd.to_numeric(target.iloc[:, 0], errors='raise').to_numpy()
        target_resp = pd.to_numeric(target.iloc[:, 1], errors='raise').to_numpy()
    except ValueError as e:
        error_exit(f"Target CSV first two columns must contain numeric values. Error: {str(e)}")

    if len(target_freq) < 2 or not np.all(np.isfinite(target_freq)) or not np.all(np.isfinite(target_resp)):
        error_exit("Target frequency and dB values must be valid, finite numbers with at least 2 points.")
    if np.any(target_freq <= 0) or np.any(np.diff(target_freq) <= 0):
        error_exit("Target frequency values must be positive and strictly increasing.")

    # Load overlay curves (normalised at 1 kHz, same as everything else in the plot).
    overlay_data = []
    for ov_path in (args.overlay or []):
        try:
            ov = pd.read_csv(ov_path, sep=',', encoding='utf-8')
            first_row = ov.columns[:2].to_list()
            if not np.isnan(pd.to_numeric(first_row, errors='coerce')).all():
                ov = pd.read_csv(ov_path, header=None, names=["frequency", "raw"], sep=',', encoding='utf-8')
            ov_freq = pd.to_numeric(ov.iloc[:, 0], errors='raise').to_numpy()
            ov_resp = pd.to_numeric(ov.iloc[:, 1], errors='raise').to_numpy()
            ov_norm = normalize_at_1kHz(ov_freq, ov_resp)
            ov_label = os.path.splitext(os.path.basename(ov_path))[0]
            overlay_data.append((ov_label, ov_freq, ov_norm))
        except Exception as e:
            print(f"Warning: could not load overlay '{ov_path}': {e}")

    if args.all_weightings:
        weightings = [
            (lambda f: np.ones_like(f), "Z-weighted"),
            (a_weight, "A-weighted"),
            (b_weight, "B-weighted"),
            (c_weight, "C-weighted")
        ]

        all_rankings = []
        for func, label in weightings:
            ranking = rank_headphones(
                args.measurements_dir, target_freq, target_resp,
                func, label, args.sort, verbose=True, top=args.top, regex_filter=args.filter, cutoff=args.cutoff
            )
            all_rankings.append(ranking)

        ranking_nums_list = [int(x) for x in args.ranking.split(',')] if args.ranking else []
        max_rank = max(ranking_nums_list) if ranking_nums_list else 0

        # Score every headphone across all weightings — no intersection required.
        # Using the full ranking length as the denominator means a headphone ranked
        # last still gets a defined (low) score rather than being silently excluded.
        all_fnames = {r[0] for ranking in all_rankings for r in ranking}
        avg_ranks = []
        for fname in all_fnames:
            ranks = []
            for ranking in all_rankings:
                n = len(ranking)
                idx = next((i for i, r in enumerate(ranking) if r[0] == fname), n)
                ranks.append(1.0 - (idx / n))
            avg_ranks.append((fname, float(np.mean(ranks))))

        if not avg_ranks:
            print("\nNo headphones found.")
            return

        df_consistent = pd.DataFrame(avg_ranks, columns=["Headphone", "AvgNormalizedRank"])
        df_consistent = df_consistent.sort_values(by="AvgNormalizedRank", ascending=False).reset_index(drop=True)

        # table_n: how many rows to print.
        #   --top N            → N
        #   --ranking only     → max requested rank (show context up to that point)
        #   neither            → all headphones (current default behaviour)
        if args.top:
            table_n = args.top
        elif ranking_nums_list:
            table_n = max_rank
        else:
            table_n = len(all_rankings[0])

        # build_n: filtered_sorted must reach at least max_rank so ranking lookups work.
        build_n = max(table_n, max_rank)

        top_to_show  = min(table_n, len(df_consistent))
        build_to_show = min(build_n, len(df_consistent))

        print(f"\nTop-{top_to_show} tonally balanced headphones by average normalized rank:\n")
        for i, row in enumerate(df_consistent.head(top_to_show).itertuples(), 1):
            clean_name = os.path.splitext(row.Headphone)[0]
            print(f"{i:3d}. {clean_name:<50} AvgNormalizedRank={row.AvgNormalizedRank:.3f}")

        rank_map_first = {r[0]: (r[1], r[2], r[3]) for r in all_rankings[0]}
        filtered_sorted = []
        for fname, avg in df_consistent.head(build_to_show).itertuples(index=False):
            if fname in rank_map_first:
                rmse, pref, _ = rank_map_first[fname]
                filtered_sorted.append((fname, rmse, pref, avg))
        filtered_sorted = sorted(filtered_sorted, key=lambda x: x[3], reverse=True)

        # Print any --ranking entries that fell outside the table block.
        extras = sorted(n for n in ranking_nums_list if n > top_to_show)
        if extras:
            print(f"\nRanked selections outside top-{top_to_show}:")
            for n in extras:
                if 0 < n <= len(filtered_sorted):
                    fn, rmse, pref, avg = filtered_sorted[n - 1]
                    clean_name = os.path.splitext(fn)[0]
                    print(f"{n:3d}. {clean_name:<50} AvgNormalizedRank={avg:.3f}")

        # Build the exact plotted set (top N ∪ ranking selections, deduplicated).
        plotted = list(filtered_sorted[:args.top]) if args.top else []
        seen_fnames = {r[0] for r in plotted}
        for n in ranking_nums_list:
            if 0 < n <= len(filtered_sorted) and filtered_sorted[n - 1][0] not in seen_fnames:
                plotted.append(filtered_sorted[n - 1])
                seen_fnames.add(filtered_sorted[n - 1][0])
        if not plotted:
            plotted = list(filtered_sorted[:top_to_show])

        # Compute and plot consensus response
        f_der, y_der, label_der = compute_consensus_response(plotted, len(plotted), args.measurements_dir, target_freq)
        if f_der is not None:
            print("Writing consensus response CSV with headers: ['frequency', 'raw']")
            pd.DataFrame({'frequency': f_der, 'raw': y_der}).to_csv("Consensus_Response.csv", index=False, header=True)

        if not args.no_plot:
            plot_top = args.top if args.top else (None if ranking_nums_list else top_to_show)
            plot_headphones(
                args.measurements_dir, target_freq, target_resp, filtered_sorted,
                top=plot_top,
                ranking_nums=ranking_nums_list if ranking_nums_list else None,
                weight_label="Tonally Balanced (All Weightings)",
                sort_metric="AvgNormalizedRank",
                derived_data=(f_der, y_der, label_der) if f_der is not None else None,
                cutoff=args.cutoff,
                overlay_data=overlay_data if overlay_data else None,
                filter_str=args.filter
            )

    else:
        if args.aweight:
            weight_func, weight_label = a_weight, "A-weighted"
        elif args.bweight:
            weight_func, weight_label = b_weight, "B-weighted"
        elif args.cweight:
            weight_func, weight_label = c_weight, "C-weighted"
        else:
            weight_func, weight_label = lambda f: np.ones_like(f), "Z-weighted"

        ranking_nums_list = [int(x) for x in args.ranking.split(',')] if args.ranking else []

        ranking = rank_headphones(
            args.measurements_dir, target_freq, target_resp, weight_func,
            weight_label, args.sort, verbose=True, top=args.top, regex_filter=args.filter, cutoff=args.cutoff
        )

        # Build the exact set of headphones being plotted (top N ∪ ranking selections,
        # deduplicated), mirroring the logic in plot_headphones.
        plotted = list(ranking[:args.top]) if args.top else []
        seen_fnames = {r[0] for r in plotted}
        for n in ranking_nums_list:
            if 0 < n <= len(ranking) and ranking[n - 1][0] not in seen_fnames:
                plotted.append(ranking[n - 1])
                seen_fnames.add(ranking[n - 1][0])
        if not plotted:
            plotted = list(ranking)

        # When --top is active, print any --ranking entries that fell outside the top-N table.
        if args.top and ranking_nums_list:
            extras = sorted(n for n in ranking_nums_list if n > args.top)
            if extras:
                print(f"\nRanked selections outside top-{args.top}:")
                for n in extras:
                    if 0 < n <= len(ranking):
                        fn, rmse, pref, comb = ranking[n - 1]
                        clean_name = os.path.splitext(fn)[0]
                        print(f"{n:3d}. {clean_name:<65} RMSE={rmse:6.3f}  Pref≈{pref:7.2f}  Combined≈{comb:.3f}")

        f_der, y_der, label_der = compute_consensus_response(plotted, len(plotted), args.measurements_dir, target_freq)

        if f_der is not None:
            print("Writing consensus response CSV with headers: ['frequency', 'raw']")
            pd.DataFrame({'frequency': f_der, 'raw': y_der}).to_csv("Consensus_Response.csv", index=False, header=True)

        if not args.no_plot:
            # When neither --top nor --ranking is given, plot everything.
            plot_top = args.top if args.top else (None if ranking_nums_list else len(ranking))
            plot_headphones(
                args.measurements_dir, target_freq, target_resp, ranking,
                top=plot_top,
                ranking_nums=ranking_nums_list if ranking_nums_list else None,
                weight_label=weight_label, sort_metric=args.sort,
                derived_data=(f_der, y_der, label_der) if f_der is not None else None,
                cutoff=args.cutoff,
                overlay_data=overlay_data if overlay_data else None,
                filter_str=args.filter
            )

if __name__ == "__main__":
    main()