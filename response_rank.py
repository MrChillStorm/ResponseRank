#!/usr/bin/env python3
# ResponseRank — RMSE + Pref with Spearman correlation optimization and Plotly plotting
# Added: --all-weightings option with automatic plotting of tonally balanced top headphones

import os
import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter1d

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
    # Approximate B between A and C (informational)
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

def gaussian_smooth(freq, resp, sigma=3.0):
    """
    Apply Gaussian smoothing to a response curve.
    freq: frequency array (Hz)
    resp: response array (dB)
    sigma: smoothing sigma (in index units)
    """
    resp = np.asarray(resp)
    smoothed = gaussian_filter1d(resp, sigma=sigma)
    return smoothed

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
    # protect against constant arrays
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
    to_plot=[]
    title_parts=[]
    if top:
        to_plot.extend(ranking[:top])
        title_parts.append(f"Top {top}")
    if ranking_nums:
        to_plot.extend([ranking[i-1] for i in ranking_nums if 0<i<=len(ranking)])
        title_parts.append(f"Ranks {','.join(map(str, ranking_nums))}")
    # Deduplicate preserving order
    seen=set()
    unique=[]
    for r in to_plot:
        if r[0] not in seen:
            unique.append(r); seen.add(r[0])
    to_plot=unique
    title_suffix=" & ".join(title_parts) if title_parts else "All Headphones"
    if to_plot:
        fig=go.Figure()
        target_norm=normalize_at_1kHz(target_freq,target_resp)
        fig.add_trace(go.Scatter(x=target_freq,y=target_norm,mode='lines',
                                 name='Target Curve',line=dict(color='black',width=2)))
        for fname,rmse,pref,comb in to_plot:
            meas_path = os.path.join(measurements_dir, fname)
            # If the incoming 'fname' is already a cleaned name (no .csv) skip reading by filename
            if os.path.exists(meas_path):
                meas = pd.read_csv(meas_path)
                meas_freq,meas_resp = meas.iloc[:,0].values,meas.iloc[:,1].values
                meas_norm = normalize_at_1kHz(meas_freq,meas_resp)
            else:
                # If the caller passed a tuple where meas data isn't available, we skip reading and plot nothing
                # (defensive fallback)
                continue

            clean_name = os.path.splitext(fname)[0]
            if sort_metric == "AvgNormalizedRank":
                label = f"{clean_name} (AvgNorm={comb:.3f})"
            else:
                label = f"{clean_name} (RMSE={rmse:.2f}, Pref={pref:.2f}, Comb={comb:.2f})"
            fig.add_trace(go.Scatter(
                x=meas_freq,y=meas_norm,mode='lines',
                name=label
            ))
        fig.update_layout(
            title=f"{title_suffix} vs Target ({weight_label})",
            xaxis_type='log',xaxis_title='Frequency (Hz)',yaxis_title='Response (dB)',
            template='plotly_white',hovermode='x unified',
            margin=dict(t=100,b=80),
            annotations=[
                dict(x=0,y=0,xref='paper',yref='paper',
                     text=f'Sort metric: {sort_metric}; Pref uses 1/3-octave smoothed data; curves are raw',
                     showarrow=False,font=dict(size=10),xanchor='left',yanchor='bottom')
            ]
        )
        fig.show()

# ----------------------------------------------------------------------
# Core ranking function
# ----------------------------------------------------------------------
def rank_headphones(measurements_dir, target_freq, target_resp, weight_func, weight_label, sort_metric="combined", verbose=True, top=None):
    """
    Rank headphones in a directory against a target response using RMSE and Pref metrics.

    Returns:
        ranking: List of tuples (filename, RMSE, Pref, CombinedScore) sorted by sort_metric.
    """
    weights = weight_func(target_freq)
    results = []

    for fname in sorted(os.listdir(measurements_dir)):
        if not fname.endswith(".csv"):
            continue
        try:
            meas = pd.read_csv(os.path.join(measurements_dir, fname))
            meas_freq, meas_resp = meas.iloc[:, 0].values, meas.iloc[:, 1].values
            rmse = compute_rmse(meas_freq, meas_resp, target_freq, target_resp, weights)
            pref = compute_pref_metrics(meas_freq, meas_resp, target_freq, target_resp)
            results.append((fname, rmse, pref))
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
    target = pd.read_csv(args.target_path)
    target_freq, target_resp = target.iloc[:, 0].values, target.iloc[:, 1].values

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
                top=args.top
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
            meas = pd.read_csv(meas_path)
            freq, resp = meas.iloc[:, 0].values, meas.iloc[:, 1].values
            interp_resp = np.interp(target_freq, freq, resp)
            norm_resp = normalize_at_1kHz(target_freq, interp_resp)
            derived_responses.append(norm_resp)

        if derived_responses:
            derived_target = np.mean(derived_responses, axis=0)

            # -----------------------------
            # Find optimal sigma for 2.5–4.5 kHz hump
            # -----------------------------
            mask_hump = (2500, 4500)
            # Find the peak amplitude in the original Oratory target in that region
            idx_mask = (target_freq >= mask_hump[0]) & (target_freq <= mask_hump[1])
            target_peak = np.max(target_resp[idx_mask])

            # Scan candidate sigmas
            sigma_candidates = np.linspace(1, 20, 1000)
            best_sigma = sigma_candidates[0]
            min_error = float('inf')
            for sigma in sigma_candidates:
                smoothed = gaussian_filter1d(derived_target, sigma=sigma)
                peak_amp = np.max(smoothed[idx_mask])
                error = abs(peak_amp - target_peak)
                if error < min_error:
                    min_error = error
                    best_sigma = sigma

            #print(f"Optimal sigma for hump amplitude match: {best_sigma:.3f}")

            # Apply Gaussian smoothing with optimal sigma
            derived_target_smooth = gaussian_smooth(target_freq, derived_target, sigma=best_sigma)

            # -----------------------------
            # Plot all together
            # -----------------------------
            fig = go.Figure()
            target_norm = normalize_at_1kHz(target_freq, target_resp)
            fig.add_trace(go.Scatter(
                x=target_freq, y=target_norm, mode='lines',
                name='Original Target', line=dict(color='black', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=target_freq, y=derived_target_smooth, mode='lines',
                name=f'Derived Target (empirical neutral, σ={best_sigma:.2f})',
                line=dict(color='orange', width=3, dash='dot')
            ))

            for fname, rmse, pref, avg in filtered_sorted[:top_to_show]:
                meas = pd.read_csv(os.path.join(args.measurements_dir, fname))
                freq, resp = meas.iloc[:, 0].values, meas.iloc[:, 1].values
                meas_norm = normalize_at_1kHz(freq, resp)
                clean_name = os.path.splitext(fname)[0]
                fig.add_trace(go.Scatter(
                    x=freq, y=meas_norm, mode='lines',
                    name=f"{clean_name} (AvgNorm={avg:.3f})"
                ))

            fig.update_layout(
                title=f"Tonally Balanced Top-{top_to_show} vs Targets",
                xaxis_type='log', xaxis_title='Frequency (Hz)',
                yaxis_title='Response (dB)',
                template='plotly_white', hovermode='x unified',
                margin=dict(t=100, b=80)
            )
            fig.show()

            # -----------------------------
            # Export to CSV
            # -----------------------------
            pd.DataFrame({'frequency': target_freq, 'raw': derived_target_smooth}).to_csv(
                "Derived_Target.csv", index=False
            )

    else:
        # Single weighting
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
            top=args.top
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