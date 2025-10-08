# ResponseRank

Rank headphone frequency response measurements by how closely they match a target curve (e.g., Harman over-ear 2018).
Closeness is calculated using **root-mean-square error (RMSE)** between the interpolated measurement and the target response.

The script also computes a **preference score** based on the AES 2018 model for headphone preference:

> Olive, S., Welti, T., & McMullin, E. (2018).
> *A Statistical Model that Predicts Listeners’ Preference Ratings of Around-Ear and On-Ear Headphones*.
> AES Convention: 144, Paper Number: 9919, Publication Date: 2018-05-06.
> [AES e-Library link](https://aes2.org/publications/elibrary-page/?id=19436)

This model estimates listener preference from headphone frequency response using smoothness and slope metrics.

---

## Features

* Compares all `.csv` measurement files in a folder against a target curve.

* Interpolates measurements to the target frequencies for fair comparison.

* Calculates **RMSE** and **Preference Score** for each headphone.

* Finds **Spearman-optimized combination weights** for RMSE vs. preference.

* Ranks results from closest to furthest.

* Prints a clean, sorted list of rankings in the terminal, including **RMSE, Pref, and Combined scores**.

* Sorting options:

  * **combined** – Spearman-optimized combined score (default)
  * **rmse** – RMSE only
  * **pref** – Preference score only

* Optional RMSE weighting:

  * **A-weighting** (perceptually relevant at moderate levels)
  * **B-weighting** (medium listening levels)
  * **C-weighting** (for loud/high-level listening)
  * **Flat** (no weighting)

* **Tonally balanced top headphones**: With `--all-weightings`, the script identifies headphones that consistently appear in the top ranks across **Flat, A, B, and C weightings**. Only the tonally balanced top list is plotted to reduce clutter.

* Interactive Plotly plots:

  * Toggle traces on/off
  * Zoom, pan, and hover for detailed frequency response
  * Combine `--top` and `--ranking` selections in one plot

---

## Requirements

* Python 3.7+
* [pandas](https://pandas.pydata.org/)
* [numpy](https://numpy.org/)
* [scikit-learn](https://scikit-learn.org/) (for regression slope analysis)
* [plotly](https://plotly.com/python/) (for interactive plotting)
* [scipy](https://www.scipy.org/) (for Spearman correlation)

Install dependencies:

```bash
pip install pandas numpy plotly scikit-learn scipy
```

---

## Usage

```bash
python response_rank.py <measurements_dir> <target_csv> [options]
```

### Options

| Option                      | Description                                                                                            |
| --------------------------- | ------------------------------------------------------------------------------------------------------ |
| `--aweight`                 | Use A-weighting for RMSE                                                                               |
| `--bweight`                 | Use B-weighting for RMSE (medium levels)                                                               |
| `--cweight`                 | Use C-weighting for RMSE                                                                               |
| `--all-weightings`          | Run all weightings (Flat, A, B, C) and show **tonally balanced top headphones**                        |
| `--top N`                   | Plot the top N ranked headphones                                                                       |
| `--ranking R1,R2,...`       | Plot specific ranked items by their rank number                                                        |
| `--sort combined rmse pref` | Sort the printed rankings by **combined score**, **RMSE**, or **preference score** (default: combined) |
| `--top` and `--ranking`     | Can be used **together**                                                                               |

---

### Examples

#### Single Weighting

```bash
python response_rank.py \
  ~/git/AutoEq/measurements/oratory1990/data/over-ear \
  "~/git/AutoEq/targets/oratory1990 optimum hifi over-ear.csv" \
  --aweight --top 10
```

This plots the top 10 headphones ranked using **A-weighted RMSE** combined with preference scores.

---

#### All Weightings / Tonally Balanced Top

```bash
python response_rank.py \
  ~/git/AutoEq/measurements/oratory1990/data/over-ear \
  "~/git/AutoEq/targets/oratory1990 optimum hifi over-ear.csv" \
  --all-weightings --top 10
```

This runs each weighting individually, finds the headphones that rank in the top-N across all weightings, and plots only these tonally balanced top headphones to reduce visual clutter.

* **Derived target curve from top headphones**: When `--all-weightings` is used, the script computes an **empirical neutral curve** by averaging the normalized responses of the tonally balanced top headphones. This curve:

  * Summarizes the common tonal characteristics of the best-performing headphones.
  * Becomes the **reference target for identifying real-world neutral headphones** in this analysis.
  * Is smoothed for clarity and plotted alongside the original target and top headphone traces.
  * Is exported as `Derived_Target.csv` for reference or optional use.
  * **Important**: While it serves as the effective target within this script, it is **not necessarily intended as a general EQ target** for listening — it is primarily a tool for analysis and ranking.

  <p align="center">
   <img src="https://i.imgur.com/RfmCffi.png" alt="Derived target curve plot" width="1244">
  </p>

---

### Sample Output

**Individual Ranking (single weighting)**

```bash
Optimal weights → RMSE=0.70, Pref=0.30, Spearman ρ≈0.8086

Ranked headphones (sorted by combined):
  1. Sennheiser HD 600.csv                              RMSE=1.060  Pref≈87.57  Combined≈0.987
  2. Sennheiser HE 90 Orpheus.csv                       RMSE=1.237  Pref≈85.99  Combined≈0.977
  3. Sennheiser HD 6XX.csv                              RMSE=1.387  Pref≈91.69  Combined≈0.976
  4. Sennheiser HD 650.csv                              RMSE=1.387  Pref≈91.69  Combined≈0.976
  5. HIFIMAN Sundara (post-2020 earpads).csv            RMSE=1.534  Pref≈96.82  Combined≈0.974
...
```

**Tonally Balanced Top Across All Weightings**

```bash
Top 3 tonally balanced headphones by average normalized rank:

                                       Headphone  AvgNormalizedRank
0                         Sennheiser HD 560S.csv              0.800
1        HIFIMAN Sundara (post-2020 earpads).csv              0.800
2  Sash Tres 45 (open back, leather earpads).csv              0.575
```

---

<p align="center">
  <img src="https://i.imgur.com/AHF4uIu.png" alt="ResponseRank interactive plot" width="1403">
</p>

---

## Notes

* Measurement and target files must be `.csv` with:

  * **Column 1:** Frequency (Hz)
  * **Column 2:** Response (dB)

* Non-CSV files are ignored automatically.

* Frequencies in measurement and target files **do not need to match** — the script interpolates them.

* Interactive plotting is optional and only generated when `--top`, `--ranking`, or `--all-weightings` is used.

---

## Why This Script?

### RMSE

Root-mean-square error summarizes the overall difference between a measurement and the target curve in a **single number**, penalizing large deviations more heavily than smaller ones. It’s ideal for ranking headphones by tonal accuracy.

### Preference Score

Based on AES 2018 research, the preference score uses frequency response smoothness and slope deviation to estimate how much listeners are likely to prefer the sound of a headphone.

### Spearman-Optimized Combination

The script finds an **optimal combination of RMSE and preference scores** that maximizes correlation with both metrics across all headphones, producing a robust “best-of-both-worlds” ranking.

### Interpolation

Measurements often have different frequency points than the target. Interpolation ensures a **fair, consistent comparison** across all files.

### Use Cases

* Quickly rank multiple headphone measurements.
* Compare against any reference target (e.g., Harman 2018, oratory1990 optimum hifi).
* Useful for **reviewers, audio engineers, and enthusiasts** who want automated ranking and visualization.
