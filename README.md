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

* Prints a clean, sorted list of rankings in the terminal.

* Sorting options:

  * **combined** – Spearman-optimized combined score (default)
  * **rmse** – RMSE only
  * **pref** – Preference score only

* Optional RMSE weighting:

  * **A-weighting** (perceptually relevant at moderate levels)
  * **C-weighting** (for loud/high-level listening)
  * **Flat** (no weighting)

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

| Option                                           | Description                                                                                            |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `--aweight`                                      | Use A-weighting for RMSE (default if unspecified is **flat**)                                          |
| `--cweight`                                      | Use C-weighting for RMSE                                                                               |
| `--top N`                                        | Plot the top N ranked headphones                                                                       |
| `--ranking R1,R2,...`                            | Plot specific ranked items by their rank number                                                        |
| `--sort combined rmse pref`                      | Sort the printed rankings by **combined score**, **RMSE**, or **preference score** (default: combined) |
| `--top` and `--ranking`                          | can be used **together**                                                                               |

---

### Example

```bash
python response_rank.py \
  ~/git/AutoEq/measurements/oratory1990/data/over-ear \
  "~/git/AutoEq/targets/oratory1990 optimum hifi over-ear.csv" \
  --aweight --top 10
```

---

### Sample Output

```bash
Optimal weights → RMSE=0.70, Pref=0.30, Spearman ρ≈0.8086

Ranked headphones (sorted by combined):
  1. Sennheiser HD 600.csv                              RMSE=1.060  Pref≈87.57  Combined≈0.987
  2. Sennheiser HE 90 Orpheus.csv                       RMSE=1.237  Pref≈85.99  Combined≈0.977
  3. Sennheiser HD 6XX.csv                              RMSE=1.387  Pref≈91.69  Combined≈0.976
  4. Sennheiser HD 650.csv                              RMSE=1.387  Pref≈91.69  Combined≈0.976
  5. HIFIMAN Sundara (post-2020 earpads).csv            RMSE=1.534  Pref≈96.82  Combined≈0.974
  6. Sennheiser HD 560S.csv                             RMSE=1.549  Pref≈97.45  Combined≈0.974
  7. Audio-Technica ATH-R70x.csv                        RMSE=1.284  Pref≈81.32  Combined≈0.970
  8. Sash Tres 45 (open back, leather earpads).csv      RMSE=1.568  Pref≈93.98  Combined≈0.970
  9. Philips Fidelio X2HR.csv                           RMSE=1.236  Pref≈79.03  Combined≈0.969
 10. Moondrop Para.csv                                  RMSE=1.492  Pref≈88.87  Combined≈0.968
...
```

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
* Interactive plotting is optional and only generated when `--top` or `--ranking` is used.

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
