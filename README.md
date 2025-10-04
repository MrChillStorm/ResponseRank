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
* Ranks results from closest to furthest.
* Prints a clean, sorted list of rankings in the terminal.
* Optional weighting:

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

Install dependencies:

```bash
pip install pandas numpy plotly scikit-learn
````

---

## Usage

```bash
python response_rank.py <measurements_dir> <target_csv> [options]
```

### Options

| Option                                           | Description                                                   |
| ------------------------------------------------ | ------------------------------------------------------------- |
| `--aweight`                                      | Use A-weighting for RMSE (default if unspecified is **flat**) |
| `--cweight`                                      | Use C-weighting for RMSE                                      |
| `--top N`                                        | Plot the top N ranked headphones                              |
| `--ranking R1,R2,...`                            | Plot specific ranked items by their rank number               |
| `--top` and `--ranking` can be used **together** |                                                               |

---

### Example

```bash
python response_rank.py \
  ~/git/AutoEq/measurements/oratory1990/data/over-ear \
  "~/git/AutoEq/targets/oratory1990 optimum hifi over-ear.csv" \
  --aweight --ranking 1,5,13,14
```

---

### Sample Output

```bash
Ranked headphones (closest first):
  1. Sennheiser HD 600.csv                              RMSE=1.0602  Pref≈87.57
  2. Beyerdynamic DT 880 (worn earpads).csv             RMSE=1.1515  Pref≈62.77
  3. Dan Clark Audio EXPANSE.csv                        RMSE=1.2031  Pref≈32.70
  4. Philips Fidelio X2HR.csv                           RMSE=1.2364  Pref≈79.03
  5. Sennheiser HE 90 Orpheus.csv                       RMSE=1.2368  Pref≈85.99
  6. FiiO FT3 (pleather earpads).csv                    RMSE=1.2597  Pref≈66.24
  7. Audio-Technica ATH-R70x.csv                        RMSE=1.2842  Pref≈81.32
  8. FiiO FT3 (suede earpads).csv                       RMSE=1.3243  Pref≈46.06
  9. Onkyo A800.csv                                     RMSE=1.3265  Pref≈75.24
 10. Sennheiser HE 1 Orpheus 2.csv                      RMSE=1.3307  Pref≈75.99
 11. Shure SRH440.csv                                   RMSE=1.3587  Pref≈81.41
 12. Dan Clark Audio Stealth.csv                        RMSE=1.3663  Pref≈33.25
 13. Sennheiser HD 6XX.csv                              RMSE=1.3870  Pref≈91.69
 14. Sennheiser HD 650.csv                              RMSE=1.3870  Pref≈91.69
...
```

<p align="center">
  <img src="https://i.imgur.com/VhffhYQ.png" alt="ResponseRank interactive plot" width="1219">
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

### Interpolation

Measurements often have different frequency points than the target. Interpolation ensures a **fair, consistent comparison** across all files.

### Use Cases

* Quickly rank multiple headphone measurements.
* Compare against any reference target (e.g., Harman 2018, oratory1990 optimum hifi).
* Useful for **reviewers, audio engineers, and enthusiasts** who want automated ranking and visualization.
