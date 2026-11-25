# Exoplanet Transit Detection Pipeline

A Python script for downloading TESS light curves, cleaning them up, and searching for transiting exoplanets using Box Least Squares (BLS) periodograms.

## Layout

- `src/fetcher.py` — Downloads SPOC short-cadence light curves from MAST using `lightkurve`.
- `src/cleaner.py` — Clips cosmic-ray outliers ($\sigma = 5$), flattens stellar variability with a Savitzky-Golay filter, and stitches multiple sectors together.
- `src/detector.py` — Runs a BLS search to find orbital period ($P$), transit time ($t_0$), and duration, then masks out the transits to look for a second planet.
- `main.py` — Runs the full workflow and saves the output plots to `outputs/`.

## Test Run: Pi Mensae

Tested on TESS Sectors 1, 4, and 8. The first BLS pass recovered **Pi Mensae c**:

- **Period ($P$):** `6.26808 days`
- **Transit time ($t_0$):** `1325.49696 BTJD`

![Pi Mensae Detection](outputs/pi_mensae_detection.png)

## Setup & Usage

```bash
conda env create -f environment.yml
conda activate exoplanet
python main.py
```