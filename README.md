# Multi-Sector Exoplanet Transit & Characterization Suite

An end-to-end, object-oriented Python pipeline for automated TESS photometric detrending, multi-sector stitching, two-stage blind transit detection (BLS + limb-darkened TLS), false-positive vetting, and Bayesian MCMC physical exoplanet characterization.

![WASP-18b 2x2 Characterization Dashboard](docs/wasp-18_characterization.png?raw=true)

## Pipeline Architecture
- **`src/fetcher.py` (`LightCurveFetcher`):** Queries NASA MAST archives via `lightkurve` with automatic sector discovery and local FITS disk caching (`data/`).
- **`src/cleaner.py` (`LightCurveCleaner`):** Applies $5\sigma$ cosmic-ray outlier clipping, Savitzky–Golay polynomial detrending, and median flux normalization across stitched sectors.
- **`src/detector.py` (`TransitDetector`):** Executes broad blind Box Least Squares (BLS) periodograms with iterative transit masking for multi-planet searches.
- **`src/vetter.py` (`TransitVetter`):** Performs physical limb-darkened Transit Least Squares (TLS) refinement and Odd/Even transit depth validation ($<3\sigma$ threshold) to eliminate eclipsing binary false positives.
- **`src/characterizer.py` (`PlanetCharacterizer`):** Queries the TESS Input Catalog (TIC) via `astroquery` (with local JSON caching) to derive planetary radius ($R_p$), semi-major axis ($a$), equilibrium temperature ($T_{\text{eq}}$), and individual transit midpoints for Transit Timing Variation ($O - C$) analysis.
- **`src/mcmc_fitter.py` (`BayesianTransitFitter`):** Wraps an analytical Mandel–Agol transit model (`batman`) inside an affine-invariant Markov Chain Monte Carlo sampler (`emcee`) to extract formal $1\sigma$ posterior uncertainties and `corner` covariance plots.

## Bayesian Posterior Analysis (WASP-18b)
![WASP-18b Bayesian MCMC Corner Plot](docs/wasp-18_mcmc_corner.png?raw=true)

## Benchmark Validations
- **Pi Mensae c (`TIC 261136679`):** Recovered blindly across Sectors 1, 4, and 8 at $P = 6.267605\text{ days}$ ($\text{SNR} = 40.9$), yielding $R_p = 2.06\,R_\oplus$ (literature: $2.04 \pm 0.05\,R_\oplus$) and $T_{\text{eq}} = 1081\text{ K}$.
- **WASP-18b (`TIC 100100827`):** Recovered across 4 auto-discovered sectors at $P = 0.941443\text{ days}$ ($\text{SNR} = 1156.8$), yielding $R_p/R_\star = 0.10161^{+0.00010}_{-0.00021}$ via MCMC and $T_{\text{eq}} = 2255\text{ K}$.

## Installation & CLI Usage
```bash
conda env create -f environment.yml
conda activate exoplanet
python -m pytest -v

# Run standard detection & 2x2 characterization dashboard on any TESS target:
python main.py --target "Pi Mensae" --sectors 1 4 8

# Run with full Bayesian MCMC posterior sampling & corner plot generation:
python main.py --target "WASP-18" --mcmc

# Force fresh MAST download instead of local FITS cache:
python main.py --target "TOI-700" --max-period 40 --refresh
```
