import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightkurve")
warnings.filterwarnings("ignore", category=UserWarning, module="transitleastsquares")

from src.fetcher import LightCurveFetcher
from src.cleaner import LightCurveCleaner
from src.detector import TransitDetector
from src.vetter import TransitVetter
from src.characterizer import PlanetCharacterizer
from src.mcmc_fitter import BayesianTransitFitter


def run_pipeline(
        target: str,
        sectors: list[int] | None = None,
        max_period: float = 15.0,
        refresh: bool = False,
        run_mcmc: bool = False,
    ):
     # 1. Ingest Data (Check local cache first)
    fetcher = LightCurveFetcher(target=target)
    stitched_lc = None if refresh else fetcher.load_cached_stitched(sectors=sectors)

    if stitched_lc is None:
        lc_collection = fetcher.fetch_sector(sectors=sectors)
        # 2. Clean & Stitch
        cleaner = LightCurveCleaner(sigma=5.0, window_length=1001)
        stitched_lc = cleaner.stitch_and_clean(lc_collection)
        fetcher.save_cached_stitched(stitched_lc, sectors=sectors)

    # 3. Stage 1: Broad Blind BLS Search
    detector = TransitDetector(min_period=0.5, max_period=max_period)
    candidates = detector.search_multi_planet(stitched_lc, max_planets=1)
    primary = candidates [0]  # Focus on the first detected candidate
    bls_period = float(primary["period"].value)

    # 4. Stage 2: Physical TLS Refinement & False-Positive Vetting
    vetter = TransitVetter(sigma_threshold=3.0)
    tls_res = vetter.refine_with_tls(stitched_lc, bls_period=bls_period)
    odd_even = vetter.check_odd_even(
        stitched_lc,
        period=tls_res["tls_period"],
        t0=tls_res["tls_t0"],
        duration=tls_res["tls_duration"],
    )

    # 5. Stage 3: Physical Characterization & TTVs
    characterizer = PlanetCharacterizer()
    star = characterizer.fetch_stellar_params(target)
    planet = characterizer.compute_planetary_params(
        period_days=tls_res["tls_period"],
        depth=tls_res["tls_depth"],
        stellar_params=star,
    )
    ttv = characterizer.measure_ttvs(
        stitched_lc,
        period=tls_res["tls_period"],
        t0=tls_res["tls_t0"],
        duration=tls_res["tls_duration"],
        depth=tls_res["tls_depth"],
    )

    print(f"\n================ {target.upper()} REPORT ================")
    print(f"Host Star (TIC {star['tic_id']}) : R* = {star['r_star_solar']:.2f} R_sun | M* = {star['m_star_solar']:.2f} M_sun | Teff = {star['teff_k']:.0f} K")
    print(f"Refined Period        : {tls_res['tls_period']:.6f} days (SNR = {tls_res['tls_snr']:.1f})")
    print(f"Odd/Even Vetting      : Diff = {odd_even['sigma_diff']:.2f} sigma -> PASSED: {odd_even['passed']}")
    print(f"Planetary Radius      : {planet['radius_earth']:.2f} Earth radii")
    print(f"Semi-Major Axis       : {planet['semi_major_axis_au']:.4f} AU")
    print(f"Equilibrium Temp      : {planet['equilibrium_temp_k']:.0f} K")
    print(f"Individual Transits   : Measured {len(ttv['epochs'])} epochs for TTV O-C analysis")
    print("=========================================================\n")

    if run_mcmc:
        fitter = BayesianTransitFitter(n_walkers=24, n_steps=800, burn_in=250)
        mcmc_res = fitter.fit(
            stitched_lc,
            period=tls_res["tls_period"],
            t0_init=tls_res["tls_t0"],
            duration=tls_res["tls_duration"],
            depth_init=tls_res["tls_depth"],
            target_name=target,
        )
        rp_med, rp_up, rp_lo = mcmc_res["rp_rs"]
        r_earth_med = rp_med * star["r_star_solar"] * (characterizer.R_SUN / characterizer.R_EARTH)
        r_earth_err = max(rp_up, rp_lo) * star["r_star_solar"] * (characterizer.R_SUN / characterizer.R_EARTH)
        inc_med, inc_up, inc_lo = mcmc_res["inc"]

        print("\n================ BAYESIAN MCMC POSTERIORS ================")
        print(f"Rp / R* Ratio   : {rp_med:.5f} (+{rp_up:.5f} / -{rp_lo:.5f})")
        print(f"MCMC Radius     : {r_earth_med:.2f} +/- {r_earth_err:.2f} Earth radii")
        print(f"Inclination (i) : {inc_med:.2f} (+{inc_up:.2f} / -{inc_lo:.2f}) deg")
        print("==========================================================\n")


    # 6. Plot & Save 2x2 Widescreen End-to-End Summary
    folded = stitched_lc.fold(period=tls_res["tls_period"], epoch_time=tls_res["tls_t0"])
    binned = folded.bin(time_bin_size=0.02)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Top-Left: Full Stitched Time-Series Light Curve
    stitched_lc.scatter(ax=axes[0, 0], color="black", s=1, alpha=0.4, label="Stitched Flux")
    axes[0, 0].set_title(f"{target} — Multi-Sector TESS Light Curve")

    # Top-Right: Stage 1 Broad BLS Periodogram
    primary["bls_object"].plot(ax=axes[0, 1], color="navy")
    axes[0, 1].set_title("Stage 1: Blind BLS Periodogram")

    # Bottom-Left: Stage 2 Phase-Folded Physical Transit Fit
    folded.plot(ax=axes[1, 0], color="gray", alpha=0.2, label="Folded Flux")
    binned.plot(ax=axes[1, 0], color="black", lw=1.5, label="Binned Data")
    model_days = (tls_res["model_folded_phase"] - 0.5) * tls_res["tls_period"]
    axes[1, 0].plot(model_days, tls_res["model_folded_model"], color="crimson", lw=2.5, label="TLS Physical Model")
    axes[1, 0].set_xlim(-0.3, 0.3)
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].set_title(f"Stage 2: Transit Fit (Rp = {planet['radius_earth']:.2f} R_earth | Teq = {planet['equilibrium_temp_k']:.0f} K)")

    # Bottom-Right: Stage 3 Transit Timing Variations (O - C)
    axes[1, 1].errorbar(ttv["epochs"], ttv["oc_minutes"], yerr=ttv["oc_errors_min"], fmt="o", color="teal", ecolor="lightgray", capsize=3)
    axes[1, 1].axhline(0.0, color="black", linestyle="--", lw=1)
    axes[1, 1].set_xlabel("Transit Epoch (Orbit Number)")
    axes[1, 1].set_ylabel("O - C Residual [minutes]")
    axes[1, 1].set_title("Stage 3: Transit Timing Variations (O - C)")

    safe_name = target.lower().replace(" ", "_")
    out_path = f"outputs/{safe_name}_characterization.png"
    plt.tight_layout(pad=2.0)
    plt.savefig(out_path, dpi=300)
    print(f"[Pipeline] Saved 2x2 summary figure to {out_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Sector TESS Exoplanet Transit & Characterization Pipeline")
    parser.add_argument("--target", type=str, default="Pi Mensae", help="Target star name or TIC ID (e.g. 'TOI-700' or 'WASP-18')")
    parser.add_argument("--sectors", type=int, nargs="*", default=None, help="Specific TESS sectors (optional)")
    parser.add_argument("--max-period", type=float, default=15.0, help="Maximum orbital period to search in days")
    parser.add_argument("--refresh", action="store_true", help="Force re-download from MAST instead of using local cache")
    parser.add_argument("--mcmc", action="store_true", help="Run Bayesian MCMC physical transit fit and generate corner plot")
    args = parser.parse_args()

    run_pipeline(target=args.target, sectors=args.sectors, max_period=args.max_period, refresh=args.refresh, run_mcmc=args.mcmc)
