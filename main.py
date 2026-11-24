import argparse
import matplotlib.pyplot as plt
from src.fetcher import LightCurveFetcher
from src.cleaner import LightCurveCleaner
from src.detector import TransitDetector
from src.vetter import TransitVetter
from src.characterizer import PlanetCharacterizer


def run_pipeline(target: str, sectors: list[int] | None = None, max_period: float = 15.0, refresh: bool = False):
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

    # 6. Plot & Save
    folded = stitched_lc.fold(period=tls_res["tls_period"], epoch_time=tls_res["tls_t0"])
    binned = folded.bin(time_bin_size=0.02)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    primary["bls_object"].plot(ax=axes[0], color="navy")
    axes[0].set_title(f"{target} — Stage 1 Broad BLS Periodogram")

    folded.plot(ax=axes[1], color="gray", alpha=0.2, label="Phase-Folded Flux")
    binned.plot(ax=axes[1], color="black", lw=1.5, label="Binned Data")
    model_days = (tls_res["model_folded_phase"] - 0.5) * tls_res["tls_period"]
    axes[1].plot(model_days, tls_res["model_folded_model"], color="crimson", lw=2.5, label="TLS Limb-Darkened Model")
    axes[1].set_xlim(-0.3, 0.3)
    axes[1].legend(loc="lower right")
    axes[1].set_title(f"Stage 2 Physical Fit: Rp = {planet['radius_earth']:.2f} R_earth | Teq = {planet['equilibrium_temp_k']:.0f} K")

    axes[2].errorbar(ttv["epochs"], ttv["oc_minutes"], yerr=ttv["oc_errors_min"], fmt="o", color="teal", ecolor="lightgray", capsize=3)
    axes[2].axhline(0.0, color="black", linestyle="--", lw=1)
    axes[2].set_xlabel("Transit Epoch (Orbit Number)")
    axes[2].set_ylabel("O - C Residual [minutes]")
    axes[2].set_title("Stage 3 Transit Timing Variations (TTV O - C Diagram)")

    safe_name = target.lower().replace(" ", "_")
    out_path = f"outputs/{safe_name}_characterization.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[Pipeline] Saved 3-panel figure to {out_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Sector TESS Exoplanet Transit & Characterization Pipeline")
    parser.add_argument("--target", type=str, default="Pi Mensae", help="Target star name or TIC ID (e.g. 'TOI-700' or 'WASP-18')")
    parser.add_argument("--sectors", type=int, nargs="*", default=None, help="Specific TESS sectors (optional)")
    parser.add_argument("--max-period", type=float, default=15.0, help="Maximum orbital period to search in days")
    parser.add_argument("--refresh", action="store_true", help="Force re-download from MAST instead of using local cache")
    args = parser.parse_args()

    run_pipeline(target=args.target, sectors=args.sectors, max_period=args.max_period, refresh=args.refresh)
