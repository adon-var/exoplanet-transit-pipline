import matplotlib.pyplot as plt
from src.fetcher import LightCurveFetcher
from src.cleaner import LightCurveCleaner
from src.detector import TransitDetector

def run_pipeline():
    target = "Pi Mensae"
    sector = [1,4,8]

    #Ingest Data
    fetcher = LightCurveFetcher(target=target)
    lc_collection = fetcher.fetch_sector(sectors=sector)

    #Clean and stitch
    cleaner = LightCurveCleaner(sigma=5.0, window_length=1001)
    stitched_lc = cleaner.stitch_and_clean(lc_collection)

    #Blind Multi-Planet Detection (2 iterative passes)
    detector = TransitDetector(min_period=1.0, max_period=15.0)
    candidates = detector.search_multi_planet(stitched_lc, max_planets=2)

    #Plot the pass no.1 (Primary planet detected)
    primary = candidates[0]
    folded = primary["lc_snapshot"].fold(period=primary["period"], epoch_time=primary["t0"])
    binned = folded.bin(time_bin_size=0.02)

    fig,axes = plt.subplots(2, 1, figsize=(10, 8))
    primary["bls_object"].plot(ax=axes[0], color="navy")
    axes[0].set_title(f"{target} - Blind BLS Peridogram (Pass #1)")

    folded.plot(ax=axes[1], color="gray", alpha=0.2, label="Phase-folded Flux")
    binned.plot(ax=axes[1], color="crimson", lw=2.5, label="Binned Transit Profile")
    axes[1].set_xlim(-0.3, 0.3)
    axes[1].set_title(f"Recovered Transit : P = {primary['period'].value:.2f} days")

    plt.tight_layout()
    plt.savefig("outputs/pi_mensae_detection.png", dpi=300)
    print("[Pipeline] Saved high-res figure to outputs/pi_mensae_detection.png")
    plt.show()

if __name__ == "__main__":
    run_pipeline()
