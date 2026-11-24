from pathlib import Path
import lightkurve as lk


class LightCurveFetcher:
    """Queries NASA MAST archives and caches stitched light curves locally as FITS files."""

    def __init__(self, target: str, mission: str = "TESS", author: str = "SPOC", cache_dir: str = "data"):
        self.target = target
        self.mission = mission
        self.author = author
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, sectors: list[int] | None) -> Path:
        safe_name = self.target.lower().replace(" ", "_")
        sec_tag = "_".join(map(str, sectors)) if sectors else "auto"
        return self.cache_dir / f"{safe_name}_sec_{sec_tag}_raw.fits"

    def load_cached_stitched(self, sectors: list[int] | None = None) -> lk.LightCurve | None:
        path = self._cache_path(sectors)
        if path.exists():
            print(f"[Fetcher] Loading cached light curve from {path}...")
            return lk.read(str(path))
        return None

    def save_cached_stitched(self, lc: lk.LightCurve, sectors: list[int] | None = None) -> None:
        path = self._cache_path(sectors)
        lc.to_fits(str(path), overwrite=True)
        print(f"[Fetcher] Saved stitched light curve cache to {path}")

    def fetch_sector(
        self, sectors: list[int] | None = None, cadence: str = "short", max_sectors: int = 4
    ) -> lk.LightCurveCollection:
        label = sectors if sectors is not None else f"auto (up to {max_sectors})"
        print(f"[Fetcher] Querying MAST for {self.target} (Sectors: {label})...")

        search = lk.search_lightcurve(
            self.target,
            mission=self.mission,
            author=self.author,
            cadence=cadence,
            sector=sectors,
        )
        if len(search) == 0:
            raise ValueError(f"No light curves found for {self.target}.")

        if sectors is None and len(search) > max_sectors:
            print(f"[Fetcher] Found {len(search)} sectors; selecting first {max_sectors} to preserve RAM.")
            search = search[:max_sectors]

        print(f"[Fetcher] Downloading {len(search)} light curve files...")
        return search.download_all()
