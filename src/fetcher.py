import lightkurve as lk


class LightCurveFetcher:
    """Queries and downloads multi-sector TESS light curves from NASA MAST."""

    def __init__(self, target: str, mission: str = "TESS", author: str = "SPOC"):
        self.target = target
        self.mission = mission
        self.author = author

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

        # Cap auto-discovered sectors so high-latitude stars don't overload local RAM
        if sectors is None and len(search) > max_sectors:
            print(f"[Fetcher] Found {len(search)} sectors; selecting first {max_sectors} to preserve RAM.")
            search = search[:max_sectors]

        print(f"[Fetcher] Downloading {len(search)} light curve files...")
        return search.download_all()
