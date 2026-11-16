import lightkurve as lk

class LightCurveFetcher:
    """Queries and downloads multi-sector TESS light curves from NASA MAST."""

    def __init__(self, target: str, mission: str = "TESS", author: str = "SPOC"):
        self.target = target
        self.mission = mission
        self.author = author

    def fetch_sector(self, sectors: list[int], cadence: str ="short") -> lk.LightCurveCollection:
        print(f"[Fetcher] Querying MAST for {self.target} (Sectors: {sectors})....")
        search = lk.search_lightcurve(
            self.target,
            mission=self.mission,
            author=self.author,
            cadence=cadence,
            sector=sectors
        )
        if len(search) == 0:
            raise ValueError(f"No light curves found for {self.target} in sectors {sectors}.")

        print(f"[Fetcher] Downloading {len(search)} light curve files ....")
        return search.download_all()
    
        