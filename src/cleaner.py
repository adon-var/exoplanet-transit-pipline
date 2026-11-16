import lightkurve as lk

class LightCurveCleaner:
    """Cleans instrumental noise, flattens stellar variability, and stitches sectors."""

    def __init__(self, sigma: float = 5.0, window_length: int = 1001):
        self.sigma = sigma
        self.window_length = window_length

    def _clean_single(self, lc: lk.LightCurve) -> lk.LightCurve:
        return (
            lc.remove_nans()
            .remove_outliers(sigma=self.sigma)
            .flatten(window_length=self.window_length)
        )

    def stitch_and_clean(self, lc_collection: lk.LightCurveCollection) -> lk.LightCurve:
        print("[Cleaner] Detrending and stitching sectors....")
        return lc_collection.stitch(corrector_func=self._clean_single)