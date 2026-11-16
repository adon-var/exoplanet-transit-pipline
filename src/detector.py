import numpy as np
import lightkurve as lk


class TransitDetector:
    """Performs blind Box Least Squares (BLS) searches with iterative multi-planet masking."""

    def __init__(self, min_period: float = 1.0, max_period: float = 15.0, n_periods: int = 15000):
        self.period_grid = np.linspace(min_period, max_period, n_periods)

    def search_multi_planet(self, lc: lk.LightCurve, max_planets: int = 2) -> list[dict]:
        working_lc = lc.copy()
        candidates = []

        for i in range(max_planets):
            print(f"[Detector] Running blind BLS pass #{i}....")
            bls = working_lc.to_periodogram(
                method="bls",
                period=self.period_grid,
                frequency_factor=500
            )

            period = bls.period_at_max_power
            t0 = bls.transit_time_at_max_power
            duration = bls.duration_at_max_power
            max_power = float(np.max(bls.power))

            candidates.append({
                "candidate_id": i,
                "period": period,
                "t0": t0,
                "duration": duration,
                "max_power": max_power,
                "bls_object": bls,
                "lc_snapshot": working_lc.copy()
            })

            print(f" -> Candidate {i}: P = {period.value:.5f} d | t0 = {t0.value:.5f} | Power = {max_power:.1f}")

            # AI Extract clean floats and unmasked timestamps to bypass Astropy MaskedNDArray bug
            p_val = float(period.value)
            t0_val = float(t0.value)
            dur_val = float(duration.value) * 1.5

            time_vals = np.ma.getdata(working_lc.time.value)
            half_p = 0.5 * p_val
            transit_mask = np.abs((time_vals - t0_val + half_p) % p_val - half_p) < 0.5 * dur_val

            working_lc = working_lc[~transit_mask]

        return candidates
