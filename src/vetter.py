import numpy as np
import lightkurve as lk
from transitleastsquares import transitleastsquares


class TransitVetter:
    """Validates BLS candidates using TLS refinement, Odd/Even depth tests, and Phase 0.5 checks."""

    def __init__(self, sigma_threshold: float = 3.0):
        self.sigma_threshold = sigma_threshold

    def refine_with_tls(self, lc: lk.LightCurve, bls_period: float) -> dict:
        """Runs a focused Transit Least Squares search around the initial BLS period."""
        print("[Vetter] Running physical limb-darkened TLS refinement...")
        clean_lc = lc.remove_nans()
        time = np.asarray(clean_lc.time.value, dtype=float)
        flux = np.asarray(clean_lc.flux.value, dtype=float)

        # Zoom within +/- 3% of the BLS period to save CPU time while getting high precision
        min_p = max(0.5, bls_period * 0.97)
        max_p = bls_period * 1.03

        model = transitleastsquares(time, flux)
        results = model.power(
            period_min=min_p,
            period_max=max_p,
            show_progress_bar=False
        )

        return {
            "tls_period": float(results.period),
            "tls_t0": float(results.T0),
            "tls_duration": float(results.duration),
            "tls_depth": float(1.0 - results.depth),
            "tls_sde": float(results.SDE),
            "tls_snr": float(results.snr),
            "folded_phase": results.folded_phase,
            "folded_y": results.folded_y,
            "model_folded_phase": results.model_folded_phase,
            "model_folded_model": results.model_folded_model,
        }

    def check_odd_even(self, lc: lk.LightCurve, period: float, t0: float, duration: float) -> dict:
        """Compares odd-numbered vs even-numbered transit depths to rule out eclipsing binaries."""
        time = np.asarray(lc.time.value, dtype=float)
        flux = np.asarray(lc.flux.value, dtype=float)

        # Calculate integer orbit epoch for every timestamp
        epochs = np.round((time - t0) / period).astype(int)
        phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period

        in_transit = np.abs(phase) < (0.4 * duration)
        odd_mask = in_transit & (epochs % 2 != 0)
        even_mask = in_transit & (epochs % 2 == 0)

        odd_flux = flux[odd_mask]
        even_flux = flux[even_mask]

        odd_depth = 1.0 - np.nanmean(odd_flux)
        even_depth = 1.0 - np.nanmean(even_flux)

        odd_err = np.nanstd(odd_flux) / np.sqrt(len(odd_flux))
        even_err = np.nanstd(even_flux) / np.sqrt(len(even_flux))

        combined_err = np.sqrt(odd_err**2 + even_err**2)
        sigma_diff = abs(odd_depth - even_depth) / combined_err

        return {
            "odd_depth_ppm": odd_depth * 1e6,
            "even_depth_ppm": even_depth * 1e6,
            "sigma_diff": float(sigma_diff),
            "passed": bool(sigma_diff < self.sigma_threshold),
        }

    def check_secondary_eclipse(self, lc: lk.LightCurve, period: float, t0: float, duration: float) -> dict:
        """Checks orbital phase 0.5 for a secondary occultation dip."""
        time = np.asarray(lc.time.value, dtype=float)
        flux = np.asarray(lc.flux.value, dtype=float)

        # Shift t0 by half a period to center on phase 0.5
        t_sec = t0 + 0.5 * period
        phase_sec = ((time - t_sec + 0.5 * period) % period) - 0.5 * period

        in_secondary = np.abs(phase_sec) < (0.4 * duration)
        out_of_eclipse = (np.abs(phase_sec) > duration) & (np.abs(phase_sec) < 0.25 * period)

        sec_flux = flux[in_secondary]
        base_flux = flux[out_of_eclipse]

        sec_depth = np.nanmean(base_flux) - np.nanmean(sec_flux)
        sec_err = np.nanstd(sec_flux) / np.sqrt(len(sec_flux))
        sec_snr = sec_depth / sec_err

        return {
            "secondary_depth_ppm": float(sec_depth * 1e6),
            "secondary_snr": float(sec_snr),
            "passed": bool(sec_snr < self.sigma_threshold),
        }