import numpy as np
import batman
import emcee
import corner
import matplotlib.pyplot as plt
import lightkurve as lk


class BayesianTransitFitter:
    """Fits an analytical Mandel-Agol transit model using MCMC (batman + emcee)."""

    def __init__(self, n_walkers: int = 24, n_steps: int = 800, burn_in: int = 250):
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.burn_in = burn_in

    def _extract_transit_windows(
        self, lc: lk.LightCurve, period: float, t0: float, duration: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Isolates data within +/- 3 transit durations to speed up MCMC evaluation."""
        time = np.asarray(lc.time.value, dtype=float)
        flux = np.asarray(lc.flux.value, dtype=float)
        flux_err = np.asarray(lc.flux_err.value, dtype=float)

        # Fallback if flux_err contains NaNs
        if np.all(np.isnan(flux_err)):
            flux_err = np.full_like(flux, np.nanstd(flux))
        else:
            flux_err = np.where(np.isnan(flux_err), np.nanmedian(flux_err), flux_err)

        phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
        mask = np.abs(phase) < (3.0 * duration)
        return time[mask], flux[mask], flux_err[mask]

    def _build_batman_model(
        self, theta: tuple[float, float, float, float], time: np.ndarray, period: float
    ) -> np.ndarray:
        t0_fit, rp_rs, a_rs, inc = theta
        params = batman.TransitParams()
        params.t0 = t0_fit
        params.per = period
        params.rp = rp_rs
        params.a = a_rs
        params.inc = inc
        params.ecc = 0.0
        params.w = 90.0
        params.limb_dark = "quadratic"
        params.u = [0.3, 0.2]  # Standard solar-type TESS quadratic limb darkening

        m = batman.TransitModel(params, time)
        return m.light_curve(params)

    def fit(
        self,
        lc: lk.LightCurve,
        period: float,
        t0_init: float,
        duration: float,
        depth_init: float,
        target_name: str = "target",
    ) -> dict:
        print(f"[MCMC] Slicing transit windows and initializing {self.n_walkers} walkers...")
        time, flux, flux_err = self._extract_transit_windows(lc, period, t0_init, duration)

        rp_init = float(np.sqrt(max(depth_init, 1e-6)))
        a_init = max(3.0, period / (np.pi * duration))
        inc_init = 89.0

        def log_prior(theta):
            t0_f, rp_rs, a_rs, inc = theta
            if (
                abs(t0_f - t0_init) < 0.1
                and 0.001 < rp_rs < 0.3
                and 1.5 < a_rs < 100.0
                and 70.0 < inc <= 90.0
            ):
                return 0.0
            return -np.inf

        def log_likelihood(theta):
            model_flux = self._build_batman_model(theta, time, period)
            inv_sigma2 = 1.0 / (flux_err**2)
            return -0.5 * np.sum(((flux - model_flux) ** 2) * inv_sigma2)

        def log_probability(theta):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(theta)

        # Scatter walkers tightly around initial TLS estimates
        p0_center = np.array([t0_init, rp_init, a_init, inc_init])
        scales = np.array([1e-4, rp_init * 0.05, a_init * 0.05, 0.2])
        pos = p0_center + scales * np.random.randn(self.n_walkers, 4)
        pos[:, 3] = np.clip(pos[:, 3], 75.0, 89.9)  # Keep initial inclination inside prior

        sampler = emcee.EnsembleSampler(self.n_walkers, 4, log_probability)
        sampler.run_mcmc(pos, self.n_steps, progress=True)

        flat_samples = sampler.get_chain(discard=self.burn_in, thin=5, flat=True)
        labels = ["t0 [BTJD]", "Rp / R*", "a / R*", "Inclination [deg]"]

        # Extract 16th, 50th (median), and 84th percentiles (1-sigma uncertainties)
        results = {}
        for idx, key in enumerate(["t0", "rp_rs", "a_rs", "inc"]):
            q16, q50, q84 = np.percentile(flat_samples[:, idx], [16, 50, 84])
            results[key] = (float(q50), float(q84 - q50), float(q50 - q16))

        # Generate and save posterior corner plot
        safe_name = target_name.lower().replace(" ", "_")
        corner_path = f"outputs/{safe_name}_mcmc_corner.png"
        fig = corner.corner(
            flat_samples,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".5f",
            bins=20,
            plot_contours=False,
        )
        fig.savefig(corner_path, dpi=250)
        plt.close(fig)
        print(f"[MCMC] Saved posterior corner plot to {corner_path}")

        return results
