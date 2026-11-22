import numpy as np
import lightkurve as lk
from astroquery.mast import Catalogs
from scipy.optimize import curve_fit


class PlanetCharacterizer:
    """Derives physical planetary parameters and measures Transit Timing Variations (TTVs)."""

    # Physical constants (SI units)
    G = 6.67430e-11
    M_SUN = 1.98847e30
    R_SUN = 6.957e8
    R_EARTH = 6.371e6
    AU = 1.495978707e11

    def fetch_stellar_params(self, target: str) -> dict:
        """Queries the TESS Input Catalog (TIC) via MAST for host star mass, radius, and Teff."""
        print(f"[Characterizer] Querying TESS Input Catalog for {target} stellar parameters...")
        tic_table = Catalogs.query_object(target, radius=0.02, catalog="TIC")
        if len(tic_table) == 0:
            raise ValueError(f"No TIC catalog entry found for {target}.")

        star = tic_table[0]
        return {
            "tic_id": int(star["ID"]),
            "r_star_solar": float(star["rad"]),
            "m_star_solar": float(star["mass"]),
            "teff_k": float(star["Teff"]),
        }

    def compute_planetary_params(
        self, period_days: float, depth: float, stellar_params: dict, albedo: float = 0.3
    ) -> dict:
        """Calculates planetary radius (Earth radii), semi-major axis (AU), and Teq (K)."""
        r_star_m = stellar_params["r_star_solar"] * self.R_SUN
        m_star_kg = stellar_params["m_star_solar"] * self.M_SUN
        teff = stellar_params["teff_k"]
        period_sec = period_days * 86400.0

        # 1. Planetary Radius: Rp = R_star * sqrt(depth)
        r_planet_m = r_star_m * np.sqrt(depth)
        r_planet_earth = r_planet_m / self.R_EARTH

        # 2. Semi-Major Axis via Kepler's Third Law: a = (G * M_star * P^2 / 4pi^2)^(1/3)
        a_m = ((self.G * m_star_kg * (period_sec**2)) / (4.0 * (np.pi**2))) ** (1.0 / 3.0)
        a_au = a_m / self.AU

        # 3. Equilibrium Temperature: Teq = Teff * sqrt(R_star / 2a) * (1 - A_B)^(1/4)
        t_eq = teff * np.sqrt(r_star_m / (2.0 * a_m)) * ((1.0 - albedo) ** 0.25)

        return {
            "radius_earth": float(r_planet_earth),
            "semi_major_axis_au": float(a_au),
            "equilibrium_temp_k": float(t_eq),
        }

    def measure_ttvs(
        self, lc: lk.LightCurve, period: float, t0: float, duration: float, depth: float
    ) -> dict:
        """Fits individual transit midpoints to compute Observed - Calculated (O - C) TTVs."""
        print("[Characterizer] Measuring individual transit midpoints for TTV analysis...")
        time = np.asarray(lc.time.value, dtype=float)
        flux = np.asarray(lc.flux.value, dtype=float)

        # Identify which integer epochs actually contain telescope observations
        all_epochs = np.round((time - t0) / period).astype(int)
        unique_epochs = np.unique(all_epochs)

        observed_epochs = []
        oc_minutes = []
        oc_errors_min = []

        # Gaussian dip approximation to fit the exact centroid of each individual transit
        def transit_dip(t, t_mid, amp):
            sigma = duration / 2.355
            return 1.0 - amp * np.exp(-0.5 * ((t - t_mid) / sigma) ** 2)

        for epoch in unique_epochs:
            t_calc = t0 + epoch * period
            window = np.abs(time - t_calc) < (1.5 * duration)
            t_win = time[window]
            f_win = flux[window]

            # Require sufficient data coverage across the transit window
            if len(t_win) < 15:
                continue

            try:
                popt, pcov = curve_fit(
                    transit_dip,
                    t_win,
                    f_win,
                    p0=[t_calc, depth],
                    bounds=([t_calc - 0.5 * duration, 0.0], [t_calc + 0.5 * duration, 0.05]),
                )
                t_obs = popt[0]
                t_err = np.sqrt(np.diag(pcov))[0]

                # Convert O - C from days to minutes
                observed_epochs.append(int(epoch))
                oc_minutes.append(float((t_obs - t_calc) * 1440.0))
                oc_errors_min.append(float(t_err * 1440.0))
            except RuntimeError:
                continue

        return {
            "epochs": np.array(observed_epochs),
            "oc_minutes": np.array(oc_minutes),
            "oc_errors_min": np.array(oc_errors_min),
        }
