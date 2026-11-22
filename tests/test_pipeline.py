import numpy as np
import lightkurve as lk
from astropy.time import Time
from src.characterizer import PlanetCharacterizer
from src.vetter import TransitVetter
from src.detector import TransitDetector


def test_planetary_radius_and_orbit_math():
    """Verifies Earth-Sun transit benchmark: 1 R_earth around 1 R_sun at 1 AU."""
    char = PlanetCharacterizer()
    sun_params = {"tic_id": 0, "r_star_solar": 1.0, "m_star_solar": 1.0, "teff_k": 5778.0}

    # Earth-to-Sun radius ratio is ~0.009158 -> depth = (R_earth / R_sun)^2
    earth_depth = (char.R_EARTH / char.R_SUN) ** 2
    res = char.compute_planetary_params(
        period_days=365.25, depth=earth_depth, stellar_params=sun_params, albedo=0.3
    )

    assert np.isclose(res["radius_earth"], 1.0, rtol=1e-3)
    assert np.isclose(res["semi_major_axis_au"], 1.0, rtol=1e-2)
    assert 250.0 < res["equilibrium_temp_k"] < 260.0


def test_odd_even_vetting_passes_consistent_transits():
    """Ensures synthetic light curve with equal odd/even depths passes vetting."""
    time = np.linspace(1000.0, 1050.0, 5000)
    flux = np.ones_like(time)
    period = 5.0
    t0 = 1002.0
    duration = 0.15

    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    flux[np.abs(phase) < 0.5 * duration] = 0.998  # 2000 ppm depth

    lc = lk.LightCurve(time=time, flux=flux)
    vetter = TransitVetter(sigma_threshold=3.0)
    res = vetter.check_odd_even(lc, period=period, t0=t0, duration=duration)

    assert res["passed"] is True
    assert res["sigma_diff"] < 0.1


def test_detector_handles_masked_astropy_time():
    """Regression test ensuring masked Time arrays do not crash multi-planet masking."""
    raw_time = np.ma.masked_array(np.linspace(1325.0, 1350.0, 2000), mask=False)
    raw_time.mask[10:15] = True
    time = Time(raw_time, format="btjd")
    flux = np.ones(2000)

    lc = lk.LightCurve(time=time, flux=flux)
    detector = TransitDetector(min_period=1.0, max_period=5.0, n_periods=500)
    candidates = detector.search_multi_planet(lc, max_planets=2)

    assert len(candidates) == 2
