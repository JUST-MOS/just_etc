"""Physical and integration checks for the opt-in Sersic fiber model."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j0, j1
from scipy.stats import ncx2

from just_etc import (JUSTExposureTimeCalculator, SersicProfile,
                      etc_fiber_fraction, gaussian_fiber_fraction)
from just_etc import ETC_py_optimized as engine
from just_etc.fiber_coupling import _etc_kernel


def configured_calculator(seeing=.8):
    calc = JUSTExposureTimeCalculator(calc_mode="accurate")
    calc.set_obs_conditions(seeing_fwhm_800=seeing, field_angle=0., decenter=.03)
    calc._set_mode()
    calc._configure_spectro(1)
    return calc


def test_gaussian_circular_exponential_matches_independent_integral():
    profile = SersicProfile(1.5, 0., 0., 1.)
    sigma = 1. / (2 * np.sqrt(2 * np.log(2)))
    rs = 1.5 / 1.67834699001666
    integrand = lambda r: (r / rs**2 * np.exp(-r / rs) *
                           ncx2.cdf((.75 / sigma)**2, 2, (r / sigma)**2))
    expected = quad(integrand, 0, .75 + 12 * sigma, epsabs=1e-11)[0]
    assert gaussian_fiber_fraction(profile) == pytest.approx(expected, rel=2e-4)


def test_shapes_and_miscentering_are_physical():
    with pytest.raises(ValueError):
        SersicProfile(1., 1., 0., 1.)
    with pytest.raises(ValueError):
        SersicProfile(0., 0., 0., 1.)
    profile = SersicProfile(1.5, .4, 0., 2.5)
    assert profile.axis_ratio == pytest.approx(.6 / 1.4)
    centered = gaussian_fiber_fraction(profile)
    assert 0 < centered < 1
    assert centered == pytest.approx(gaussian_fiber_fraction(profile, offset_angle_deg=90.), rel=1e-13)
    major = gaussian_fiber_fraction(profile, decenter_arcsec=.7, offset_angle_deg=0.)
    minor = gaussian_fiber_fraction(profile, decenter_arcsec=.7, offset_angle_deg=90.)
    assert centered > major > minor > 0


def test_etc_gaussian_mixture_matches_independent_fourier_integral():
    calc = configured_calculator()
    sp, obs = calc._spectro, calc._make_obs()
    wave = 623.1
    radius = sp.fiber_ent_rad / sp.EFL[0] * engine.ARCSEC_PER_URAD
    sigma = sp.rms_spot[0] / sp.EFL[0] * engine.ARCSEC_PER_URAD
    theta = .001 * wave / sp.D_outer / (1 - sp.centobs) * engine.ARCSEC_PER_URAD
    us = .465 / obs.seeing_fwhm_800 * (wave / 800.)**.2
    kernel = _etc_kernel(float(radius), float(obs.seeing_fwhm_800), wave,
                         float(sigma), float(theta))
    for offset in (0., .5, 1., 2., 5.):
        def integrand(u):
            return (radius * 2 * np.pi * j1(2 * np.pi * u * radius) *
                    j0(2 * np.pi * u * offset) *
                    np.exp(-2 * np.pi**2 * u**2 * sigma**2 - (u/us)**(5/3) - 4*u*theta/np.pi))
        expected = quad(integrand, 0., 4., epsabs=1e-11, limit=1000)[0]
        assert float(kernel.point_response(offset)) == pytest.approx(expected, abs=2e-7)


def test_etc_exponential_case_agrees_with_original_model():
    calc = configured_calculator(seeing=1.0505042136624068)
    sp, obs = calc._spectro, calc._make_obs()
    for radius in (.5, 1., 2.):
        profile = SersicProfile(radius, 0., 0., 1.)
        new = etc_fiber_fraction(profile, sp, obs, 623.1, decenter_arcsec=.03)
        old = engine.gsGeometricThroughput(sp, obs, 623.1, radius, .03, 0., 0)
        assert new == pytest.approx(old, rel=.005)


def test_api_opt_in_changes_only_source_coupling_and_keeps_default():
    calc = configured_calculator()
    # The sky line kernel needs at least its 33-pixel LSF support.
    calc._spectro.npix[:] = 64
    calc.set_obs_conditions(r_eff=1.5)
    wave = np.linspace(3000., 10000., 200)
    flux = np.full_like(wave, 1e-17)
    a = calc.compute_snr(wave, flux, 750.)
    b = calc.compute_snr(wave, flux, 750., galaxy_profile=None)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x["signal"], y["signal"])
    profile = SersicProfile(1.5, .3, .2, 3.)
    changed = calc.compute_snr(wave, flux, 750., galaxy_profile=profile)
    for old, new in zip(a, changed):
        w = old["wave_nm"][3]
        old_geo = engine.gsGeometricThroughput(calc._spectro, calc._make_obs(), w, 1.5, .03, 0., 0)
        new_geo = etc_fiber_fraction(profile, calc._spectro, calc._make_obs(), w)
        assert new["signal"][3] / old["signal"][3] == pytest.approx(new_geo / old_geo, rel=2e-4)
        # Source Poisson noise changes with the source; background stays fixed.
        assert (new["noise_var"][3] - old["noise_var"][3]) == pytest.approx(
            new["signal"][3] - old["signal"][3], rel=1e-10, abs=1e-8)


def test_mock_and_exposure_solver_propagate_optional_profile():
    calc = configured_calculator()
    calc._spectro.npix[:] = 64
    wave = np.linspace(3000., 10000., 200)
    flux = np.full_like(wave, 1e-17)
    profile = SersicProfile(1.2, .2, .1, 2.)
    baseline = calc.compute_snr(wave, flux, 750., galaxy_profile=profile)
    mock = calc.simulate_mock_observation(wave, flux, 750., seed=123,
                                          galaxy_profile=profile)
    for expected, observed in zip(baseline, mock):
        np.testing.assert_allclose(expected["signal"], observed["signal_e"], rtol=1e-13)
    ref_wave = baseline[0]["wave_nm"][30]
    target = baseline[0]["snr"][30]
    solved = calc.solve_exposure_time(wave, flux, target, ref_wave,
                                     t_exp_init=750., galaxy_profile=profile)
    assert solved["converged"]
    assert solved["t_exp"] == pytest.approx(750., rel=1e-6)
