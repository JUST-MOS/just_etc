"""Physical checks of exact Gaussian-mixture aperture integration."""
import numpy as np
import pytest
from scipy.integrate import dblquad
from scipy.stats import ncx2

from just_etc import JUSTExposureTimeCalculator
from just_etc import ETC_py_optimized as engine
from just_etc.mixture_fiber_coupling import (
    GaussianMixtureProfile, gaussian_mixture_fiber_fraction,
    etc_mixture_fiber_fraction, compute_snr_with_mixture,
)


def test_circular_mixture_matches_ncx2_and_offcenter_point():
    amps = np.array([.65, .30, .05])
    variances = np.array([.2, 1.4, 0.])
    re, sigma = 1.2, 1./2.354820045
    profile = GaussianMixtureProfile(re, 0., 0., amps, variances)
    for offset in (0., .03, .5, 1.5):
        var = sigma**2 + variances*re**2
        expected = np.dot(amps, ncx2.cdf(.75**2/var, 2, offset**2/var))
        actual = gaussian_mixture_fiber_fraction(profile, decenter_arcsec=offset)
        assert actual == pytest.approx(expected, rel=1e-11, abs=1e-12)


def test_elliptical_offset_matches_independent_2d_integral():
    profile = GaussianMixtureProfile(1.3, .4, 0., [.8, .2], [.5, 0.])
    sigma = 1./2.354820045
    offset, angle = .4, .7
    dx, dy = offset*np.cos(angle), offset*np.sin(angle)
    expected = 0.
    for amplitude, v in zip(profile.amplitudes, profile.variances):
        cx = sigma**2+v*profile.r_eff_arcsec**2
        cy = sigma**2+v*(profile.axis_ratio*profile.r_eff_arcsec)**2
        def density(r, theta):
            x, y = r*np.cos(theta), r*np.sin(theta)
            return (r*np.exp(-.5*((x-dx)**2/cx+(y-dy)**2/cy)) /
                    (2*np.pi*np.sqrt(cx*cy)))
        expected += amplitude*dblquad(density, 0., 2*np.pi,
                                     lambda _:0., lambda _:.75, epsabs=1e-10)[0]
    actual = gaussian_mixture_fiber_fraction(profile, decenter_arcsec=offset,
                                              offset_angle_deg=np.rad2deg(angle))
    assert actual == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_invalid_mixture_is_rejected():
    with pytest.raises(ValueError):
        GaussianMixtureProfile(1., 0., 0., [.4, .5], [1., 2.])
    with pytest.raises(ValueError):
        GaussianMixtureProfile(1., 0., 0., [1.], [-1.])


def test_etc_wrapper_updates_source_and_poisson_noise_only():
    calc = JUSTExposureTimeCalculator(calc_mode="accurate")
    calc.set_obs_conditions(seeing_fwhm_800=.8, field_angle=0., decenter=.03, r_eff=1.2)
    calc._spectro.npix[:] = 64
    wave = np.linspace(3000.,10000.,200)
    flux = np.full_like(wave, 1e-17)
    profile = GaussianMixtureProfile(1.2,.2,.1,[.75,.2,.05],[.2,1.,0.])
    original = calc.compute_snr(wave, flux, t_exp=750., n_exp=2)
    updated = compute_snr_with_mixture(calc, wave, flux, 750., 2, profile)
    for a,b in zip(original,updated):
        wavelength = a["wave_nm"][20]
        old_geo = engine.gsGeometricThroughput(calc._spectro,calc._make_obs(),
                                               wavelength,1.2,.03,0.,0)
        new_geo = etc_mixture_fiber_fraction(profile,calc._spectro,calc._make_obs(),wavelength)
        assert b["signal"][20]/a["signal"][20] == pytest.approx(new_geo/old_geo,rel=2e-4)
        assert (b["noise_var"][20]-a["noise_var"][20]) == pytest.approx(
            b["signal"][20]-a["signal"][20],abs=1e-8,rel=1e-10)
        np.testing.assert_allclose(b["snr"], b["snr_1exp"]*np.sqrt(2.))
