"""Exact circular-fiber coupling of elliptical Gaussian-mixture profiles.

This module ships no external survey-model coefficients. The caller supplies
normalized amplitudes and dimensionless variances, including any point core.
The existing Sersic and default ETC paths are unchanged.  LEGACY catalog pixel
rendering is a separate measurement convention, not part of this sky model.
"""
import numpy as np
from scipy.special import ndtr
from numpy.polynomial.legendre import leggauss

from .fiber_coupling import SersicProfile, _gaussian_kernel, _etc_kernel


class GaussianMixtureProfile(object):
    """An elliptical normalized Gaussian-mixture galaxy profile.

    ``r_eff_arcsec`` is the catalog major-axis scale; component major-axis
    variance is ``variance * r_eff_arcsec**2``. ``e1`` and ``e2`` use Legacy's
    ellipticity convention. A zero variance describes a central point source.
    Signed amplitudes are permitted for fitted Gaussian approximations; they
    must sum to one and all variances must be nonnegative.
    """

    def __init__(self, r_eff_arcsec, e1, e2, amplitudes, variances):
        base = SersicProfile(r_eff_arcsec, e1, e2, 1.)
        amp = np.asarray(amplitudes, dtype=float)
        var = np.asarray(variances, dtype=float)
        if amp.ndim != 1 or amp.size == 0 or amp.shape != var.shape:
            raise ValueError("amplitudes and variances must be equally sized 1-D arrays")
        if np.any(~np.isfinite(amp)) or np.any(~np.isfinite(var)) or np.any(var < 0):
            raise ValueError("mixture values must be finite and variances nonnegative")
        if not np.isclose(amp.sum(), 1., rtol=1e-10, atol=1e-12):
            raise ValueError("mixture amplitudes must sum to one")
        self.r_eff_arcsec = base.r_eff_arcsec
        self.e1, self.e2 = base.e1, base.e2
        self.axis_ratio = base.axis_ratio
        self.amplitudes, self.variances = amp.copy(), var.copy()


_QUAD_X, _QUAD_W = leggauss(64)
_THETA = np.concatenate([k*np.pi/2. + (_QUAD_X+1.)*np.pi/4. for k in range(4)])
_THETA_W = np.tile(_QUAD_W, 4) / 8.


def _coupling(profile, kernel, decenter_arcsec, offset_angle_deg):
    """Convolve source/PSF Gaussians and integrate each over the fiber."""
    if not isinstance(profile, GaussianMixtureProfile):
        raise TypeError("profile must be GaussianMixtureProfile")
    if not np.isfinite(decenter_arcsec) or decenter_arcsec < 0 or not np.isfinite(offset_angle_deg):
        raise ValueError("invalid fiber decenter or angle")
    source_var = profile.variances[None, :] * profile.r_eff_arcsec**2
    psf_var = kernel.sigmas[:, None]**2
    cx = (psf_var + source_var).reshape(-1, 1)
    cy = (psf_var + source_var*profile.axis_ratio**2).reshape(-1, 1)
    weights = (kernel.weights[:, None]*profile.amplitudes[None, :]).reshape(-1)
    co, si = np.cos(_THETA)[None, :], np.sin(_THETA)[None, :]
    direction = np.deg2rad(offset_angle_deg)
    dx = decenter_arcsec*np.cos(direction)
    dy = decenter_arcsec*np.sin(direction)
    a = co**2/cx + si**2/cy
    b = dx*co/cx + dy*si/cy
    c = dx*dx/cx + dy*dy/cy
    radius = kernel.radius
    e0 = np.exp(-c/2.)
    eR = np.exp(-.5*(a*radius**2 - 2*b*radius + c))
    root_a = np.sqrt(a)
    cdf_diff = ndtr(root_a*radius-b/root_a) - ndtr(-b/root_a)
    radial_gauss = np.sqrt(2*np.pi/a) * np.exp(-.5*(c-b*b/a)) * cdf_diff
    radial_integral = (e0-eR)/a + (b/a)*radial_gauss
    probability = np.sum(radial_integral / np.sqrt(cx*cy) * _THETA_W[None, :], axis=1)
    result = float(np.dot(weights, probability))
    if not np.isfinite(result) or result < -1e-10 or result > 1.+1e-10:
        raise ValueError("invalid Gaussian-mixture aperture fraction")
    return float(np.clip(result, 0., 1.))  # Only protects floating-point roundoff.


def gaussian_mixture_fiber_fraction(profile, fiber_diameter_arcsec=1.5,
                                    seeing_fwhm_arcsec=1., decenter_arcsec=0.,
                                    offset_angle_deg=0.):
    """Fraction through a circular fiber for a Gaussian seeing PSF."""
    if not np.isfinite(fiber_diameter_arcsec) or fiber_diameter_arcsec <= 0:
        raise ValueError("fiber diameter must be positive")
    kernel = _gaussian_kernel(float(fiber_diameter_arcsec/2.), float(seeing_fwhm_arcsec))
    return _coupling(profile, kernel, decenter_arcsec, offset_angle_deg)


def etc_mixture_fiber_fraction(profile, spectro, obs, wavelength_nm,
                               decenter_arcsec=.03, field_angle_deg=0., offset_angle_deg=0.):
    """Fraction at wavelengths [nm] using the existing ETC geometric PSF."""
    from . import ETC_py_optimized as engine
    wave = np.asarray(wavelength_nm, dtype=float)
    if wave.ndim > 1 or np.any(~np.isfinite(wave)) or np.any(wave <= 0):
        raise ValueError("wavelength_nm must be positive and scalar or 1-D")
    if not np.isfinite(field_angle_deg):
        raise ValueError("field_angle_deg must be finite")
    grid = np.arange(5, dtype=float)
    t = np.clip(4.*field_angle_deg/spectro.rfov, 0., 4.)
    efl = float(np.interp(t, grid, spectro.EFL))
    spot = float(np.interp(t, grid, spectro.rms_spot))
    radius = float(spectro.fiber_ent_rad/efl*engine.ARCSEC_PER_URAD)
    sigma = float(spot/efl*engine.ARCSEC_PER_URAD)
    result=[]
    for value in np.atleast_1d(wave):
        theta = .001*value/spectro.D_outer/(1.-spectro.centobs)*engine.ARCSEC_PER_URAD
        kernel = _etc_kernel(radius, float(obs.seeing_fwhm_800), float(value), sigma, float(theta))
        result.append(_coupling(profile, kernel, decenter_arcsec, offset_angle_deg))
    return float(result[0]) if wave.ndim == 0 else np.asarray(result)


def compute_snr_with_mixture(calc, wave_aa, flux_flam, t_exp, n_exp, profile,
                             offset_angle_deg=0.):
    """Opt-in ETC spectrum/SNR from a total-light spectrum and mixture profile.

    Evaluate the original ETC once, then replace only its source coupling and
    corresponding source Poisson variance. Sky, detector noise, dust, telescope
    response and extraction remain those of ``calc``. The previous API and
    archived research implementation are deliberately not modified.
    """
    from . import ETC_py_optimized as engine
    old = calc.compute_snr(wave_aa, flux_flam, t_exp=t_exp, n_exp=n_exp)
    sp, obs = calc._spectro, calc._make_obs()
    fa, decenter = calc._obs_params["field_angle"], calc._obs_params["decenter"]
    old_re = calc._obs_params.get("r_eff", 0.)
    nodes = np.linspace(min(sp.lmin[:sp.N_arms]), max(sp.lmax[:sp.N_arms]), 21)
    new_geo = etc_mixture_fiber_fraction(profile, sp, obs, nodes, decenter, fa, offset_angle_deg)
    out = []
    for arm in old:
        wave = arm["wave_nm"]
        old_geo = np.array([engine.gsGeometricThroughput(sp, obs, float(w), old_re,
                            decenter, fa, 0) for w in wave])
        if np.any(old_geo <= 0):
            raise ValueError("invalid original geometric throughput")
        counts = arm["signal"] * (np.interp(wave, nodes, new_geo)/old_geo)
        sample_factor = 1.2 if sp.Dtype[arm["arm"]] == 1 else 1.
        variance = arm["noise_var"] + sample_factor*(counts-arm["signal"])
        if np.any(~np.isfinite(variance)) or np.any(variance < 0):
            raise ValueError("invalid variance after updating source counts")
        snr = np.divide(counts, np.sqrt(variance), out=np.zeros_like(counts), where=variance > 0)
        updated = dict(arm)
        updated.update(signal=counts, noise_var=variance, snr_1exp=snr,
                       snr=snr*np.sqrt(n_exp))
        out.append(updated)
    return out
