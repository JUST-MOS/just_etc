"""Opt-in elliptical-Sersic galaxy coupling to circular fibers.

Profiles are normalized to total light and are assumed constant with wavelength.
Legacy ellipticity is ``sqrt(e1**2+e2**2)=(a-b)/(a+b)`` and ``r_eff`` is the
half-light semi-major axis in arcsec.  A miscentering angle is measured relative
to that major axis; catalog ellipticity does not specify the offset direction.

The JUST PSF uses the same seeing, optical-spot, and diffraction transfer
functions as ETC_py_optimized.  A non-negative mixture of circular Gaussians
approximates that transfer function, then a radial/angular integral averages the
point-source aperture response over the galaxy.  This is a research model;
neither the catalog profile nor the assumed PSF is an on-sky calibration.
"""

from functools import lru_cache
import math

import numpy as np
from scipy.optimize import nnls
from scipy.special import gammaincinv, gammaln, roots_genlaguerre
from scipy.stats import ncx2


class SersicProfile(object):
    """A wavelength-independent elliptical Sersic light profile.

    Parameters are half-light semi-major radius [arcsec], Legacy ellipticity
    components e1/e2, and Sersic index. Values satisfy radius > 0, index > 0,
    and hypot(e1,e2) < 1.  For a point source use the existing ETC point model.
    """

    def __init__(self, r_eff_arcsec, e1, e2, sersic_index):
        self.r_eff_arcsec = float(r_eff_arcsec)
        self.e1 = float(e1)
        self.e2 = float(e2)
        self.sersic_index = float(sersic_index)
        values = (self.r_eff_arcsec, self.e1, self.e2, self.sersic_index)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("Sersic parameters must be finite")
        e = math.hypot(self.e1, self.e2)
        if self.r_eff_arcsec <= 0 or self.sersic_index <= 0 or e >= 1:
            raise ValueError("require r_eff > 0, sersic_index > 0 and |e| < 1")
        self.axis_ratio = (1. - e) / (1. + e)

    def offsets(self, decenter_arcsec=0., offset_angle_deg=0., nrad=128, ntheta=128):
        """Return source-element offsets and normalized radial probabilities.

        ``offset_angle_deg`` is relative to the galaxy major axis, not a sky
        position angle. Its default is an explicit major-axis assumption.
        """
        if not np.isfinite(decenter_arcsec) or decenter_arcsec < 0:
            raise ValueError("decenter_arcsec must be finite and nonnegative")
        if not np.isfinite(offset_angle_deg) or nrad < 8 or ntheta < 8:
            raise ValueError("invalid offset angle or quadrature dimensions")
        n = self.sersic_index
        nodes, weights = roots_genlaguerre(int(nrad), 2. * n - 1.)
        bn = gammaincinv(2. * n, .5)
        r = self.r_eff_arcsec * (nodes / bn)**n
        theta = (np.arange(int(ntheta)) + .5) * (2. * np.pi / int(ntheta))
        angle = np.deg2rad(offset_angle_deg)
        dx = r[:, None] * np.cos(theta)[None, :] - decenter_arcsec * np.cos(angle)
        dy = r[:, None] * self.axis_ratio * np.sin(theta)[None, :] - decenter_arcsec * np.sin(angle)
        prob = weights / np.exp(gammaln(2. * n))
        if np.any(~np.isfinite(prob)) or not np.isclose(prob.sum(), 1., rtol=1e-10):
            raise ValueError("Sersic radial quadrature could not be normalized")
        return np.hypot(dx, dy), prob


class _FiberKernel(object):
    """Interpolated circular-fiber response to an offset point source."""

    def __init__(self, fiber_radius_arcsec, sigmas_arcsec, weights):
        radius = float(fiber_radius_arcsec)
        sigmas = np.asarray(sigmas_arcsec, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if radius <= 0 or np.any(sigmas <= 0) or np.any(weights < 0):
            raise ValueError("invalid Gaussian-mixture fiber PSF")
        if not np.isclose(weights.sum(), 1., rtol=1e-10):
            raise ValueError("Gaussian-mixture weights must sum to one")
        # Dense where coupling changes quickly, wider in the faint PSF wings.
        tail_end = max(1200., radius + 12. * float(np.max(sigmas)))
        grid = np.r_[np.linspace(0., 5., 2501), np.linspace(5.02, 20., 750),
                     np.geomspace(20.02, tail_end, 1200)]
        response = np.zeros_like(grid)
        for sigma, weight in zip(sigmas, weights):
            # Beyond 12 sigma outside the aperture, the omitted Gaussian tail
            # probability is bounded by exp(-72).
            use = grid <= radius + 12. * sigma
            if np.any(use):
                response[use] += weight * ncx2.cdf((radius / sigma)**2, 2,
                                                   (grid[use] / sigma)**2)
        if np.any(~np.isfinite(response)) or np.any(response < 0):
            raise ValueError("non-finite point-source aperture response")
        self.radius = radius
        self.sigmas = sigmas
        self.weights = weights
        self.grid = grid
        self.response = response

    def point_response(self, offsets):
        """Coupled fractions at radial offsets [arcsec]."""
        return np.interp(offsets, self.grid, self.response, left=self.response[0], right=0.)

    def integrate(self, offsets, radial_probability):
        """Weighted angular and radial integral for one Sersic profile."""
        p = self.point_response(offsets)
        result = float(np.dot(radial_probability, np.mean(p, axis=1)))
        if not np.isfinite(result) or not 0 <= result <= 1:
            raise ValueError("invalid galaxy coupling")
        return result


@lru_cache(maxsize=256)
def _gaussian_kernel(fiber_radius_arcsec, seeing_fwhm_arcsec):
    if not np.isfinite(seeing_fwhm_arcsec) or seeing_fwhm_arcsec <= 0:
        raise ValueError("Gaussian seeing FWHM must be positive")
    sigma = seeing_fwhm_arcsec / (2. * np.sqrt(2. * np.log(2.)))
    return _FiberKernel(fiber_radius_arcsec, [sigma], [1.])


def gaussian_fiber_fraction(profile, fiber_diameter_arcsec=1.5,
                            seeing_fwhm_arcsec=1., decenter_arcsec=0.,
                            offset_angle_deg=0., nrad=128, ntheta=128):
    """Circular-aperture fraction for a Gaussian PSF.

    Defaults reproduce the stated Legacy FIBERFLUX geometry, but a matched
    analytical Sersic profile need not exactly reproduce Tractor rendering.
    Diameter, seeing and offset are in arcseconds; offset angle is in degrees.
    """
    if not isinstance(profile, SersicProfile):
        raise TypeError("profile must be SersicProfile")
    if not np.isfinite(fiber_diameter_arcsec) or fiber_diameter_arcsec <= 0:
        raise ValueError("fiber diameter must be positive")
    kernel = _gaussian_kernel(float(fiber_diameter_arcsec / 2.), float(seeing_fwhm_arcsec))
    offsets, weights = profile.offsets(decenter_arcsec, offset_angle_deg, nrad, ntheta)
    return kernel.integrate(offsets, weights)


@lru_cache(maxsize=256)
def _etc_kernel(fiber_radius_arcsec, seeing_800, wavelength_nm, optical_sigma, theta_diffraction):
    """Fit the ETC radial OTF with a positive Gaussian scale mixture."""
    if (fiber_radius_arcsec <= 0 or seeing_800 <= 0 or wavelength_nm <= 0 or
            optical_sigma < 0 or theta_diffraction < 0):
        raise ValueError("invalid ETC geometry or PSF parameters")
    uscale = .465 / seeing_800 * (wavelength_nm / 800.)**.2
    u_max = max(3., 8. * uscale)
    u = np.linspace(0., u_max, 601)
    sigmas = np.geomspace(.005, max(80., 10. * seeing_800), 100)
    design = np.exp(-2. * np.pi**2 * u[:, None]**2 * sigmas[None, :]**2)
    target = np.exp(-(u / uscale)**(5. / 3.) - (4. / np.pi) * u * theta_diffraction)
    importance = np.ones_like(u)
    importance[0] = 100.
    weights = nnls(design * importance[:, None], target * importance, maxiter=2000)[0]
    if weights.sum() <= 0:
        raise ValueError("could not fit ETC PSF")
    weights /= weights.sum()
    error = np.max(np.abs(design.dot(weights) - target))
    if error > 2e-5:
        raise ValueError("Gaussian-mixture approximation to ETC PSF is inadequate")
    keep = weights > 1e-10
    weights = weights[keep]
    weights /= weights.sum()
    sigmas = np.sqrt(sigmas[keep]**2 + optical_sigma**2)
    return _FiberKernel(fiber_radius_arcsec, sigmas, weights)


def etc_fiber_fraction(profile, spectro, obs, wavelength_nm, decenter_arcsec=.03,
                       field_angle_deg=0., offset_angle_deg=0., nrad=128, ntheta=128):
    """Return wavelength-dependent coupling under the ETC geometric PSF.

    ``spectro`` and ``obs`` are the ETC engine's configured objects. Wavelength
    may be a scalar or 1-D nm array. ``decenter_arcsec`` is the centroid offset;
    its angle is measured relative to the galaxy major axis.  The default
    treatment has no color gradients and uses the same profile at all wavelengths.
    """
    from . import ETC_py_optimized as engine
    if not isinstance(profile, SersicProfile):
        raise TypeError("profile must be SersicProfile")
    wave = np.asarray(wavelength_nm, dtype=float)
    if wave.ndim > 1 or np.any(~np.isfinite(wave)) or np.any(wave <= 0):
        raise ValueError("wavelength_nm must be positive and scalar or 1-D")
    if not np.isfinite(field_angle_deg):
        raise ValueError("field angle must be finite")
    grid = np.arange(5, dtype=float)
    t = np.clip(4. * field_angle_deg / spectro.rfov, 0., 4.)
    efl = float(np.interp(t, grid, spectro.EFL))
    spot = float(np.interp(t, grid, spectro.rms_spot))
    fiber_r = spectro.fiber_ent_rad / efl * engine.ARCSEC_PER_URAD
    sigma = spot / efl * engine.ARCSEC_PER_URAD
    offsets, weights = profile.offsets(decenter_arcsec, offset_angle_deg, nrad, ntheta)
    result = []
    for value in np.atleast_1d(wave):
        theta_d = .001 * value / spectro.D_outer / (1. - spectro.centobs) * engine.ARCSEC_PER_URAD
        kernel = _etc_kernel(float(fiber_r), float(obs.seeing_fwhm_800), float(value),
                             float(sigma), float(theta_d))
        result.append(kernel.integrate(offsets, weights))
    return float(result[0]) if wave.ndim == 0 else np.asarray(result)
