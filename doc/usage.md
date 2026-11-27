# API and scientific conventions

## Spectra and instrument data

```python
from just_etc import (
    JUSTExposureTimeCalculator, load_template, list_templates, normalize_to_mag,
)

templates = list_templates()  # category/filename -> installed Path
wave, flux = load_template("starburst/sb4_kinney_fuv_001.fits")
flux, scale = normalize_to_mag(wave, flux, target_mag=21.5, band="r")
etc = JUSTExposureTimeCalculator(calc_mode="fast")
```

`load_template` accepts an existing local file (absolute or relative), a bundled
`category/filename`, or `templates/category/filename`. Existing local paths take
precedence. It loads the first FITS binary or ASCII table with wavelength and flux columns,
then sorts wavelengths in ascending order. Supply wavelength in Å and flux density
in erg/s/cm²/Å; the loader does not perform arbitrary FITS unit conversion.

The instrument default is bundled `data/spec.dat`. Override it explicitly with
`JUSTExposureTimeCalculator(spec_file="/path/to/spec.dat")`.

## Observing conditions

```python
etc.set_obs_conditions(
    seeing_fwhm_800=0.8,  # arcsec at 800 nm
    zenith_angle=45.0,   # degrees
    ebv=0.03,
    r_eff=0.5,           # arcsec; 0 for a spatial point source
    decenter=0.03,       # arcsec
    lunar_za=135.0,      # degrees; moon below the horizon
)
```

Additional parameters and defaults are in `JUSTExposureTimeCalculator._DEFAULTS`.
The three calculation modes are `fast`, `balanced`, and `accurate`. The API default
is `accurate`, whereas the CLI default is `fast`. The engine uses global calculation
settings; avoid concurrently using different modes in threads. Batch examples use
separate processes.

## Optional elliptical Sérsic fiber coupling

For a total-light input spectrum, a catalog profile can be used explicitly:

```python
from just_etc import SersicProfile

# SHAPE_R is the half-light semi-major axis in arcsec; E1/E2 are Legacy
# ellipticity components and n is the catalog Sérsic index.
galaxy = SersicProfile(r_eff_arcsec=1.2, e1=0.25, e2=0.10, sersic_index=2.5)
arms = etc.compute_snr(wave, flux, t_exp=900., n_exp=4,
                       galaxy_profile=galaxy, offset_angle_deg=0.)
```

When supplied, `galaxy_profile` replaces the `r_eff` observing parameter for
the source fiber coupling. The original circular exponential is retained when
it is omitted. `offset_angle_deg` is the direction of the fiber miscentering
relative to the galaxy's major axis; the default of 0 means along that axis.
The same optional parameters are accepted by `simulate_mock_observation` and
`solve_exposure_time`. The profile is assumed constant with wavelength; do not
apply this coupling to a spectrum already normalized to fiber flux.

The optional `gaussian_fiber_fraction` function provides a 1.5-arcsec aperture,
1-arcsec Gaussian-seeing comparison with Legacy catalog fiber fluxes. This and
the ETC PSF are model predictions, not on-sky calibration. They do not reproduce
Legacy Tractor image rendering exactly, especially for concentrated galaxies.
The JUST BGS study in `just_specsim` documents the sample validation.

### Optional catalog-consistent Gaussian-mixture coupling

For source profiles supplied as normalized elliptical Gaussian mixtures, an
independent opt-in wrapper can compute the JUST ETC response and noise:

```python
from just_etc.mixture_fiber_coupling import GaussianMixtureProfile, compute_snr_with_mixture

# Coefficients must be supplied by the caller from the appropriate catalog
# model. Their values here are illustrative only, not Legacy coefficients.
galaxy = GaussianMixtureProfile(r_eff_arcsec=1.2, e1=0.25, e2=0.10,
                                amplitudes=[0.6, 0.4], variances=[0.2, 1.0])
arms = compute_snr_with_mixture(etc, wave, flux, t_exp=900., n_exp=4,
                                profile=galaxy, offset_angle_deg=0.)
```

The mixture's `variances` are dimensionless major-axis variances in units of
`SHAPE_R**2`; zero variance represents a central point component. The input
flux must describe total source light. The wrapper uses the existing ETC PSF
and retains its sky and detector noise, replacing source coupling and its Poisson
variance. It does not modify the default `compute_snr` path. Legacy's image
pixelization is used only when reproducing its catalog aperture measurement;
it is not applied to the continuous JUST fiber. The JUST BGS study in
`just_specsim` documents the DR9.4 coefficients, comparison and scope.

The packaged `data/DESI.spec.dat` is the historical 2019 DESI comparison
configuration used in this research. Treat it as an engineering reference,
not a verified current description of the DESI instrument.

## SNR and exposure time

```python
arms = etc.compute_snr(wave, flux, t_exp=900.0, n_exp=4)
solution = etc.solve_exposure_time(
    wave, flux, target_snr=8.0, ref_wave_nm=600.0, n_exp=4,
    t_exp_init=600.0,
)
print(solution["t_exp"], solution["t_total"], solution["snr_achieved"])
```

Each SNR result dictionary contains:

| Key | Meaning |
| --- | --- |
| `arm` | Arm index, 0–2 |
| `wave_nm`, `wave_aa` | Native pixel-center wavelength grids |
| `signal` | Source electrons per pixel per exposure |
| `noise_var` | Noise variance per pixel per exposure, including configured terms |
| `snr_1exp` | Single-exposure per-pixel SNR |
| `snr` | Combined per-pixel SNR for `n_exp` exposures |

Compute summaries with NumPy; `snr_mean` and `snr_median` are not result keys.
The solver reference wavelength is in nm. Check the achieved SNR and warnings when
the target cannot be reached within the configured time/iteration bounds.

## Mock observations

```python
mock = etc.simulate_mock_observation(wave, flux, t_exp=900, n_exp=4, seed=42)
```

Results include `wave_aa`, `flux_intrinsic`, `flux_mock`, `signal_e`, `noise_e` and
`snr`. Noise is drawn in electron space using the supplied seed; source-free pixels
are treated as in the original implementation.

## Model limitations retained from the source

This packaging migration does not recalibrate the physical model. Magnitude
normalization uses mean FLAM in approximate band windows and an effective
wavelength; it is not synthetic photometry integrated over a real response curve.
Point/extended source settings describe the spatial profile, independently of the
spectral template. Input wavelengths outside the SED range are assigned zero flux.
Consult the source and historical technical manuals for the adopted atmosphere,
sky, throughput and systematic-noise assumptions before using results in research.
