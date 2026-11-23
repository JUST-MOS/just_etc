"""Check that the source-atmosphere research hook preserves the default ETC."""
import numpy as np

from just_etc import JUSTExposureTimeCalculator
from just_etc import ETC_py_optimized as engine


def test_default_source_atmosphere_matches_native_engine():
    calc = JUSTExposureTimeCalculator(calc_mode="fast")
    obs = calc._make_obs()
    for wavelength_nm in (400.0, 550.0, 765.0, 900.0):
        assert calc._source_atmospheric_transmission(obs, wavelength_nm) == engine.gsAtmTrans(obs, wavelength_nm, 0x0)


def test_research_subclass_can_replace_source_transmission():
    class FixedSourceAtmosphere(JUSTExposureTimeCalculator):
        def __init__(self, transmission):
            super().__init__(calc_mode="fast")
            self.transmission = transmission
            self._spectro.npix[:self._spectro.N_arms] = 64
            self.set_obs_conditions(field_angle=0.0)

        def _source_atmospheric_transmission(self, obs, lambda_nm):
            return self.transmission

        def _arm_noise(self, obs, t_exp):
            return [(np.ones(int(self._spectro.npix[arm])), 1.0)
                    for arm in range(self._spectro.N_arms)]

    wave = np.linspace(3000., 10000., 200)
    flux = np.full_like(wave, 1e-17)
    full = FixedSourceAtmosphere(1.0).compute_snr(wave, flux, t_exp=750)
    half = FixedSourceAtmosphere(0.5).compute_snr(wave, flux, t_exp=750)
    for one, two in zip(full, half):
        assert np.any(one["signal"] > 0)
        np.testing.assert_allclose(two["signal"], 0.5 * one["signal"], rtol=1e-12)
