"""Decoder-backend regression tests.

The pipeline/worker builds the DEM with
``decompose_errors=getattr(decoder, "decompose_errors", False)``.  PyMatching
requires a graphlike (decomposed) model: feeding it the raw DEM silently
mangles hyperedges and costs real effective distance (measured on the
diagonal-schedule d=3 memory: LER ~1.9x worse at p=2e-3, LER-slope d_eff
~1.8 instead of ~2.7).  Hypergraph decoders (mwpf/bposd/relay-bp) must keep
receiving the UNdecomposed DEM.
"""
import numpy as np
import pytest

from lightstim.noise.config import NoiseConfig
from lightstim.simulation.decoder_backend.registry import get_decoder

from tests.test_mixed_wall import memory_baseline

pytestmark = pytest.mark.smoke


def test_pymatching_declares_decompose_errors():
    dec = get_decoder("pymatching")
    assert getattr(dec, "decompose_errors", False) is True
    assert dec.enable_correlations is True  # K&F reference default
    assert get_decoder(
        "pymatching", enable_correlations=False).enable_correlations is False


def test_hypergraph_decoders_keep_raw_dem():
    for name in ("mwpf", "bposd"):
        try:
            dec = get_decoder(name)
        except (ImportError, ValueError):
            continue
        assert getattr(dec, "decompose_errors", False) is False


def test_decomposed_matcher_beats_raw_dem_matcher():
    """The bug's mechanism, deterministically: on identical shots, a matcher
    built from the raw hyperedge DEM loses to one built from the decomposed
    DEM by a wide margin (1.03e-2 vs 5.9e-3 at p=2e-3, seed 11)."""
    pymatching = pytest.importorskip("pymatching")
    p = 2e-3
    noisy = memory_baseline(3).build_noisy_circuit(
        noise_params=NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p,
                                 p_idle=p),
        noise_model='circuit_level')
    m_raw = pymatching.Matching.from_detector_error_model(
        noisy.detector_error_model())
    m_dec = pymatching.Matching.from_detector_error_model(
        noisy.detector_error_model(decompose_errors=True))
    det, obs = noisy.compile_detector_sampler(seed=11).sample(
        50_000, separate_observables=True)
    ler_raw = np.any(m_raw.decode_batch(det) != obs, axis=1).mean()
    ler_dec = np.any(m_dec.decode_batch(det) != obs, axis=1).mean()
    assert ler_dec < 0.75 * ler_raw, (ler_raw, ler_dec)
