"""Feasibility-only rotation fallback (design decision 2026-08-05): with the
rotation planner OFF, a step whose live orientations admit no legal
construction must still get its blocked patch rotated — the fallback is
independent of ``auto_rotate``.

The natural trigger needs blocked legal faces, which the Square Sparse
floor avoids by construction on today's benchmark set, so the mechanism
test induces the blockage: the route oracle refuses every candidate that
keeps q1 in its birth orientation and delegates to the real router once
q1 is flipped.  Everything downstream of the probe (litinski protocol,
bookkeeping, re-registration, detectors) is real.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from benchsuite import dj, teleport_chain

from circls.pipeline import compile_qasm
from circls.core.sequential_ppm_ls import SequentialPPMExperiment


def _silent(cp, shots=256):
    det, _ = cp.circuit.compile_detector_sampler(seed=0).sample(
        shots, separate_observables=True)
    return not det.any()


def test_planner_off_is_default_and_rotation_free_here():
    cp = compile_qasm(dj(8))
    assert cp.experiment.auto_rotate is False
    assert cp.experiment.rotation_log == []
    assert _silent(cp)


def test_fallback_rotates_when_blocked(monkeypatch):
    orig = SequentialPPMExperiment._route_result
    birth_orient = {}

    def blocking(self, specs, step, bus, probe=False, conj=None,
                 raise_errors=False, **kw):
        if not birth_orient:
            birth_orient.update({s.name: s.orientation for s in self.patches})
        tgt = [nm for nm, _ in step.interaction_type]
        if "q1" in tgt:
            sp1 = next(s for s in specs if s.name == "q1")
            if sp1.orientation == birth_orient["q1"]:
                if raise_errors:
                    raise ValueError("induced blockage: q1 faces walled off")
                return None
        return orig(self, specs, step, bus, probe=probe, conj=conj,
                    raise_errors=raise_errors, **kw)

    monkeypatch.setattr(SequentialPPMExperiment, "_route_result", blocking)
    cp = compile_qasm(teleport_chain(2))
    rotated = [nm for _, nm, kind in cp.experiment.rotation_log
               if kind == "litinski"]
    assert "q1" in rotated, "blocked patch was never rotated"
    assert cp.experiment.auto_rotate is False
    assert _silent(cp)


def test_fallback_reraises_when_unrepairable(monkeypatch):
    def always_fail(self, specs, step, bus, probe=False, conj=None,
                    raise_errors=False, **kw):
        if raise_errors:
            raise ValueError("induced: nothing routes")
        return None

    monkeypatch.setattr(SequentialPPMExperiment, "_route_result", always_fail)
    with pytest.raises(ValueError, match="nothing routes"):
        compile_qasm(teleport_chain(2))
