"""K&F flag fold on a mixed (X<->Z) stretched wall -- the mixed-observable
decoder defect of the T-tier programs, on the smallest circuit that shows it.

test_mixed_wall.build_case(3, 'Z', 'X') (teleport init: one observable with
the joint outcome folded in) runs three rounds of the K&F relay wall.  Per
wall row a flag aux A (RX .. CZ feet .. MZ), a relay S and a syndrome aux B
(RZ .. CX feet .. MX) measure one stretched check; an X fault on A, S or B
before its feet leaves the check's whole one-side residual on the data, which
only the two neighbouring wall checks and a flag see.  With the flags booked
as their own detectors the decoder graph never used them: stim split the
three-symptom hook through a corner boundary edge and PyMatching miscorrected
it as a single fault (22 dangling-L pieces / 22 miscorrected mechanisms here,
LER 0.063 regardless of d on toffoli/fredkin at d=5).  The fold
(CircuitBuilder._kf_flag_fold_plan) XORs each row's flag records into ONE
neighbour's detector in the round that neighbour first sees the hook, so
every single fault flips at most two detectors with one observable mask.
"""
import collections

import numpy as np
import pymatching
import pytest
import stim

from experiments.noise_inject import inject_uniform_noise
from lightstim.ir.builder import CircuitBuilder
from tests.test_mixed_wall import build_case

pytestmark = pytest.mark.smoke

P = 5e-4


def _audit(circuit):
    """Decomposed-DEM shape and a single-fault PyMatching audit."""
    noisy = inject_uniform_noise(circuit, P)
    # raises on any mechanism stim cannot decompose into graphlike pieces
    dem = noisy.detector_error_model(decompose_errors=True)
    masks = collections.defaultdict(set)
    big = dangling = 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        groups = [[]]
        for t in inst.targets_copy():
            if t.is_separator():
                groups.append([])
            else:
                groups[-1].append(t)
        for g in groups:
            dets = frozenset(t.val for t in g if t.is_relative_detector_id())
            obs = frozenset(t.val for t in g if t.is_logical_observable_id())
            masks[dets].add(obs)
            big += len(dets) > 2
            dangling += (not dets) and bool(obs)
    ambiguous = sum(1 for v in masks.values() if len(v) > 1)

    matching = pymatching.Matching.from_detector_error_model(dem)
    raw = noisy.detector_error_model().flattened()
    mechs = [([t.val for t in inst.targets_copy()
               if t.is_relative_detector_id()],
              [t.val for t in inst.targets_copy()
               if t.is_logical_observable_id()])
             for inst in raw if inst.type == "error"]
    syn = np.zeros((len(mechs), noisy.num_detectors), dtype=np.uint8)
    truth = np.zeros((len(mechs), noisy.num_observables), dtype=np.uint8)
    for i, (d, o) in enumerate(mechs):
        syn[i, d] = 1
        truth[i, o] = 1
    pred = matching.decode_batch(syn).astype(np.uint8)
    miscorrected = int((pred ^ truth).any(axis=1).sum())
    return dict(pieces_gt2=big, ambiguous=ambiguous, dangling=dangling,
                miscorrected=miscorrected)


def _kf_rows(system):
    """(A, S, B) global indices of every registered kf check."""
    out = []
    for st in system.stabilizers:
        kf = st.get('kf')
        if kf:
            out.append((system.index_map[tuple(kf['flag'])],
                        system.index_map[tuple(kf['shared'])],
                        system.index_map[tuple(st['syn_coord'])]))
    return out


def _hook_flips(circuit, rows):
    """p=0 injection of the pair hooks: X_ERROR(1) on B right after its reset
    and on A right after CX(A, S), for every relay row and round.  Returns
    [(flipped detector ids, observable flipped, own-flag detector ids)]."""
    flat = list(circuit.flattened())
    dets = []
    nmeas = 0
    for inst in flat:
        if inst.name == "DETECTOR":
            dets.append(frozenset(nmeas + t.value
                                  for t in inst.targets_copy()))
        elif inst.name in ("M", "MZ", "MX", "MR", "MRZ", "MRX"):
            nmeas += sum(t.is_qubit_target for t in inst.targets_copy())
    ref_obs = circuit.compile_detector_sampler().sample(
        1, separate_observables=True)[1][0]

    def flips(pos, q):
        c = stim.Circuit()
        for i, inst in enumerate(flat):
            if i == pos:
                c.append("X_ERROR", [q], 1.0)
            c.append(inst)
        d, o = c.compile_detector_sampler().sample(
            1, separate_observables=True)
        return (set(int(x) for x in np.flatnonzero(d[0])),
                bool((o[0] != ref_obs).any()))

    out = []
    last_reset = {}
    cx_as = {}
    nmeas = 0
    for pos, inst in enumerate(flat):
        tg = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if inst.name in ("R", "RZ", "RX"):
            for q in tg:
                last_reset[q] = pos
        elif inst.name in ("CX", "CNOT"):
            for u, v in zip(tg[::2], tg[1::2]):
                cx_as[(u, v)] = pos
        elif inst.name in ("M", "MZ", "MX", "MR", "MRZ", "MRX"):
            rec = {q: nmeas + i for i, q in enumerate(tg)}
            nmeas += len(tg)
            for A, S, B in rows:
                if not {A, S, B} <= set(tg) or (A, S) not in cx_as:
                    continue
                own = {i for i, dr in enumerate(dets)
                       if dr and dr <= {rec[A], rec[S]}}
                for pos_inj, q in ((last_reset[B] + 1, B),
                                   (cx_as[(A, S)] + 1, A)):
                    d, ob = flips(pos_inj, q)
                    out.append((d, ob, own))
            last_reset, cx_as = {}, {}
    return out


def test_kf_fold_mixed_wall_graphlike_and_single_fault_clean():
    builder = build_case(3, 'Z', 'X')
    c = builder.circuit
    det, obs = c.compile_detector_sampler(seed=0).sample(
        256, separate_observables=True)
    assert not det.any() and not obs.any()
    assert _audit(c) == dict(pieces_gt2=0, ambiguous=0, dangling=0,
                             miscorrected=0)


def test_kf_fold_pair_hooks_flip_at_most_two_detectors(monkeypatch):
    builder = build_case(3, 'Z', 'X')
    rows = _kf_rows(builder.system)
    assert len(rows) == 3
    hooks = _hook_flips(builder.circuit, rows)
    assert len(hooks) == 2 * 3 * 3          # two hooks x rows x rounds
    for flipped, _, _ in hooks:
        assert len(flipped) <= 2, flipped
    # The same hooks on the UNFOLDED circuit (flags as their own detectors)
    # show what a hook does physically: one that flips the observable must
    # be seen by a check detector, else its residual is a bare logical the
    # fold could only hide behind a flag (a residual seen by nobody is a
    # stabilizer of the state and flips no observable).  Here the two hooks
    # of the open-end domino carry a logical component and are seen by the
    # neighbour check; after the fold they become the flag-boundary edge.
    with monkeypatch.context() as m:
        m.setattr(CircuitBuilder, "_kf_flag_fold_plan",
                  lambda self, syn_qubit_indices: None)
        raw = _hook_flips(build_case(3, 'Z', 'X').circuit, rows)
    assert [ob for _, ob, _ in raw] == [ob for _, ob, _ in hooks]
    for flipped, obs_flipped, own_flags in raw:
        if obs_flipped:
            assert flipped - own_flags, (flipped, own_flags)
    assert sum(ob for _, ob, _ in hooks) == 6


def test_kf_fold_is_what_removes_the_defect(monkeypatch):
    # the old bookkeeping (flags as their own differential detectors) on the
    # same circuit: dangling-L pieces and single-fault miscorrections
    monkeypatch.setattr(CircuitBuilder, "_kf_flag_fold_plan",
                        lambda self, syn_qubit_indices: None)
    a = _audit(build_case(3, 'Z', 'X').circuit)
    assert a["dangling"] > 0 and a["miscorrected"] > 0, a
