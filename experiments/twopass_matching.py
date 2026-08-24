"""Correlated frame post-pass ("two-pass") for PyMatching.

Some decomposed DEMs contain FICTIONAL weight-1 boundary edges: no
physical mechanism produces that single-detector syndrome alone -- the
edge exists only as the leftover component of several hyperedge
decompositions.  When those hyperedge families sit on opposite sides
of an observable worldsheet, the merged edge carries two frame
variants of equal probability, PyMatching keeps whichever was
inserted first, and the losing family is mis-corrected at FIRST
order: the LER floor equals that family's probability mass (measured
0.048 at p = 5e-4 on toffoli_n3 d5, 17 conflicted edges; fredkin 16;
adder and bell have none).  No frame or weight assignment on the
existing graph can help -- the exhaustive frame optimization recovers
zero -- but the full syndromes of the families are globally unique,
and the pass-1 matching solution itself identifies the family through
the companion edges it uses.

The post-pass: give every conflicted edge (and every companion edge
of its families) an extra indicator fault id; after decode_batch,
wherever a conflicted edge fired, re-apply the frame of the
highest-probability family whose companion edges all fired in the
same solution.  Pairings are untouched; circuits without conflicted
edges decode identically to stock PyMatching.  Measured on toffoli_n3
d5 (denested): 128/128 first-order mis-corrections eliminated, LER
0.062 -> 0.022 at p = 5e-4 (MWPF reference 0.0084 -- the remaining
gap is generic second-order matching loss, not this defect).

The input DEM must be decomposable (build with decompose_errors=True;
de-nest the detector basis first where the raw basis is not).
"""
import collections
import math

import numpy as np
import pymatching
import stim


def _parse_mechs(dem):
    """[(all_dets, all_obs, p, [(comp_dets, comp_obs), ...])] from a
    flattened decomposed DEM."""
    mechs = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        comps, dd, ob = [], set(), set()
        for t in inst.targets_copy():
            if t.is_separator():
                comps.append((frozenset(dd), frozenset(ob)))
                dd, ob = set(), set()
            elif t.is_relative_detector_id():
                dd ^= {t.val}
            else:
                ob ^= {t.val}
        comps.append((frozenset(dd), frozenset(ob)))
        alld, allo = frozenset(), frozenset()
        for cd, co in comps:
            alld ^= cd
            allo ^= co
        mechs.append((alld, allo, p, comps))
    return mechs


class TwoPassMatching:
    """PyMatching with the conflicted-edge frame post-pass."""

    def __init__(self, dem):
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables
        nd, no = self.num_detectors, self.num_observables
        mechs = _parse_mechs(dem)

        edge_obs_p = collections.defaultdict(
            lambda: collections.defaultdict(float))
        edge_first = {}
        for _alld, _allo, p, comps in mechs:
            for cd, co in comps:
                q = edge_obs_p[cd][co]
                edge_obs_p[cd][co] = q * (1 - p) + p * (1 - q)
                edge_first.setdefault(cd, co)
        conflicted = sorted((cd for cd, v in edge_obs_p.items() if len(v) > 1),
                            key=sorted)
        conf_set = set(conflicted)

        families = collections.defaultdict(list)
        for _alld, _allo, p, comps in mechs:
            if len(comps) < 2:
                continue
            for i, (cd, _co) in enumerate(comps):
                if cd in conf_set:
                    others = frozenset(c2 for j, (c2, _o2) in enumerate(comps)
                                       if j != i)
                    families[cd].append((p, others, comps[i][1]))
        for cd in families:
            agg = {}
            for p, others, co in families[cd]:
                k = (others, co)
                q = agg.get(k, 0.0)
                agg[k] = q * (1 - p) + p * (1 - q)
            families[cd] = sorted(((q, o, co) for (o, co), q in agg.items()),
                                  reverse=True)

        ind_edges = set(conf_set)
        for cd in families:
            for _p, others, _co in families[cd]:
                ind_edges |= set(others)
        ind_edges = sorted(ind_edges, key=sorted)
        ind_id = {cd: no + i for i, cd in enumerate(ind_edges)}

        def fbits(co):
            return tuple(1 if b in co else 0 for b in range(no))

        f_def = {cd: fbits(edge_first[cd]) for cd in conflicted}

        mm = pymatching.Matching()
        bound = nd
        for cd, variants in edge_obs_p.items():
            if not cd:
                # a detector-less component flips observables with no
                # syndrome: no matching edge can represent it, and no
                # decoder can act on it (it is a distance-0 hole of the
                # circuit, not of the decoder) -- leave it out of the
                # graph; the audit still counts its mis-correction.
                continue
            ptot = 0.0
            for q in variants.values():
                ptot = ptot * (1 - q) + q * (1 - ptot)
            w = math.log((1 - ptot) / ptot)
            co = edge_first[cd] if cd in conf_set else next(iter(variants))
            fids = set(co)
            if cd in ind_id:
                fids |= {ind_id[cd]}
            l = sorted(cd)
            a, b = (l[0], bound) if len(l) == 1 else (l[0], l[1])
            mm.add_edge(a, b, fault_ids=fids, weight=w,
                        error_probability=ptot, merge_strategy="disallow")
        mm.set_boundary_nodes({bound})
        self.matching = mm

        self._conflicted = conflicted
        self._conf_idx = {cd: ind_id[cd] - no for cd in conflicted}
        self._fam_lookup = {
            cd: [(np.array([ind_id[o] - no for o in others], dtype=np.int64),
                  np.array([a ^ b for a, b in zip(fbits(co), f_def[cd])],
                           dtype=np.uint8),
                  p)
                 for p, others, co in families[cd]]
            for cd in conflicted}

    @property
    def num_conflicted_edges(self):
        return len(self._conflicted)

    def decode_batch(self, dets):
        """dets: (n, num_detectors) bool/uint8 -> obs (n, num_observables)."""
        pred = self.matching.decode_batch(dets)
        no = self.num_observables
        # pymatching's output width is its max fault id + 1; observables
        # (or indicators) that appear on no edge are absent -- pad so the
        # caller always gets num_observables columns
        obs = np.zeros((pred.shape[0], no), dtype=np.uint8)
        w = min(no, pred.shape[1])
        obs[:, :w] = pred[:, :w]
        ind = pred[:, no:] if pred.shape[1] > no else \
            np.zeros((pred.shape[0], 0), dtype=np.uint8)
        for cd in self._conflicted:
            rows = np.nonzero(ind[:, self._conf_idx[cd]])[0]
            if len(rows) == 0:
                continue
            unresolved = np.ones(len(rows), dtype=bool)
            for comp_ids, flip, _p in self._fam_lookup[cd]:
                ok = ind[np.ix_(rows, comp_ids)].all(axis=1) & unresolved
                if ok.any():
                    obs[rows[ok]] ^= flip
                    unresolved &= ~ok
        return obs


def conflict_audit(dem):
    """Single-fault sweep: how much first-order mis-correction the stock
    matching carries on this DEM, and how much the two-pass removes.
    Returns a dict; ``stock_mass`` is the linear-in-p LER floor."""
    tp = TwoPassMatching(dem)
    mm0 = pymatching.Matching.from_detector_error_model(dem)
    nd, no = tp.num_detectors, tp.num_observables
    agg = {}
    for alld, allo, p, _comps in _parse_mechs(dem):
        k = (alld, allo)
        q = agg.get(k, 0.0)
        agg[k] = q * (1 - p) + p * (1 - q)
    keys = list(agg)
    dets = np.zeros((len(keys), nd), dtype=np.uint8)
    truth = np.zeros((len(keys), no), dtype=np.uint8)
    for i, (alld, allo) in enumerate(keys):
        for d in alld:
            dets[i, d] = 1
        for o in allo:
            truth[i, o] = 1
    stock = mm0.decode_batch(dets)[:, :no].astype(np.uint8)
    twop = tp.decode_batch(dets)
    out = {"mechanism_classes": len(keys),
           "conflicted_edges": tp.num_conflicted_edges}
    for name, pred in (("stock", stock), ("twopass", twop)):
        bad = (pred != truth).any(axis=1)
        out[f"{name}_mis"] = int(bad.sum())
        out[f"{name}_mass"] = float(sum(agg[keys[i]]
                                        for i in np.nonzero(bad)[0]))
    return out


def twopass_from_circuit(noisy_circuit):
    return TwoPassMatching(
        noisy_circuit.detector_error_model(decompose_errors=True))
