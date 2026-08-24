"""Paper Section 3: patch lifetimes.

compute_lifetimes derives each patch's first/last-use window from
the PPM sequence (the lifetime= hook may widen it), and
LivenessMixin carries the driver's retirement sweep and the
freed-cell reuse map.  Mixed into SequentialPPMExperiment; methods
run on the driver's state.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from circls.core.sequential_ppm_ls import PPMStep


def compute_lifetimes(ppm_sequence: List[PPMStep]) -> Dict[str, Tuple[int, int]]:
    """{patch_name: (first_use, last_use)} — layer indices of the first and last
    PPM touching each patch."""
    first: Dict[str, int] = {}
    last: Dict[str, int] = {}
    for i, step in enumerate(ppm_sequence):
        for nm, _P in step.interaction_type:
            first.setdefault(nm, i)
            last[nm] = i
    return {nm: (first[nm], last[nm]) for nm in first}



class LivenessMixin:
    """Retirement sweep + freed-cell reuse (docs/ARCHITECTURE.md)."""

    def _retire_consumed(self, i, step, owner, measured, allocated):
        """liveness: measure out + retire every CONSUMED patch whose lifetime
        window closes at or before step ``i`` (not an output in keep_patches,
        not the final layer).  Sweeps every STANDING patch, not only step
        ``i``'s targets: a widened ``lifetime_overrides`` free layer falls on
        a step that does not target the patch, and only a full sweep executes
        the retirement there — the router's absent-rule (``lu < i``) assumes
        it happened.  ``<=`` (not ``==``) so a missed call self-heals at the
        next step (M4: the wall/snake branches used to skip retirement while
        the absent-rule assumed it had happened).
        Called from ALL step-execution paths."""
        if not self.liveness:
            return
        if i >= len(self.ppm_sequence) - 1:
            return
        for nm in sorted(set(allocated) - measured):
            lt = self.lifetimes.get(nm)
            if (lt is not None and lt[1] <= i
                    and nm not in self.keep_patches):
                dq = [q for q in self.system.data_indices
                      if owner.get(q) == nm]
                self._readout_with_capture(
                    {q: self.final_measure_states[nm] for q in dq})
                self.system.retire_measured_patch(nm)
                measured.add(nm)

    def _lazy_reuse_map(self):
        """Identify PPMs that REUSE a retired patch's freed space as their corridor.

        Such a PPM carries an explicit ``route`` passing through the coarse cell of a
        patch whose LAST PPM use precedes it (so, under liveness, that patch is retired
        before this step runs). Its coupler must be registered LAZILY — after the
        retirement — because registering up front would steal the still-live patch's
        data qubits.  Returns ``{ppm_index: {retired_patch_names_to_exclude}}``.
        Reuse requires liveness (only then is a mid-sequence patch actually retired)."""
        lazy = {}
        if not self.liveness:
            return lazy
        for i, step in enumerate(self.ppm_sequence):
            if self._adjacent_pair(step):
                continue
            # every routed step after any retirement registers lazily: the
            # (auto- or explicitly-routed) corridor may reuse a retired
            # patch's freed cell, and binding up front would steal the
            # still-live patch's qubits (the v2m2 rule, generalized from
            # explicit routes to any corridor)
            excl = {nm for nm in self.lifetimes
                    if self.lifetimes[nm][1] < i
                    and nm not in self.keep_patches}
            if excl:
                lazy[i] = excl
        return lazy

