"""Paper Section 3: parallel merge windows.

Plans which contiguous patch-disjoint PPM steps share one merge
window and executes the shared window (the window is one
decision-plus-emission unit; batches demote to the serial path
whenever disjointness cannot be proven).  Mixed into
SequentialPPMExperiment.
"""
from __future__ import annotations

from circls.core.common import _CONJ
from circls.core.multi_patch_coupler import BentLayoutError
from lightstim.qec_code.surface_code.rotated.diagonal_se import (
    DiagonalSurfaceCodeExtractionBlock)
from lightstim.qec_code.surface_code.rotated.bent_joint_se import se_round_chunk


class ParallelWindowsMixin:
    """Parallel-window planning and execution (docs/ARCHITECTURE.md)."""

    def _plan_step_batches(self):
        """Contiguous runs of patch-disjoint steps become BATCHES executed
        in one shared merge window (parallel_steps).  Planning-time
        eligibility keeps v1 conservative: walls, snake hosts, steps with
        planned rotations, |Y>-gadget steps, birth-ride carriers and
        fallback-birth tops stay singletons.  Runtime demotion handles the
        rest (a member that classifies into a wall, or a batch the router
        cannot serve) by falling back to today's serial path — the design
        floor is current behaviour."""
        n = len(self.ppm_sequence)
        eligible = []
        for i, st in enumerate(self.ppm_sequence):
            names = {nm for nm, _ in st.interaction_type}
            # _lazy members reuse ground freed by a RETIREMENT — inside a
            # batch the freeing step shares the window, so its patch is
            # still alive on the reused cells (measured 2026-08-05:
            # row_major ghz-mixed batch [3,4,5,6], member 6 routed over
            # q9's cells while member 4 still owned them — add_patch's
            # collision guard fired mid-registration).  Same order
            # dependence for _rot_lazy.  Singletons, per v1 conservatism.
            ok = (i not in self._snake_plans
                  and not self._rotate_plan.get(i)
                  and i not in self._lazy
                  and i not in self._rot_lazy
                  and not self._birth_batch.get(i)
                  and not self._birth_batch.get(i + 1)
                  and not (names & self._y_names))
            eligible.append(ok)
        batches, cur, cur_names = [], [], set()
        for i, st in enumerate(self.ppm_sequence):
            names = {nm for nm, _ in st.interaction_type}
            if not eligible[i]:
                if cur:
                    batches.append(cur)
                batches.append([i])
                cur, cur_names = [], set()
                continue
            if cur and (cur_names & names):
                batches.append(cur)
                cur, cur_names = [], set()
            cur.append(i)
            cur_names |= names
        if cur:
            batches.append(cur)
        self._step_batches = batches
        self._batch_first = {b[0]: b for b in batches if len(b) > 1}
        self._batch_done = set()
        self._batch_members = {j for b in batches if len(b) > 1 for j in b}

    def _try_run_batch(self, batch, reg, owner, measured, allocated):
        """Attempt one shared-window execution of ``batch``.  Returns True
        with every member marked done, or False after only idempotent /
        harmless work (allocations stick; any couplers already registered
        carry pinned routes and execute correctly on the serial path)."""
        steps = {j: self.ppm_sequence[j] for j in batch}
        # A pre-registered route was built under the SERIAL contract: a
        # corridor may BORROW a later-born patch's cell (it is measured out
        # at the split, before the birth — see _specs_for_step).  This
        # batch's preamble births member patches EARLY, so such a route can
        # find a live patch standing on its borrowed ground (measured
        # 2026-08-05: row_major ghz-mixed, ppm_0's corridor through q9/q10's
        # cells while the demoted batch's preamble had already born both —
        # K&F validator caught the double Z-check).  Demote BEFORE any side
        # effect; executed batches route every member fresh, inside the
        # batch window's true obstacle set.
        if any(self._routes[j] is not None for j in batch):
            return False
        # per-member preamble: deferred allocation + adjacent classification
        newly = [nm for j in batch for nm, _ in steps[j].interaction_type
                 if nm not in allocated]
        # lifetime_overrides init side (see the deferred-allocation site in
        # build): a widened init layer falling inside the batch window is
        # allocated at the window start — the whole batch shares one time
        # window, so that IS layer fu.  Empty without overrides.
        newly += [nm for nm, (fu, _lu) in self.lifetimes.items()
                  if fu <= max(batch) and nm not in allocated
                  and nm not in newly and nm not in self._y_names]
        for nm in sorted(set(newly)):
            self._alloc_patch(nm, reg)
            allocated.add(nm)
        if newly:
            self._init_and_baseline(set(newly))
        for j in batch:
            if j in self._adj_steps and j not in self._walls \
                    and self._routes[j] is None:
                self._register_adjacent(j, steps[j])
            if j in self._walls:
                return False          # wall member: whole batch serializes
        todo = [j for j in batch if self._routes[j] is None]
        # every member's patches are alive through the SHARED window — the
        # per-step absence rules (first_use later, retired earlier) do not
        # apply to batch-mates
        force = frozenset(nm for j in batch
                          for nm, _ in steps[j].interaction_type)
        if todo:
            # clean-field keys: (bottleneck legal-attach count ASC, clean
            # shortest-route length DESC, index ASC) = most-constrained
            # first; portfolio adds original / shortest / longest orders
            probes = {}
            for j in todo:
                specs = [sp for sp in self._specs_for_step(
                             j, self._eff_orient, force_present=force)
                         if sp.name not in self._lazy.get(j, frozenset())]
                r = self._route_result(specs, steps[j],
                                       bus=self._bus_for(j, steps[j]),
                                       probe=True, step_index=j)
                if r is None:
                    return False      # infeasible even alone: serial path
                probes[j] = (specs, r)
            def tree_len(j):
                return len(probes[j][1].tree or ())
            orders = [sorted(todo, key=lambda j: (
                          -getattr(probes[j][1], 'n_walls', 0),
                          -tree_len(j), j)),
                      list(todo),
                      sorted(todo, key=lambda j: (tree_len(j), j)),
                      sorted(todo, key=lambda j: (-tree_len(j), j))]
            best = None
            for order in orders:
                claimed, trees, blocked, cells = set(), {}, 0, 0
                for j in order:
                    specs, _ = probes[j]
                    r = self._route_result(specs, steps[j],
                                           bus=self._bus_for(j, steps[j]),
                                           probe=True, step_index=j,
                                           blocked_cells=frozenset(claimed))
                    if r is None:
                        blocked += 1
                        continue
                    t = set(map(tuple, r.tree or ()))
                    trees[j] = sorted(t)
                    claimed |= t
                    cells += len(t)
                score = (blocked, cells)
                if best is None or score < best[0]:
                    best = (score, order, trees)
            (blocked, _), order, trees = best
            if blocked:
                return False          # someone starves under every order
            if not hasattr(self, '_planned_tree'):
                self._planned_tree = {}
            claimed = set()
            for j in order:
                self._planned_tree[j] = [list(c) for c in trees[j]]
                n0 = self.system.num_qubits
                try:
                    self._register_ppm(j, steps[j],
                                       exclude=self._lazy.get(j, frozenset()),
                                       blocked_cells=frozenset(claimed),
                                       force_present=force)
                except (BentLayoutError, ValueError):
                    # geometry-stage failures (route status, bent layout)
                    # precede any system mutation: safe to demote.  A
                    # failure AFTER the system grew (add_patch collision
                    # mid-insert) leaves leaked qubits, a half-registered
                    # patch entry and a stale tracker width — no serial
                    # fallback can run on that; fail loud instead
                    # (measured 2026-08-05: leaked 'ppm_6' + 3 qubits made
                    # a later corridor init vstack-crash 3 steps away).
                    if self.system.num_qubits != n0:
                        raise
                    return False      # registered prefix runs serially
                claimed |= set(map(tuple, trees[j]))
        # final disjointness verify over ALL members (portfolio-routed and
        # adjacent-registered alike): shared corridor cells or shared data
        # qubits mean one merged window would couple two PPMs' apparatus —
        # serialize.  blocked_cells should already prevent this; the verify
        # is the executable form of that invariant.
        seen_cells, seen_qubits = set(), set()
        for j in batch:
            r = self._routes[j]
            cells = set(map(tuple, r.tree or ())) if r is not None else set()
            cname = f'ppm_{j}'
            qubits = set()
            if cname in self.system.coupler_patches:
                l2g = self.system.local_to_global_map[cname]
                qubits = {l2g[q] for q in
                          self.system.coupler_patches[cname].data_indices}
            if (cells & seen_cells) or (qubits & seen_qubits):
                return False
            seen_cells |= cells
            seen_qubits |= qubits
        self._apply_ppm_batch(batch)
        for j in batch:
            self._retire_consumed(j, steps[j], owner, measured, allocated)
        self._batch_done.update(batch)
        return True

    def _apply_ppm_batch(self, batch):
        """Pluralized _apply_ppm_step: one shared merge window for every
        member.  The SE emission is system-global, so a single chunk drives
        all active couplers; ordering mirrors the single-step path exactly
        (pre records, activate+init, rounds, deactivate, post records,
        corridor readout)."""
        steps = {j: self.ppm_sequence[j] for j in batch}
        for j in batch:
            self.step_joint_records_pre[j] = self._parity_records(
                self._pauli_vec(steps[j].interaction_type))
        if self.idle_rounds:
            self._standalone_se(self.idle_rounds)
        all_init = {}
        for j in batch:
            cname = f'ppm_{j}'
            self.builder.activate_coupler(cname)
            cp = self.system.coupler_patches[cname]
            l2g = self.system.local_to_global_map[cname]
            bus = self._bus_for(j, steps[j])
            all_init.update({l2g[q]: _CONJ[bus] for q in cp.data_indices})
        if all_init:
            self.builder.initialize(init_dict=all_init,
                                    n=self.system.num_qubits)
        needs_diag = any(
            self._sched.get(j, 'bent') == 'diagonal'
            or (self._routes[j] is not None
                and self._routes[j].layout is not None
                and any(c.get('kf') for c in self._routes[j].layout.checks))
            for j in batch)
        if needs_diag:
            chunk = DiagonalSurfaceCodeExtractionBlock(self.system).circuit
        else:
            domains = {}
            for j in batch:
                lay = self._routes[j].layout if self._routes[j] else None
                if lay is not None:
                    domains.update(dict(getattr(lay, 'domains', {}) or {}))
            _owner = self.system.index_to_owner_map
            for _q in self.system.data_indices:
                _nm = _owner.get(_q)
                if _nm in self._orient:
                    domains.setdefault(
                        tuple(self.system.qubit_coords[_q]),
                        self._orient[_nm])
            chunk = se_round_chunk(self.system, domains=domains)
        self.builder.apply_syndrome_extraction(circuit_chunk=chunk,
                                               rounds=self.rounds)
        for j in batch:
            self.builder.deactivate_coupler(f'ppm_{j}')
        if all_init:
            self.builder.apply_data_readout(final_measurements=dict(all_init),
                                            resolve_absorbed=False)
        # post captures at the LAST moment of the batch window, after every
        # teardown and the combined corridor readout: with all corridor
        # gauges fixed by their readout records, each member's closure is
        # gauge-clean.  Capturing mid-teardown let a closure mix in a
        # batch-mate's not-yet-read corridor gauge (measured 2026-08-05:
        # ghz-mixed lost the deterministic parity of bits 4 and 9, both in
        # batch 2, while the circuit's own observables stayed perfectly
        # deterministic).
        for j in batch:
            self.step_joint_records_post[j] = self._parity_records(
                self._pauli_vec(steps[j].interaction_type))

