"""Benchmark suite registry.

Composition (every case carries its provenance):
* QASMBench (github.com/pnnl/QASMBench) — the Clifford-compilable subset,
  scanned 2026-08-04 with our front-end as the filter; some files pass only
  under the disclosed normalizations of ``circls.interop.nwqec.qasm_norm``
  (register rename / full terminal measurement).
* Parameterized families — GHZ-n / BV-n / DJ-n / ring graph state, written
  to the standard definitions (as in MQT Bench; generated locally to avoid
  the qiskit dependency).
* Consumable-heavy customs — S-chains whose |Y> gadget ancillas are born,
  used once and retired (the mid-run consumables that exercise liveness
  retirement and, later, dynamic slot reuse); a terminal-form teleport chain.
"""
import dataclasses
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT.parent / "LightStim"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

QASMBENCH_ROOT = Path(os.environ.get(
    "QASMBENCH_ROOT", str(_ROOT.parent / "QASMBench")))

_HDR = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


@dataclasses.dataclass(frozen=True)
class BenchCase:
    name: str
    qasm: str
    source: str            # "QASMBench" | "family" | "custom"
    n_qubits: int
    tags: tuple            # e.g. ("named", "scale") / ("consumable",)
    normalization: str = ""


# ── QASMBench curated subset (scan of 2026-08-04: 23/118 compile) ─────────────
# name -> relative path; entries marked norm=True need qasm_norm rescues.
_QASMBENCH = {
    "deutsch_n2": ("small/deutsch_n2/deutsch_n2.qasm", False),
    "grover_n2": ("small/grover_n2/grover_n2.qasm", False),
    "hs4_n4": ("small/hs4_n4/hs4_n4.qasm", False),
    "iswap_n2": ("small/iswap_n2/iswap_n2.qasm", False),
    "lpn_n5": ("small/lpn_n5/lpn_n5.qasm", False),
    "qrng_n4": ("small/qrng_n4/qrng_n4.qasm", False),
    "cat_state_n4": ("small/cat_state_n4/cat_state_n4.qasm", True),
    "bv_n14": ("medium/bv_n14/bv_n14.qasm", True),
    "bv_n19": ("medium/bv_n19/bv_n19.qasm", True),
    "cat_state_n22": ("medium/cat_state_n22/cat_state_n22.qasm", False),
    "ghz_state_n23": ("medium/ghz_state_n23/ghz_state_n23.qasm", False),
    "bv_n30": ("large/bv_n30/bv_n30.qasm", True),
    "bv_n70": ("large/bv_n70/bv_n70.qasm", True),
    "bv_n140": ("large/bv_n140/bv_n140.qasm", True),
    "bv_n280": ("large/bv_n280/bv_n280.qasm", True),
    "cat_n35": ("large/cat_n35/cat_n35.qasm", False),
    "cat_n65": ("large/cat_n65/cat_n65.qasm", False),
    "cat_n130": ("large/cat_n130/cat_n130.qasm", False),
    "cat_n260": ("large/cat_n260/cat_n260.qasm", False),
    "ghz_n40": ("large/ghz_n40/ghz_n40.qasm", False),
    "ghz_n78": ("large/ghz_n78/ghz_n78.qasm", False),
    "ghz_n127": ("large/ghz_n127/ghz_n127.qasm", False),
    "ghz_state_n255": ("large/ghz_n255/ghz_state_n255.qasm", False),
}


def qasmbench_cases(root: Path = QASMBENCH_ROOT,
                    names: Optional[List[str]] = None) -> List[BenchCase]:
    from circls.interop.nwqec.qasm_norm import normalize_qasm
    import re
    out = []
    for name, (rel, needs_norm) in _QASMBENCH.items():
        if names is not None and name not in names:
            continue
        text = (root / rel).read_text()
        norm_note = ""
        if needs_norm:
            text, notes = normalize_qasm(text)
            norm_note = "+".join(notes)
        nq = int(re.search(r"qreg\s+\w+\[(\d+)\]", text).group(1))
        tag = ("named", "scale") if nq >= 14 else ("named",)
        if name in ("bv_n140", "bv_n280"):
            # kept for stress runs; they time out everywhere, so the paper's
            # 45-program suite is suite() minus the two oversize cases
            tag = tag + ("oversize",)
        out.append(BenchCase(name, text, "QASMBench", nq, tag, norm_note))
    return out


# ── parameterized families (MQT-Bench-standard definitions, local gen) ────────

def ghz(n: int) -> str:
    body = "h q[0];\n" + "".join(f"cx q[{i}],q[{i+1}];\n" for i in range(n - 1))
    return f"{_HDR}qreg q[{n}];\n{body}creg c[{n}];\nmeasure q -> c;\n"


def bv(n: int, secret: Optional[str] = None) -> str:
    """Bernstein-Vazirani with hidden string ``secret`` (data qubits
    0..n-2, phase ancilla n-1); default secret = alternating 101...  """
    m = n - 1
    s = secret if secret is not None else ("10" * m)[:m]
    body = [f"x q[{m}];"] + [f"h q[{i}];" for i in range(n)]
    body += [f"cx q[{i}],q[{m}];" for i in range(m) if s[i] == "1"]
    body += [f"h q[{i}];" for i in range(m)]
    return (f"{_HDR}qreg q[{n}];\n" + "\n".join(body)
            + f"\ncreg c[{n}];\nmeasure q -> c;\n")


def dj(n: int, balanced: bool = True) -> str:
    """Deutsch-Jozsa, balanced oracle = CX fan onto the phase ancilla."""
    m = n - 1
    body = [f"x q[{m}];"] + [f"h q[{i}];" for i in range(n)]
    if balanced:
        body += [f"cx q[{i}],q[{m}];" for i in range(m)]
    body += [f"h q[{i}];" for i in range(m)]
    return (f"{_HDR}qreg q[{n}];\n" + "\n".join(body)
            + f"\ncreg c[{n}];\nmeasure q -> c;\n")


def graph_state_ring(n: int) -> str:
    body = [f"h q[{i}];" for i in range(n)]
    body += [f"cz q[{i}],q[{(i+1) % n}];" for i in range(n)]
    return (f"{_HDR}qreg q[{n}];\n" + "\n".join(body)
            + f"\ncreg c[{n}];\nmeasure q -> c;\n")


# ── consumable-heavy customs ──────────────────────────────────────────────────

def twisted_ghz(n: int) -> str:
    """GHZ chain with an S·H twist on every OTHER link: the twisted links
    leave Y letters in the terminal operators (one |Y> gadget ancilla each
    — born, consumed by one joint PPM, retired), while the untwisted links
    keep deterministic parity bits, so the circuit has real observables for
    the LER story (the all-link twist variant compiles to ZERO observables
    — measured 2026-08-04, that design is dead)."""
    body = ["h q[0];"]
    for i in range(n - 1):
        body.append(f"cx q[{i}],q[{i+1}];")
        if i % 2 == 0:
            body += [f"s q[{i+1}];", f"h q[{i+1}];"]
    return (f"{_HDR}qreg q[{n}];\n" + "\n".join(body)
            + f"\ncreg c[{n}];\nmeasure q -> c;\n")


def steane_encode() -> str:
    """|0̄> preparation for the [[7,1,3]] Steane code, synthesized from the
    stabilizer group with ``stim.Tableau.from_stabilizers`` (correct by
    construction; verified: all 6 generators + Z̄ stabilize the output).
    tqec's example gallery ships the same computation (steane_encoding), so
    track-① circuit-quality comparisons can use an opponent-chosen program."""
    import stim
    gens = ["___XXXX", "_XX__XX", "X_X_X_X",
            "___ZZZZ", "_ZZ__ZZ", "Z_Z_Z_Z", "ZZZZZZZ"]
    circ = stim.Tableau.from_stabilizers(
        [stim.PauliString(g) for g in gens]).to_circuit("elimination")
    body = []
    for inst in circ:
        ts = [t.qubit_value for t in inst.targets_copy()]
        if inst.name == "H":
            body += [f"h q[{q}];" for q in ts]
        elif inst.name == "CX":
            body += [f"cx q[{a}],q[{b}];" for a, b in zip(ts[::2], ts[1::2])]
        elif inst.name != "I":
            raise ValueError(f"unexpected gate {inst.name} from stim synthesis")
    return (f"{_HDR}qreg q[7];\n" + "\n".join(body)
            + "\ncreg c[7];\nmeasure q -> c;\n")


def bbpssw_chain(k: int) -> str:
    """Terminal-form BBPSSW distillation chain: a target Bell pair (q0,q1)
    distilled through ``k`` sacrificial Bell pairs via bilateral CNOTs; the
    sacrificial pairs' check measurements are deferred to the end (they
    commute there — nothing acts on them after their round), so
    post-selection becomes classical post-processing on the output bits
    (disclosed).  Each sacrificial pair's joint involvement ends at its own
    round -> retires early under liveness: named-protocol consumables."""
    n = 2 + 2 * k
    body = ["h q[0];", "cx q[0],q[1];"]
    for j in range(k):
        a, b = 2 + 2 * j, 3 + 2 * j
        body += [f"h q[{a}];", f"cx q[{a}],q[{b}];",
                 f"cx q[0],q[{a}];", f"cx q[1],q[{b}];"]
    return (f"{_HDR}qreg q[{n}];\n" + "\n".join(body)
            + f"\ncreg c[{n}];\nmeasure q -> c;\n")


def random_clifford(n: int, depth: int, seed: int) -> str:
    """Reproducible random Clifford circuit (whitelist gate sampling, NOT
    uniform over the Clifford group — same convention as TopoLS's Random
    Clifford instances).  Roles per benchmark_set.md: LaSsynth
    gap-to-optimal small instances + pipeline stress testing; NOT in the
    main named table."""
    import random
    rng = random.Random(seed)
    one_q = ["h", "s", "sdg", "sx", "x", "z"]
    body = []
    for _ in range(depth):
        if n >= 2 and rng.random() < 0.5:
            a, b = rng.sample(range(n), 2)
            body.append(f"cx q[{a}],q[{b}];")
        else:
            body.append(f"{rng.choice(one_q)} q[{rng.randrange(n)}];")
    return (f"{_HDR}qreg q[{n}];\n" + "\n".join(body)
            + f"\ncreg c[{n}];\nmeasure q -> c;\n")


def teleport_chain(hops: int) -> str:
    """Terminal-form teleportation relay over ``hops`` Bell pairs
    (2*hops+1 qubits); all measurements terminal, relays are consumed."""
    n = 2 * hops + 1
    body = []
    for h in range(hops):
        a, b = 2 * h + 1, 2 * h + 2
        body += [f"h q[{a}];", f"cx q[{a}],q[{b}];"]
    for h in range(hops):
        src, a = 2 * h, 2 * h + 1
        body += [f"cx q[{src}],q[{a}];", f"h q[{src}];"]
    return (f"{_HDR}qreg q[{n}];\n" + "\n".join(body)
            + f"\ncreg c[{n}];\nmeasure q -> c;\n")


def _topols_fill_cases() -> List[BenchCase]:
    """Mixed-basis fills from TopoLS's OWN benchmark generator
    (fill_ports_for_minimal_simulation, frozen in the A1 v2 manifest):
    terminal X/Z bases interleave along the chain, so the reduced terminal
    set keeps small joint steps scattered over disjoint patch pairs — the
    step structure where shared merge windows apply (the pure-basis family
    cases either fold to weight-1 terminals or are hub chains).  Added
    2026-08-06 with parallel step execution; roster still TopoLS-derived."""
    man_path = (Path(__file__).resolve().parent / "results"
                / "topols_circuits" / "manifest.json")
    if not man_path.exists():
        return []
    entries = json.loads(man_path.read_text())["entries"]
    out = []
    for key in ("ghz_16_k1_f1",):
        if key not in entries:
            continue
        from compare_topols import matching_qasm
        name = key.rsplit("_k", 1)[0]
        qasm = matching_qasm(name, entries[key]["fill"])
        n = int(name.split("_")[1])
        out.append(BenchCase(f"{name}_mixed", qasm, "topols_fill", n,
                             ("named", "mixed")))
    return out


# ── QASMBench T-class subset for the S-state proxy (t_as_s) ──────────────────
# Census of 2026-08-19: 131 programs = 32 Clifford / 25 T-class (every
# non-Clifford gate is t/tdg/ccx/cswap or a pi/4-multiple rotation, so the
# T->S substitution is exact at the gate level) / 74 arbitrary-angle.
# 19 of the 25 run; the six exclusions are structural, not curation:
#   multiplier_n350, multiplier_n400, square_root_n45, square_root_n60
#     (55k-428k T gates: proxy compilation is fine but the circuits are
#      beyond any sampling budget), shor_n5 (classically conditioned gates:
#     nwqec has no conditional-statement support), square_root_n18
#     (mid-circuit reset: ancilla recycling, outside the terminal-
#      measurement contract).
# "core" = LER-sampled at d3/d5; "scale" = compile/metrics only.
_TCLASS = {
    # name -> (relative path, core?)
    "teleportation_n3": ("small/teleportation_n3/teleportation_n3.qasm", True),
    "qec_en_n5": ("small/qec_en_n5/qec_en_n5.qasm", True),
    "toffoli_n3": ("small/toffoli_n3/toffoli_n3.qasm", True),
    "fredkin_n3": ("small/fredkin_n3/fredkin_n3.qasm", True),
    "bell_n4": ("small/bell_n4/bell_n4.qasm", True),
    "adder_n4": ("small/adder_n4/adder_n4.qasm", True),
    "simon_n6": ("small/simon_n6/simon_n6.qasm", True),
    "multiply_n13": ("medium/multiply_n13/multiply_n13.qasm", True),
    "sat_n7": ("small/sat_n7/sat_n7.qasm", True),
    "seca_n11": ("medium/seca_n11/seca_n11.qasm", False),
    "qram_n20": ("medium/qram_n20/qram_n20.qasm", False),
    "adder_n28": ("large/adder_n28/adder_n28.qasm", False),
    "multiplier_n15": ("medium/multiplier_n15/multiplier_n15.qasm", False),
    "sat_n11": ("medium/sat_n11/sat_n11.qasm", False),
    "adder_n64": ("large/adder_n64/adder_n64.qasm", False),
    "adder_n118": ("large/adder_n118/adder_n118.qasm", False),
    "adder_n433": ("large/adder_n433/adder_n433.qasm", False),
    "multiplier_n45": ("large/multiplier_n45/multiplier_n45.qasm", False),
    "multiplier_n75": ("large/multiplier_n75/multiplier_n75.qasm", False),
}


def tclass_suite(root: Path = QASMBENCH_ROOT) -> List[BenchCase]:
    """The 19 T-class QASMBench programs, normalized where needed (the
    applied rewrites land in ``BenchCase.normalization``), for
    ``compile_qasm(..., t_as_s=True)``.  Separate from ``suite()``: the
    Clifford suite stays exactly the paper's 45 programs."""
    import re as _re
    from circls.interop.nwqec.frontend import load_clifford_t_as_s
    from circls.interop.nwqec.qasm_norm import normalize_qasm
    out = []
    for name, (rel, core) in _TCLASS.items():
        text = (root / rel).read_text()
        norm_note = ""
        try:
            load_clifford_t_as_s(text)
        except (ValueError, RuntimeError):
            text, notes = normalize_qasm(text)
            norm_note = "+".join(notes)
        nq = int(_re.search(r"qreg\s+\w+\[(\d+)\]", text).group(1))
        tags = ("tclass",) if core else ("tclass", "scale")
        out.append(BenchCase(name, text, "QASMBench", nq, tags, norm_note))
    return out


def suite(include_qasmbench: bool = True,
          family_sizes: List[int] = (8, 16, 32, 64),
          consumable_sizes: List[int] = (4, 8),
          ) -> List[BenchCase]:
    cases: List[BenchCase] = []
    if include_qasmbench and QASMBENCH_ROOT.exists():
        cases += qasmbench_cases()
    for n in family_sizes:
        cases.append(BenchCase(f"ghz_{n}", ghz(n), "family", n, ("named", "scale")))
        cases.append(BenchCase(f"bv_{n}", bv(n), "family", n, ("named", "scale")))
        cases.append(BenchCase(f"dj_{n}", dj(n), "family", n, ("named", "scale")))
        cases.append(BenchCase(f"graphstate_{n}", graph_state_ring(n), "family",
                               n, ("named", "scale")))
    cases.append(BenchCase("steane_encode", steane_encode(), "custom", 7,
                           ("named",)))
    for mc in _topols_fill_cases():
        cases.append(mc)
    for n in consumable_sizes:
        cases.append(BenchCase(f"twistedghz_{n}", twisted_ghz(n), "custom", n,
                               ("consumable",)))
        cases.append(BenchCase(f"teleport_{n}", teleport_chain(n), "custom",
                               2 * n + 1, ("consumable",)))
        cases.append(BenchCase(f"bbpssw_{n}", bbpssw_chain(n), "custom",
                               2 + 2 * n, ("consumable",)))
    return cases
