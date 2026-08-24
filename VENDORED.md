# Vendored dependencies

## lightstim/ (Apache-2.0)

CircLS is **built on LightStim** and ships a **modified copy** of it.
Upstream: <https://github.com/QuTone/LightStim>.

The copy under `lightstim/` started from upstream and carries our
changes on top (made for this project): the multi-patch PPM stack
that CircLS grew out of, detector/observable annotation fixes, and
the protocol modules the compiler drives.  This copy is the source
of truth for CircLS; changes we consider general are offered
upstream as pull requests.

Layout (everything LightStim-related lives under this one folder):

- `lightstim/` — the importable package (`import lightstim`)
- `lightstim/tests/` — its test suite, collected with the main suite
- `lightstim/benchmarks/` — its benchmark and example scripts
- `lightstim/LICENSE` — Apache 2.0, kept verbatim per its terms

CircLS itself is released under the same Apache-2.0 license (top-level `LICENSE`).

## Installation

`pip install -e .` at the repo root installs both packages; `import
circls` and `import lightstim` then resolve to this repo from any
working directory.

Note the deliberate consequence: installing CircLS provides a
top-level package named `lightstim` — this MODIFIED copy — which
shadows any separately installed upstream LightStim in the same
environment.  Do not install both in one environment.


## Third-party notices

- `lightstim/qec_code/surface_code/rotated/y_transition_round.py` and
  `y_boundary_patch.py` port code from Craig Gidney's artifact
  *Inplace Access to the Surface Code Y Basis* (zenodo record 7487893),
  licensed **CC-BY-4.0**; both files carry their attribution in the
  module docstrings and were adapted to LightStim's data structures.
- `lightstim/simulation/decoder_backend/pcm.py` derives from
  `gongaa/SlidingWindowDecoder`, licensed **MIT**:

  MIT License

  Copyright (c) 2024 Anqi Gong

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.

- `tests/fixtures/tqec_cnot_k1.stim` and `tqec_memory_k1.stim` were
  generated with the `tqec` tool (v0.2.0) from its gallery examples and
  are checked in as regression oracles.
- `experiments/results/baseline_stim_circuits.tar.zst` bundles baseline
  circuits produced by third-party toolchains for the paper's
  comparison: 79 by TopoLS (Apache-2.0, <https://github.com/tqec/TopoLS>)
  plus tqec (Apache-2.0, v0.2.0; its NOTICE reads "TQEC / Copyright (c)
  2025 TQEC Community"), and 14 from tqec's own example gallery.
- `lightstim/tests/color_code/data/midout_color_code_*.stim` are
  reference circuits generated with Chromobius/Clorco (Apache-2.0,
  arXiv:2312.08813), as their local README states.
- `circls/interop/ir/yfree.py` can OPTIONALLY drive the
  lattice-surgery-compiler (`lsqecc`, LGPL-2.1-or-later) for its
  Y-elimination path.  lsqecc is never bundled or imported at package
  load; it is located via the `LSQECC_SRC` environment variable (or a
  sibling checkout) only when that path is used.

Files modified relative to upstream are not individually marked; this
statement records, per Apache-2.0 section 4(b), that the copy as a
whole contains CircLS's changes on top of the upstream LightStim
package (<https://github.com/QuTone/LightStim>).
