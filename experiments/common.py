"""Shared experiment utilities: parameter tables, deterministic sampling,
Wilson intervals, result serialization, plotting.

Reproducibility conventions
---------------------------
* Every data point comes from ``run_point``: shots are split evenly across
  shards, shard i runs a **single-worker** pipeline with seed ``seed + i``
  (LightStim ``base_seed``, bit-for-bit deterministic), then the counts are
  summed -- parallel yet bit-reproducible (same shots/seed/shard count
  implies the same error count).
* The result JSON records all counts, the seed, the shard count, the git
  HEAD of both repositories, and library versions.
* Confidence intervals are always Wilson; CIs for ratio plots use monotone
  propagation (numerator and denominator each take an endpoint).
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT.parent / "LightStim"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lightstim.noise.config import NoiseConfig  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

# Physical error rate <-> shot budget: per user ruling, shots = p * 1e10.
P_SHOTS = [(p, int(p * 1e10)) for p in (1e-4, 5e-4, 1e-3, 1e-2)]
P_SHOTS_SMOKE = [(1e-3, 100_000), (1e-2, 50_000)]

D_LIST = [3, 5]
D_LIST_SMOKE = [3]
DEFAULT_WORKERS = 64          # half of the 128-core cap, leaving decode headroom
DECODER = "pymatching"


def uniform_noise(p: float) -> NoiseConfig:
    return NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)


def assert_valid_scenario(circuit, name: str, shots: int = 256):
    """Scenario admission check (user ruling 2026-08-01): the observable
    count must be >=1, and at p=0 the detectors must stay quiet and the
    observables deterministic -- this rules out fake "observable vanished"
    experiments."""
    n = circuit.num_observables
    assert n >= 1, f"{name}: circuit has 0 observables -- invalid scenario"
    det, obs = circuit.compile_detector_sampler(seed=0).sample(
        shots, separate_observables=True)
    assert not det.any(), f"{name}: detector fires at p=0"
    assert not obs.any(), f"{name}: observable not deterministic at p=0"


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson interval (lo, hi); returns a conservative interval for k=0/n=0."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def _shard(args):
    (circuit_text, quota, seed, batch, target_obs) = args
    import stim
    from lightstim.simulation.decoder_backend import DecoderConfig
    from lightstim.simulation.decoder_backend.pipeline import SimulationPipeline

    circuit = stim.Circuit(circuit_text)
    pipe = SimulationPipeline(
        decoder_config=DecoderConfig(DECODER, backend="cpu"),
        max_shots=quota, max_errors=2 ** 62, batch_size=batch,
        num_workers=1, base_seed=seed,
        target_observable_indices=target_obs, print_progress=False)
    st = pipe.run(circuit)
    return st.shots, st.errors


MIN_ERRORS = 100_000          # minimum failures per point (user ruling 2026-08-01)
MAX_TOTAL_SHOTS = 4_000_000_000    # per-point shot cap (keeps low-LER points finite)


def min_errors_for(p: float) -> int:
    """Tier-B layered precision (user ruling 2026-08-01):
    p >= 1e-3 -> 1e5 (+-0.6%); p >= 5e-4 -> 1e4 (+-2%); lower -> 1e3 (+-6%).
    A low-LER point that cannot reach the quota within MAX_TOTAL_SHOTS is
    marked capped=True and scored on the failures actually observed."""
    if p >= 0.999e-3:
        return 100_000
    if p >= 0.999 * 5e-4:
        return 10_000
    return 1_000


def _round(circuit_text, shots, seed, workers, target_obs):
    workers = max(1, min(workers, shots // 10_000 or 1))
    quotas = [shots // workers] * workers
    for i in range(shots % workers):
        quotas[i] += 1
    batch = min(65_536, max(4_096, quotas[0] // 8 or 1))
    jobs = [(circuit_text, q, seed + i, batch, target_obs)
            for i, q in enumerate(quotas) if q > 0]
    if len(jobs) == 1:
        parts = [_shard(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(_shard, jobs))
    return (sum(s for s, _ in parts), sum(e for _, e in parts), len(jobs))


def run_point(circuit, shots: int, seed: int, workers: int = DEFAULT_WORKERS,
              target_observable_indices=None,
              min_errors: int = None, max_total_shots: int = None) -> dict:
    """Sample and decode one data point.

    After the baseline shots finish, if the failure count is < min_errors,
    extra shots are added in a deterministic batch ladder: the next batch
    size = 1.25 x (failures still missing) / (currently observed LER). It
    depends only on already observed counts, and the seed chain advances
    shard by shard, so a rerun with the same parameters is bit-identical.
    This repeats until the quota is met or max_total_shots is reached.
    A point that hits the cap carries capped=True in the result, and its
    interval is computed from the k actually observed."""
    if min_errors is None:
        min_errors = MIN_ERRORS
    if max_total_shots is None:
        max_total_shots = MAX_TOTAL_SHOTS
    text = str(circuit)
    t0 = time.perf_counter()
    n = k = used = 0
    rounds = 0
    while True:
        want = shots if rounds == 0 else min(
            max_total_shots - n,
            max(int(1.25 * (min_errors - k) / max(k / n, 1.0 / n)),
                10_000))
        if want <= 0:
            break
        rn, rk, rs = _round(text, want, seed + used, workers,
                            target_observable_indices)
        n += rn
        k += rk
        used += rs
        rounds += 1
        if k >= min_errors or n >= max_total_shots:
            break
    lo, hi = wilson(k, n)
    return dict(shots=n, errors=k, ler=(k / n if n else 0.0),
                ler_lo=lo, ler_hi=hi, seed=seed, shards=used,
                rounds=rounds, capped=(k < min_errors),
                min_errors=min_errors,
                seconds=round(time.perf_counter() - t0, 2),
                decoder=DECODER,
                target_observables=target_observable_indices)


def _git_head(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def metadata() -> dict:
    import stim
    return dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                python=sys.version.split()[0],
                stim=stim.__version__,
                circls_head=_git_head(_ROOT),
                lightstim_head=_git_head(_ROOT.parent / "LightStim"),
                pid=os.getpid())


def save_json(name: str, payload: dict) -> Path:
    out = RESULTS / f"{name}.json"
    payload = dict(payload)
    payload["_meta"] = metadata()
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    return out


def compose_product(components):
    """1 - prod(1 - Pi), together with the monotonically propagated (lo, hi)."""
    pred, lo, hi = 1.0, 1.0, 1.0
    for c in components:
        pred *= (1 - c["ler"])
        lo *= (1 - c["ler_hi"])   # bad end of each component -> upper prediction
        hi *= (1 - c["ler_lo"])
    return dict(pred=1 - pred, pred_lo=1 - hi, pred_hi=1 - lo)


def ratio_ci(meas: dict, pred: dict):
    """measured/predicted plus a conservative interval (each of numerator and
    denominator takes an endpoint)."""
    if pred["pred"] <= 0 or meas["ler"] <= 0:
        return dict(ratio=float("nan"), lo=float("nan"), hi=float("nan"))
    return dict(ratio=meas["ler"] / pred["pred"],
                lo=meas["ler_lo"] / max(pred["pred_hi"], 1e-300),
                hi=meas["ler_hi"] / max(pred["pred_lo"], 1e-300))


def new_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_figure(fig, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    return RESULTS / f"{name}.png"
