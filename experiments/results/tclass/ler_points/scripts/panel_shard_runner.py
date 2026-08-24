"""one shard of a panel d11 point: (tag, j) -> points/shards/<tag>_s<j>.json"""
import json, sys, time
from pathlib import Path
SP = Path("<workdir>")
REPO = SP / "circls_dev"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments"))

def _main():
    tag, j = sys.argv[1], int(sys.argv[2])
    out = SP / "resample" / "points" / "shards" / f"{tag}_s{j:02d}.json"
    out.parent.mkdir(exist_ok=True)
    if out.exists(): return
    import numpy as np, stim, pymatching as pm
    from noise_inject import inject_uniform_noise
    noisy = inject_uniform_noise(stim.Circuit((SP / "resample" / "circuits" / f"{tag}.stim").read_text()), 5e-4)
    m = pm.Matching.from_detector_error_model(noisy.detector_error_model(decompose_errors=True))
    seed = 820000 + (abs(hash(tag)) % 1000) * 100 + j
    sampler = noisy.compile_detector_sampler(seed=seed)
    PER, CHUNK = 600_000, 50_000
    err, done, t0 = 0, 0, time.time()
    while done < PER:
        n = min(CHUNK, PER - done)
        det, obs = sampler.sample(n, separate_observables=True)
        err += int(np.any(m.decode_batch(det) != obs, axis=1).sum()); done += n
    out.write_text(json.dumps({"tag": tag, "j": j, "seed": seed, "shots": done, "errors": err,
                               "s": round(time.time() - t0, 1)}))

if __name__ == "__main__":
    _main()
