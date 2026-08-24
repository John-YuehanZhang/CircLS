"""one MWPF point (its own 32-way pool): argv = tag"""
import json, sys, time, os
from pathlib import Path
SP = Path("<workdir>")
REPO = SP / "circls_dev"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments"))

def _main():
    tag = sys.argv[1]
    out = SP / "resample" / "points" / f"{tag}_p0.0005_mwpf.json"
    if out.exists(): return
    import stim
    from noise_inject import inject_uniform_noise
    from formula_deviation import _mwpf_sample_ler
    circ = stim.Circuit((SP / "resample" / "circuits" / f"{tag}.stim").read_text())
    if circ.num_observables == 0:
        out.write_text(json.dumps({"tag": tag, "p": 5e-4, "dagger": True})); return
    noisy = inject_uniform_noise(circ, 5e-4)
    seed = 810000 + abs(hash(tag)) % 10000
    t0 = time.time()
    s, j, _ = _mwpf_sample_ler(noisy, seed, 300, 400_000, workers=16)
    out.write_text(json.dumps({"tag": tag, "p": 5e-4, "seed": seed, "decoder": "mwpf",
                               "shots": s, "errors": j, "ler": j / s, "s": round(time.time() - t0, 1)}))
    print(tag, f"mwpf ler={j/s:.4g} ({j}/{s}) {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    _main()
