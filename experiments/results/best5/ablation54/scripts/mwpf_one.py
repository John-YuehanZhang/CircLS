"""One MWPF ablation point, as its own process (non-daemon so _mwpf_sample_ler
can spawn its worker pool).  Reuses the needs_mwpf placeholder's stored seed
for reproducibility.  Usage: python mwpf_one.py <tag> <p> <workers>"""
import json, sys, hashlib
from pathlib import Path
SP = Path("<workdir>/claude-1041/"
          "<host>/2f1c189c-a42c-406b-a2ef-c01703612f0b/scratchpad")
sys.path.insert(0, str(SP / "circls_dev"))
sys.path.insert(0, str(SP / "circls_dev" / "experiments"))
sys.path.insert(0, str(SP / "xval_detectors" / "dig_tclass_rebasis"))
CIRC = SP / "ablation_full" / "circuits"
PTS = SP / "ablation_full" / "points"


def main():
    tag, p = sys.argv[1], float(sys.argv[2])
    W = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    out = PTS / f"{tag}__p{p:g}.json"
    seed = None
    if out.exists():
        r = json.loads(out.read_text())
        if not r.get("needs_mwpf"):
            print(tag, p, "already-filled", flush=True)
            return
        seed = r.get("seed")
    if seed is None:
        seed = int(hashlib.sha256(f"{tag}|{p:g}".encode()).hexdigest()[:8], 16)
    import stim
    from noise_inject import inject_uniform_noise
    from formula_deviation import _mwpf_sample_ler
    circ = stim.Circuit((CIRC / f"{tag}.stim").read_text())
    noisy = inject_uniform_noise(circ, p)
    sh, j, _ = _mwpf_sample_ler(noisy, seed, 100, 1_000_000, workers=W)
    out.write_text(json.dumps({"tag": tag, "p": p, "decoder": "mwpf",
                               "seed": seed, "shots": sh, "errors": j,
                               "ler": j / sh if sh else None}))
    print(tag, p, f"mwpf {j}/{sh}", flush=True)


if __name__ == "__main__":
    main()
