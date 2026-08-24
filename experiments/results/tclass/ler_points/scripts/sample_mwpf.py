"""Stage 2b (daemon): MWPF on the tclass table circuits, p = 5e-4,
sequential points, each on its own 48-way pool; resumable."""
import json, sys, time
from pathlib import Path
SP = Path("<workdir>")
REPO = SP / "circls_dev"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments"))
CIRC, PTS = SP / "resample" / "circuits", SP / "resample" / "points"

def _main():
    import stim
    from noise_inject import inject_uniform_noise
    from formula_deviation import _mwpf_sample_ler
    done_marker = SP / "resample" / "compile_stage.log"
    while True:
        todo = [f for f in sorted(CIRC.glob("*_tclass_*.stim"))
                if not (PTS / f"{f.stem}_p0.0005_mwpf.json").exists()]
        for f in todo:
            tag = f.stem
            circ = stim.Circuit(f.read_text())
            out = PTS / f"{tag}_p0.0005_mwpf.json"
            if circ.num_observables == 0:
                out.write_text(json.dumps({"tag": tag, "p": 5e-4, "dagger": True})); continue
            noisy = inject_uniform_noise(circ, 5e-4)
            seed = 810000 + abs(hash(tag)) % 10000
            t0 = time.time()
            s, j, _ = _mwpf_sample_ler(noisy, seed, 300, 400_000, workers=48)
            out.write_text(json.dumps({"tag": tag, "p": 5e-4, "seed": seed, "decoder": "mwpf",
                                       "shots": s, "errors": j, "ler": j / s,
                                       "s": round(time.time() - t0, 1)}))
            print(tag, f"mwpf ler={j/s:.4g} ({j}/{s}) {time.time()-t0:.0f}s", flush=True)
        stage1_done = done_marker.exists() and "COMPILE STAGE DONE" in done_marker.read_text()
        sat_done = (CIRC / "sat_n7_tclass_full_d5.stim").exists()
        if stage1_done and sat_done and not [f for f in CIRC.glob("*_tclass_*.stim")
                                              if not (PTS / f"{f.stem}_p0.0005_mwpf.json").exists()]:
            break
        time.sleep(120)
    print("MWPF SAMPLER DONE", flush=True)

if __name__ == "__main__":
    _main()
