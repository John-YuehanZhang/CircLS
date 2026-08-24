"""Stage 2a (daemon): stock PyMatching + two-pass on every archived
circuit (d <= 9), p = 5e-4 (and 1e-3 for the tclass table circuits).
One result file per (circuit, p); resumable; exits when stage 1 is done
and every circuit is processed."""
import json, sys, time, os
import multiprocessing as mp
from pathlib import Path
SP = Path("<workdir>")
REPO = SP / "circls_dev"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments"))
CIRC, PTS = SP / "resample" / "circuits", SP / "resample" / "points"
SEED0 = 800000

def caps(tag):
    if "_panel_" in tag:
        d = int(tag.rsplit("_d", 1)[1])
        return (400 if "_full_" in tag and d in (5, 7, 9) else 100), (4_000_000 if d == 9 else 1_000_000)
    return 300, 3_000_000

def job(args):
    tag, p = args
    out = PTS / f"{tag}_p{p:g}_pm.json"
    if out.exists():
        return tag, p, "cached"
    import stim, pymatching as pm
    from noise_inject import inject_uniform_noise
    from compare_tqec import adaptive_ler
    from twopass_matching import TwoPassMatching
    circ = stim.Circuit((CIRC / f"{tag}.stim").read_text())
    if circ.num_observables == 0:
        out.write_text(json.dumps({"tag": tag, "p": p, "dagger": True})); return tag, p, "dagger"
    noisy = inject_uniform_noise(circ, p)
    dem = noisy.detector_error_model(decompose_errors=True)
    target, cap = caps(tag)
    seed = SEED0 + (abs(hash(tag)) % 10000) * 7 + int(p * 1e4)
    res = {"tag": tag, "p": p, "seed": seed, "target": target, "cap": cap}
    t0 = time.time()
    m0 = pm.Matching.from_detector_error_model(dem)
    s, j, _ = adaptive_ler(noisy, m0, seed, target, cap)
    res["pm"] = {"shots": s, "errors": j, "ler": j / s, "s": round(time.time() - t0, 1)}
    t0 = time.time()
    tp = TwoPassMatching(dem)
    s, j, _ = adaptive_ler(noisy, tp, seed, target, cap)
    res["pm2"] = {"shots": s, "errors": j, "ler": j / s, "conflicted_edges": tp.num_conflicted_edges,
                  "s": round(time.time() - t0, 1)}
    out.write_text(json.dumps(res)); return tag, p, f"pm={res['pm']['ler']:.4g} pm2={res['pm2']['ler']:.4g}"

def pending():
    todo = []
    for f in sorted(CIRC.glob("*.stim")):
        tag = f.stem
        if tag.endswith("_d11"): continue
        ps = [5e-4, 1e-3] if "_tclass_" in tag else [5e-4]
        for p in ps:
            if not (PTS / f"{tag}_p{p:g}_pm.json").exists(): todo.append((tag, p))
    return todo

if __name__ == "__main__":
    done_marker = SP / "resample" / "compile_stage.log"
    with mp.get_context("spawn").Pool(32) as pool:
        while True:
            todo = pending()
            if todo:
                for r in pool.imap_unordered(job, todo):
                    print(*r, flush=True)
            stage1_done = done_marker.exists() and "COMPILE STAGE DONE" in done_marker.read_text()
            if stage1_done and not pending():
                break
            time.sleep(60)
    print("PM SAMPLER DONE", flush=True)
