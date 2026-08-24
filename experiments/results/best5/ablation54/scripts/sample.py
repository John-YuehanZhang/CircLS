"""Full ablation LER stage (daemon). Deterministic seeds (sha256, not
process-random hash), obs==0 -> dagger, obs>63 -> excluded, else stock
PyMatching (de-nest fallback); non-graphlike leftovers get a needs_mwpf
placeholder that the MWPF phase (main proc, own pool) fills. Resumable."""
import json, sys, time, hashlib
import multiprocessing as mp
from pathlib import Path
SP=Path("<workdir>")
sys.path.insert(0,str(SP/"circls_dev"));sys.path.insert(0,str(SP/"circls_dev"/"experiments"))
sys.path.insert(0,str(SP/"xval_detectors"/"dig_tclass_rebasis"))
CIRC,PTS=SP/"ablation_full"/"circuits",SP/"ablation_full"/"points"
def seed_of(tag,p):
    return int(hashlib.sha256(f"{tag}|{p:g}".encode()).hexdigest()[:8],16)
def job(args):
    tag,p=args
    out=PTS/f"{tag}__p{p:g}.json"
    if out.exists():
        r=json.loads(out.read_text())
        if not r.get("needs_mwpf"): return tag,p,"cached"
    import stim,pymatching as pm
    from noise_inject import inject_uniform_noise
    from compare_tqec import adaptive_ler
    circ=stim.Circuit((CIRC/f"{tag}.stim").read_text())
    if circ.num_observables==0:
        out.write_text(json.dumps({"tag":tag,"p":p,"dagger":True}));return tag,p,"dagger"
    if circ.num_observables>63:
        out.write_text(json.dumps({"tag":tag,"p":p,"ler_excluded":"observables>63","n_obs":circ.num_observables}));return tag,p,"excluded>63"
    seed=seed_of(tag,p); noisy=inject_uniform_noise(circ,p); dec="mwpm"
    try:
        m=pm.Matching.from_detector_error_model(noisy.detector_error_model(decompose_errors=True))
    except ValueError:
        try:
            from rebasis import parse,denest,rebuild
            recs,_=parse(circ); nr,_=denest(recs); c2=rebuild(circ,nr)
            noisy=inject_uniform_noise(c2,p)
            m=pm.Matching.from_detector_error_model(noisy.detector_error_model(decompose_errors=True))
            dec="mwpm+denest"
        except Exception:
            out.write_text(json.dumps({"tag":tag,"p":p,"needs_mwpf":True,"seed":seed}))
            return tag,p,"needs_mwpf(placeholder)"
    s,j,_=adaptive_ler(noisy,m,seed,100,1_000_000)
    out.write_text(json.dumps({"tag":tag,"p":p,"decoder":dec,"seed":seed,"shots":s,"errors":j,"ler":j/s if s else None}))
    return tag,p,f"{dec} {j}/{s}"
def pending():
    todo=[]
    for f in CIRC.glob("*.done"):
        tag=f.stem
        if not (CIRC/f"{tag}.stim").exists(): continue
        for p in (1e-3,5e-4):
            o=PTS/f"{tag}__p{p:g}.json"
            if not o.exists(): todo.append((tag,p))
    return todo
def mwpf_leftovers():
    import stim
    from noise_inject import inject_uniform_noise
    from formula_deviation import _mwpf_sample_ler
    for f in sorted(PTS.glob("*.json")):
        r=json.loads(f.read_text())
        if not r.get("needs_mwpf"): continue
        tag,p=r["tag"],r["p"]; seed=r["seed"]
        noisy=inject_uniform_noise(stim.Circuit((CIRC/f"{tag}.stim").read_text()),p)
        sh,j,_=_mwpf_sample_ler(noisy,seed,100,1_000_000,workers=16)
        f.write_text(json.dumps({"tag":tag,"p":p,"decoder":"mwpf","seed":seed,"shots":sh,"errors":j,"ler":j/sh if sh else None}))
        print("mwpf",tag,p,f"{j}/{sh}",flush=True)
if __name__=="__main__":
    marker=SP/"ablation_full"/"compile.log"
    with mp.get_context("spawn").Pool(48) as pool:
        while True:
            todo=pending()
            if todo:
                for r in pool.imap_unordered(job,todo): print(*r,flush=True)
            done=marker.exists() and "ABLATION COMPILE DONE" in marker.read_text()
            if done and not pending(): break
            time.sleep(45)
    mwpf_leftovers()
    print("ABLATION SAMPLE DONE",flush=True)
