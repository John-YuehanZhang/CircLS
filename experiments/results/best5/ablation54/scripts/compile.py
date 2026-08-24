"""Full ablation compile stage: (Clifford 45 + non-Clifford 9) x 9 config
at d=3, all parallel, resumable, provenance-stamped. One circuit + one
metrics record per (program, config)."""
import json, sys, time, contextlib, io, hashlib
import multiprocessing as mp
from pathlib import Path
SP=Path("<workdir>")
sys.path.insert(0,str(SP/"circls_dev"));sys.path.insert(0,str(SP/"circls_dev"/"experiments"))
CIRC=SP/"ablation_full"/"circuits"; OUT=SP/"ablation_full"/"compile.jsonl"
CFGS=["full","no_reduce","no_place","no_fui","no_live","no_sched","no_parallel","no_schedpar","reselect_only"]
NC=["qec_en_n5","teleportation_n3","toffoli_n3","bell_n4","fredkin_n3","adder_n4","simon_n6","multiply_n13","sat_n7"]
SKIP_XL={"bv_n140","bv_n280"}
def targets():
    from benchsuite import suite
    cl=sorted(c.name for c in suite() if c.name not in SKIP_XL)
    t=[]
    for n in cl:
        for c in CFGS: t.append((n,c,"clifford"))
    for n in NC:
        for c in CFGS: t.append((n,c,"nonclifford"))
    return t
def job(task):
    name,cfg,kind=task
    tag=f"{name}__{cfg}"; out=CIRC/f"{tag}.stim"
    if (CIRC/f"{tag}.done").exists(): return {"tag":tag,"status":"cached"}
    from benchsuite import suite, tclass_suite
    from ablation import compile_kwargs
    from circls.pipeline import compile_qasm
    from circls.tools.evaluate import verify
    from circls.metrics import experiment_stats
    from circls.metrics.report import to_dict
    from provenance import provenance
    src = suite() if kind=="clifford" else tclass_suite()
    case={c.name:c for c in src}[name]
    kw=compile_kwargs(cfg,case)
    if kind=="nonclifford": kw["t_as_s"]=True
    t0=time.time()
    rec={"record":"case","name":name,"config":cfg,"kind":kind,
         "prov":provenance(True),"compile_kwargs":{k:(sorted(v) if isinstance(v,(set,frozenset)) else v) for k,v in kw.items()}}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cp=compile_qasm(case.qasm,distance=3,**kw)
    except Exception as e:
        rec.update(status=type(e).__name__,error=str(e)[:200],compile_s=round(time.time()-t0,1))
        return {"tag":tag,"status":rec["status"],"rec":rec}
    t_c=round(time.time()-t0,1)
    with contextlib.redirect_stdout(io.StringIO()):
        r=verify(cp,force_distance=False)
    flat=cp.circuit.flattened()
    out.write_text(str(flat)); (CIRC/f"{tag}.done").write_text("1")
    st=to_dict(experiment_stats(cp.experiment,cp.circuit))
    rec.update(status="OK",compile_s=t_c,verify_ok=bool(r.ok),silent=bool(r.silent),
               logical=r.logical,dets=flat.num_detectors,obs=flat.num_observables,
               sha256=hashlib.sha256(out.read_bytes()).hexdigest()[:16],
               V1_volume_blocks=st.get("V1_volume_blocks"),V2_qubit_rounds=st.get("V2_qubit_rounds"),
               T1_rounds=st.get("T1_rounds"),T4_compile_seconds=t_c,qubits=flat.num_qubits)
    return {"tag":tag,"status":"OK","rec":rec}
if __name__=="__main__":
    ts=[t for t in targets() if not (CIRC/f"{t[0]}__{t[1]}.done").exists()]
    print(f"{len(ts)} compiles pending",flush=True)
    fh=open(OUT,"a")
    with mp.get_context("spawn").Pool(64) as pool:
        for r in pool.imap_unordered(job,ts):
            if "rec" in r: fh.write(json.dumps(r["rec"])+"\n");fh.flush()
            print(r["tag"],r["status"],flush=True)
    fh.close();print("ABLATION COMPILE DONE",flush=True)
