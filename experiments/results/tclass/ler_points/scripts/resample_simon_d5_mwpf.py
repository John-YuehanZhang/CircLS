import json, sys, time
from pathlib import Path
SP=Path("<workdir>")
sys.path.insert(0,str(SP/"circls_dev")); sys.path.insert(0,str(SP/"circls_dev"/"experiments"))

def main():
    import stim
    from noise_inject import inject_uniform_noise
    from formula_deviation import _mwpf_sample_ler
    SEED=20260823
    circ=stim.Circuit((SP/"resample"/"circuits"/"simon_n6_tclass_full_d5.stim").read_text())
    noisy=inject_uniform_noise(circ,5e-4)
    t0=time.time()
    sh,j,_=_mwpf_sample_ler(noisy,SEED,300,3_000_000,workers=16)
    dt=time.time()-t0
    out={"tag":"simon_n6_tclass_full_d5","p":5e-4,"decoder":"mwpf","seed":SEED,
         "shots":sh,"errors":j,"ler":j/sh if sh else None,"wall_seconds":round(dt,1)}
    (SP/"resample"/"simon_d5_mwpf_resample.json").write_text(json.dumps(out))
    print(f"SIMON RESAMPLE DONE: {j}/{sh} = {j/sh:.4f}  in {dt:.0f}s ({dt/60:.1f} min), seed {SEED}",flush=True)

if __name__=="__main__":
    main()
