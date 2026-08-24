"""Stage 2b v2 (daemon): up to 3 MWPF points in flight, each its own
subprocess with a 32-way pool; resumable; slowest-first by circuit size."""
import subprocess, sys, time, os
from pathlib import Path
SP = Path("<workdir>")
CIRC, PTS = SP / "resample" / "circuits", SP / "resample" / "points"
MAXRUN = 6

def _main():
    running = {}
    done_marker = SP / "resample" / "compile_stage.log"
    while True:
        for tag, p in list(running.items()):
            if p.poll() is not None:
                print("finished", tag, "rc", p.returncode, flush=True); running.pop(tag)
        todo = [f.stem for f in CIRC.glob("*_tclass_*.stim")
                if not (PTS / f"{f.stem}_p0.0005_mwpf.json").exists() and f.stem not in running]
        # low-LER points (the d5 ones) are the long poles: start them first
        todo.sort(key=lambda t: (0 if t.endswith("_d5") else 1, t))
        while todo and len(running) < MAXRUN:
            tag = todo.pop(0)
            running[tag] = subprocess.Popen([sys.executable, str(SP / "resample" / "mwpf_point.py"), tag],
                                            cwd=str(SP / "circls_dev"), env={**os.environ, "MWPF_WORKERS": "32"})
            print("started", tag, flush=True)
        stage1_done = done_marker.exists() and "COMPILE STAGE DONE" in done_marker.read_text()
        sat_done = (CIRC / "sat_n7_tclass_full_d5.stim").exists()
        remaining = [f for f in CIRC.glob("*_tclass_*.stim") if not (PTS / f"{f.stem}_p0.0005_mwpf.json").exists()]
        if stage1_done and sat_done and not remaining and not running:
            break
        time.sleep(60)
    print("MWPF SAMPLER DONE", flush=True)

if __name__ == "__main__":
    _main()
