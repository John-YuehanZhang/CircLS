"""A2 harness: provenance dirty-check, config expansion, table generation."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import ablation
import provenance as prov_mod
from ablation import CONFIGS, compile_kwargs
from benchsuite import suite


_TOPOLS_GHZ = Path(os.environ.get(
    "TOPOLS_DIR",
    str(Path(__file__).resolve().parents[2] / "TopoLS"))) / "docs/benchmark/ghz_16.qasm"
needs_topols = pytest.mark.skipif(
    not _TOPOLS_GHZ.exists(),
    reason="needs a TopoLS checkout (TOPOLS_DIR); see CONTRIBUTING")


def test_provenance_dirty_refusal(tmp_path, monkeypatch):
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "a.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "init"], cwd=repo, check=True)
    monkeypatch.setattr(prov_mod, "_ROOT", repo)
    p = prov_mod.provenance()
    assert p["dirty"] is False and len(p["circls_sha"]) == 40
    assert "stim" in p["versions"]
    (repo / "b.txt").write_text("dirty")
    with pytest.raises(RuntimeError, match="dirty"):
        prov_mod.provenance()
    assert prov_mod.provenance(allow_dirty=True)["dirty"] is True


@needs_topols
def test_configs_flip_exactly_one_knob():
    cases = {c.name: c for c in suite(include_qasmbench=False)}
    plain = cases["ghz_8"]
    full = compile_kwargs("full", plain)
    assert full == {"assignment": "optimized", "measure_reduction": True,
                    "first_use_init": True, "step_scheduling": True,
                    "parallel_steps": True, "liveness": True,
                    "keep_patches": set()}
    diffs = {
        "no_reduce": ("measure_reduction", False),
        "no_place": ("assignment", "row_major"),
        "no_fui": ("first_use_init", False),
        "no_sched": ("step_scheduling", False),
        "no_parallel": ("parallel_steps", False),
    }
    for cfg, (key, val) in diffs.items():
        kw = compile_kwargs(cfg, plain)
        assert kw[key] == val
        assert {k: v for k, v in kw.items() if k != key} == \
               {k: v for k, v in full.items() if k != key}
    # liveness applies only on consumables, via the approved keep policy
    cons = cases["teleport_4"]
    fkw = compile_kwargs("full", cons)
    assert fkw["liveness"] is True and fkw["keep_patches"]
    assert "liveness" not in compile_kwargs("no_live", cons)
    assert compile_kwargs("full", plain)["keep_patches"] == set()


def test_table_generator(tmp_path):
    prov = {"record": "provenance", "circls_sha": "a" * 40, "dirty": False}

    def row(name, cfg, v1, status="OK"):
        r = {"name": name, "config": cfg, "n": 4, "status": status}
        if status == "OK":
            r["stats"] = {"V1_volume_blocks": v1, "T1_rounds": 10,
                          "S3_tiles_peak": 5, "L1_live_tile_rounds": 8,
                          "internal": {"tile_rounds": 12}}
        return r

    for cfg, v1, st in [("full", 100, "OK"), ("no_reduce", 300, "OK")]:
        with (tmp_path / f"{cfg}.jsonl").open("w") as f:
            f.write(json.dumps({**prov, "config": cfg}) + "\n")
            f.write(json.dumps(row("caseA", cfg, v1, st)) + "\n")
            f.write(json.dumps(row("caseB", cfg, v1)) + "\n")
    import ablation_table
    runs, shas, dirty = ablation_table.load(tmp_path, allow_mixed=False)
    assert set(runs) == {"full", "no_reduce"} and not dirty
    sys.argv = ["ablation_table.py", "--outdir", str(tmp_path), "--tex"]
    ablation_table.main()
    md = (tmp_path / "ablation_table.md").read_text()
    assert "+200.0%" in md          # 600 vs 200 aggregate V1
    assert (tmp_path / "ablation_table.csv").exists()
    assert "do not edit" in (tmp_path / "ablation_table.tex").read_text()


def test_table_refuses_mixed_provenance(tmp_path):
    for cfg, sha in [("full", "a" * 40), ("no_fui", "b" * 40)]:
        with (tmp_path / f"{cfg}.jsonl").open("w") as f:
            f.write(json.dumps({"record": "provenance", "circls_sha": sha,
                                "dirty": False, "config": cfg}) + "\n")
    import ablation_table
    with pytest.raises(SystemExit, match="provenance"):
        ablation_table.load(tmp_path, allow_mixed=False)
