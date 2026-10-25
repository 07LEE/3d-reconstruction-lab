import json
import subprocess
from pathlib import Path

from src.utils.log_benchmark import _normalize_output_dir, record_stage_timing

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def test_record_stage_timing_appends_entry_with_relative_output_dir(tmp_path):
    bench = tmp_path / "outputs" / "benchmark_timings.json"
    out_dir = tmp_path / "outputs" / "scene" / "2dgs"

    record_stage_timing("2dgs_train", "scene", 12.5, {"iterations": 30000}, benchmark_file=bench, output_dir=out_dir, fingerprint=False, project_root=tmp_path)
    record_stage_timing("nvs_eval_2dgs", "scene", 3.0, benchmark_file=bench, output_dir=out_dir, fingerprint=False, project_root=tmp_path)

    entries = json.loads(bench.read_text())
    assert [e["stage"] for e in entries] == ["2dgs_train", "nvs_eval_2dgs"]
    assert entries[0]["output_dir"] == "outputs/scene/2dgs"
    assert entries[0]["metadata"] == {"iterations": 30000}
    assert entries[0]["formatted_time"] == "00:00:12"

def test_normalize_output_dir_keeps_paths_outside_root_absolute(tmp_path):
    root = tmp_path / "root"
    other = tmp_path / "elsewhere" / "model"
    assert _normalize_output_dir(other, root) == other.as_posix()
    assert _normalize_output_dir(None, root) is None

def test_bench_record_shell_helper_from_other_cwd(tmp_path):
    bench = tmp_path / "timings.json"
    workdir = tmp_path / "work"
    workdir.mkdir()
    script = f"""
set -euo pipefail
source "{PROJECT_ROOT}/scripts/utils/common_logger.sh"
bench_start
bench_record sugar_mesh scene "{PROJECT_ROOT}/outputs/scene/mesh/sugar_3dgs" "source_model=3dgs/inria_30k" "resumed_from=" "eval_split=true"
"""
    subprocess.run(["bash", "-c", script], cwd=workdir, check=True, env={"PATH": "/usr/bin:/bin", "BENCHMARK_FILE_3DRC": str(bench)})

    entry = json.loads(bench.read_text())[0]
    assert entry["output_dir"] == "outputs/scene/mesh/sugar_3dgs"
    assert entry["metadata"] == {"source_model": "3dgs/inria_30k", "resumed_from": "", "eval_split": True}
    assert entry["elapsed_seconds"] >= 0
