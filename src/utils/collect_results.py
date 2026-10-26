"""3DRC Results Collector.

Aggregates per-run results under `outputs/<scene>/` and `data/<scene>/sparse/` into one comparison report:
SfM registration statistics, NVS metrics (`results.json` from Step 5b), mesh evaluation metrics
(`eval/<mesh>/eval_results.json` from Step 5), model sizes, and stage timings (`outputs/benchmark_timings.json`).

Uses only the standard library so it runs in any environment, including machines without a GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Must match METRICS_VERSION in src/mesh/metrics.py (not imported to keep this module stdlib-only)
MESH_METRICS_VERSION = 2

def load_json(path: Path) -> Optional[Any]:
    """Loads a JSON file, returning None if it is missing or malformed."""
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def read_ply_vertex_count(path: Path) -> Optional[int]:
    """Reads the vertex element count from a PLY header without loading the payload."""
    try:
        with open(path, "rb") as f:
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                text = line.decode("ascii", errors="ignore").strip()
                if text.startswith("element vertex"):
                    return int(text.split()[-1])
                if text == "end_header":
                    break
    except Exception:
        return None
    return None

def read_colmap_bin_count(path: Path) -> Optional[int]:
    """Reads the leading uint64 record count of a COLMAP images.bin / points3D.bin file."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        if len(head) == 8:
            return struct.unpack("<Q", head)[0]
    except Exception:
        return None
    return None

def parse_eval_flag(cfg_args: Path) -> Optional[bool]:
    """Returns the `eval` holdout flag recorded in a Gaussian model's cfg_args file."""
    if not cfg_args.is_file():
        return None
    match = re.search(r"\beval=(True|False)\b", cfg_args.read_text(encoding="utf-8", errors="ignore"))
    return None if match is None else match.group(1) == "True"

def _iteration_of(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else -1

def latest_nvs_metrics(results: Any) -> Optional[Dict[str, Any]]:
    """Picks the highest-iteration entry (e.g. `ours_30000`) from an upstream results.json."""
    if not isinstance(results, dict) or not results:
        return None
    key = max(results.keys(), key=_iteration_of)
    entry = results[key]
    if not isinstance(entry, dict):
        return None
    return {
        "iteration": _iteration_of(key),
        "PSNR": entry.get("PSNR"),
        "SSIM": entry.get("SSIM"),
        "LPIPS": entry.get("LPIPS"),
    }

def latest_point_cloud(run_dir: Path) -> Optional[Path]:
    """Returns point_cloud/iteration_<max>/point_cloud.ply for a trained Gaussian model."""
    candidates = [p for p in run_dir.glob("point_cloud/iteration_*/point_cloud.ply") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: _iteration_of(p.parent.name))

@dataclass
class SfmRecord:
    scene: str
    method: str
    rel_dir: str
    registered_images: Optional[int]
    total_images: Optional[int]
    points: Optional[int]
    active: bool
    timing: Optional[Dict[str, Any]] = None

@dataclass
class RunRecord:
    scene: str
    kind: str
    rel_dir: str
    engine: str
    meta: Dict[str, Any]
    eval_split: Optional[bool] = None
    nvs: Optional[Dict[str, Any]] = None
    primitives: Optional[int] = None
    model_mb: Optional[float] = None
    mesh_evals: List[Dict[str, Any]] = field(default_factory=list)
    timing: Optional[Dict[str, Any]] = None
    nvs_timing: Optional[Dict[str, Any]] = None

def load_timings(benchmark_file: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Indexes the latest stage timing and latest NVS-eval timing per output directory."""
    entries = load_json(benchmark_file)
    stage_by_dir: Dict[str, Dict[str, Any]] = {}
    nvs_by_dir: Dict[str, Dict[str, Any]] = {}
    if not isinstance(entries, list):
        return stage_by_dir, nvs_by_dir
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("output_dir"):
            continue
        target = nvs_by_dir if str(entry.get("stage", "")).startswith("nvs_eval") else stage_by_dir
        # Entries are appended chronologically, so later ones win.
        target[entry["output_dir"]] = entry
    return stage_by_dir, nvs_by_dir

def collect_sfm(root: Path, scene: str, stage_by_dir: Dict[str, Dict[str, Any]]) -> List[SfmRecord]:
    sparse_base = root / "data" / scene / "sparse"
    if not sparse_base.is_dir():
        return []
    active_link = sparse_base / "0"
    active = active_link.resolve().name if active_link.is_symlink() else None

    records = []
    for method_dir in sorted(p for p in sparse_base.iterdir() if p.is_dir() and p.name != "0"):
        info = load_json(method_dir / "sfm_info.json") or {}
        rel_dir = method_dir.relative_to(root).as_posix()
        records.append(SfmRecord(
            scene=scene,
            method=method_dir.name,
            rel_dir=rel_dir,
            registered_images=read_colmap_bin_count(method_dir / "images.bin"),
            total_images=info.get("num_images_total"),
            points=read_colmap_bin_count(method_dir / "points3D.bin"),
            active=(method_dir.name == active),
            timing=stage_by_dir.get(rel_dir),
        ))
    return records

def collect_mesh_evals(run_dir: Path, root: Path) -> List[Dict[str, Any]]:
    evals = []
    for result_file in sorted(run_dir.glob("eval/**/eval_results.json")):
        data = load_json(result_file)
        if not isinstance(data, dict):
            continue
        evals.append({
            "mesh": Path(str(data.get("mesh_path", result_file.parent.name))).name,
            "eval_dir": result_file.parent.relative_to(root).as_posix(),
            "hygiene": data.get("hygiene") or {},
            "accuracy": data.get("accuracy"),
            "alignment": data.get("alignment") or {},
            "metrics_version": data.get("metrics_version", 1),
        })
    return evals

def collect_runs(root: Path, scene: str, stage_by_dir: Dict[str, Dict[str, Any]], nvs_by_dir: Dict[str, Dict[str, Any]]) -> List[RunRecord]:
    scene_dir = root / "outputs" / scene
    if not scene_dir.is_dir():
        return []

    runs = []
    for meta_file in sorted(scene_dir.rglob("pipeline_meta.json")):
        run_dir = meta_file.parent
        if "backup" in run_dir.relative_to(scene_dir).parts:
            continue
        meta = load_json(meta_file) or {}
        stage = str(meta.get("stage", ""))
        kind = "mesh" if "Mesh" in stage else "train" if "Training" in stage else "other"
        rel_dir = run_dir.relative_to(root).as_posix()

        record = RunRecord(
            scene=scene,
            kind=kind,
            rel_dir=rel_dir,
            engine=str(meta.get("engine", run_dir.name)),
            meta=meta,
            timing=stage_by_dir.get(rel_dir),
            nvs_timing=nvs_by_dir.get(rel_dir),
        )
        if kind == "train":
            record.eval_split = parse_eval_flag(run_dir / "cfg_args")
            record.nvs = latest_nvs_metrics(load_json(run_dir / "results.json"))
            ply = latest_point_cloud(run_dir)
            if ply is not None:
                record.primitives = read_ply_vertex_count(ply)
                record.model_mb = round(ply.stat().st_size / (1024 * 1024), 1)
        record.mesh_evals = collect_mesh_evals(run_dir, root)
        runs.append(record)
    return runs

def collect_results(root: Path) -> Tuple[Dict[str, List[SfmRecord]], Dict[str, List[RunRecord]]]:
    """Scans the workspace and returns SfM and run records keyed by scene name."""
    stage_by_dir, nvs_by_dir = load_timings(root / "outputs" / "benchmark_timings.json")

    scenes = set()
    for base in (root / "outputs", root / "data"):
        if base.is_dir():
            scenes.update(d.name for d in base.iterdir() if d.is_dir() and not d.name.startswith("."))

    sfm: Dict[str, List[SfmRecord]] = {}
    runs: Dict[str, List[RunRecord]] = {}
    for scene in sorted(scenes):
        scene_sfm = collect_sfm(root, scene, stage_by_dir)
        scene_runs = collect_runs(root, scene, stage_by_dir, nvs_by_dir)
        if scene_sfm:
            sfm[scene] = scene_sfm
        if scene_runs:
            runs[scene] = scene_runs
    return sfm, runs

def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)

def _fmt_time(timing: Optional[Dict[str, Any]]) -> str:
    if not timing or timing.get("elapsed_seconds") is None:
        return "-"
    seconds = int(round(float(timing["elapsed_seconds"])))
    text = f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    if (timing.get("metadata") or {}).get("resumed_from"):
        text += " (resumed)"
    return text

def _env_fingerprints(sfm: Dict[str, List[SfmRecord]], runs: Dict[str, List[RunRecord]]) -> List[Dict[str, Any]]:
    timings = [r.timing for recs in sfm.values() for r in recs]
    timings += [t for recs in runs.values() for r in recs for t in (r.timing, r.nvs_timing)]
    seen, unique = set(), []
    for timing in timings:
        env = (timing or {}).get("env") or {}
        if not env:
            continue
        key = tuple(sorted((k, str(v)) for k, v in env.items()))
        if key not in seen:
            seen.add(key)
            unique.append(env)
    return unique

def render_markdown(sfm: Dict[str, List[SfmRecord]], runs: Dict[str, List[RunRecord]]) -> str:
    """Renders the comparison report as Markdown."""
    lines = [
        "# 3DRC Results Summary",
        "",
        "Generated by `src/utils/collect_results.py`. NVS metrics are reported only for models trained with a holdout split",
        "(`GS_EVAL_MODE=\"true\"`). Times marked `(resumed)` cover only the resumed portion of training.",
        "",
    ]

    envs = _env_fingerprints(sfm, runs)
    if envs:
        lines += ["## Environment Fingerprints", "", "| GPU | Driver | torch | Conda Env | Commit |", "| --- | --- | --- | --- | --- |"]
        for env in envs:
            commit = _fmt(env.get("git_commit"))
            if env.get("git_dirty"):
                commit += " (dirty)"
            lines.append(f"| {_fmt(env.get('gpu'))} | {_fmt(env.get('driver'))} | {_fmt(env.get('torch'))} | {_fmt(env.get('conda_env'))} | {commit} |")
        lines.append("")

    for scene in sorted(set(sfm) | set(runs)):
        lines += [f"## Scene: `{scene}`", ""]

        if scene in sfm:
            lines += ["### Step 1: SfM", "", "| Method | Active | Registered / Total Images | Sparse Points | Time |", "| --- | --- | --- | --- | --- |"]
            for r in sfm[scene]:
                lines.append(f"| {r.method} | {_fmt(r.active)} | {_fmt(r.registered_images)} / {_fmt(r.total_images)} | {_fmt(r.points)} | {_fmt_time(r.timing)} |")
            lines.append("")

        scene_runs = runs.get(scene, [])
        train_runs = [r for r in scene_runs if r.kind == "train"]
        if train_runs:
            lines += [
                "### Step 2: Gaussian Training (NVS on held-out views)",
                "",
                "| Engine | Output | Pose | Iter | Holdout | PSNR | SSIM | LPIPS | Primitives | Size (MB) | Train Time | NVS Eval Time |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            for r in train_runs:
                nvs = r.nvs if r.eval_split else None
                lines.append(
                    f"| {r.engine} | `{r.rel_dir}` | {_fmt(r.meta.get('active_sfm'))} | {_fmt(r.meta.get('iterations'))} | {_fmt(r.eval_split)} "
                    f"| {_fmt((nvs or {}).get('PSNR'), 2)} | {_fmt((nvs or {}).get('SSIM'), 4)} | {_fmt((nvs or {}).get('LPIPS'), 4)} "
                    f"| {_fmt(r.primitives)} | {_fmt(r.model_mb, 1)} | {_fmt_time(r.timing)} | {_fmt_time(r.nvs_timing)} |"
                )
            lines.append("")

        mesh_runs = [r for r in scene_runs if r.kind == "mesh"]
        legacy_evals = False
        if mesh_runs:
            lines += [
                "### Step 3 / 5: Mesh Extraction and Evaluation",
                "",
                "| Engine | Mesh | Vertices | Faces | Watertight | Sliver % | Chamfer L1 (m) | F-score | HD95 (m) | GT Coverage % | Align Fitness | Extract Time |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            for r in mesh_runs:
                if not r.mesh_evals:
                    lines.append(f"| {r.engine} | `{r.rel_dir}` (not evaluated) | - | - | - | - | - | - | - | - | - | {_fmt_time(r.timing)} |")
                    continue
                for ev in r.mesh_evals:
                    hyg, acc, align = ev["hygiene"], ev["accuracy"] or {}, ev["alignment"]
                    fscore = acc.get("fscore")
                    fscore_cell = f"{_fmt(fscore, 4)} @ {acc['fscore_threshold_m']} m" if fscore is not None else "-"
                    if ev["metrics_version"] < MESH_METRICS_VERSION and acc:
                        legacy_evals = True
                        fscore_cell += " (v1)"
                    fitness = _fmt(align.get("fitness"), 3)
                    if align.get("ambiguous"):
                        fitness += " (ambiguous)"
                    lines.append(
                        f"| {r.engine} | `{r.rel_dir}/{ev['mesh']}` | {_fmt(hyg.get('num_vertices'))} | {_fmt(hyg.get('num_faces'))} "
                        f"| {_fmt(hyg.get('is_watertight'))} | {_fmt(hyg.get('sliver_ratio_pct'), 2)} | {_fmt(acc.get('chamfer_l1_m'), 4)} "
                        f"| {fscore_cell} | {_fmt(acc.get('hausdorff_95_m'), 4)} | {_fmt(acc.get('gt_coverage_pct'), 1)} | {fitness} | {_fmt_time(r.timing)} |"
                    )
            lines.append("")
            if legacy_evals:
                lines += [
                    "Rows marked `(v1)` were evaluated before metrics v2 (per-mesh alignment, unclipped Chamfer terms) and are not",
                    "comparable with other rows. Re-run `./scripts/run_3drc.sh eval` on those meshes.",
                    "",
                ]

    if not sfm and not runs:
        lines += ["No results found under `outputs/` or `data/`.", ""]
    return "\n".join(lines)

CSV_FIELDS = [
    "scene", "step", "engine", "output_dir", "active_sfm", "iterations", "holdout", "psnr", "ssim", "lpips",
    "primitives", "model_mb", "mesh", "vertices", "faces", "watertight", "chamfer_l1_m", "fscore", "precision", "recall",
    "fscore_threshold_m", "hausdorff_95_m", "gt_coverage_pct", "align_fitness", "metrics_version", "registered_images", "total_images", "sparse_points", "elapsed_seconds", "resumed_from",
    "gpu", "torch", "git_commit",
]

def build_csv_rows(sfm: Dict[str, List[SfmRecord]], runs: Dict[str, List[RunRecord]]) -> List[Dict[str, Any]]:
    """Flattens all records into one row per SfM model, trained model, or evaluated mesh."""
    def timing_fields(timing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        timing = timing or {}
        env = timing.get("env") or {}
        return {
            "elapsed_seconds": timing.get("elapsed_seconds"),
            "resumed_from": (timing.get("metadata") or {}).get("resumed_from") or None,
            "gpu": env.get("gpu"),
            "torch": env.get("torch"),
            "git_commit": env.get("git_commit"),
        }

    rows: List[Dict[str, Any]] = []
    for scene, records in sfm.items():
        for r in records:
            rows.append({"scene": scene, "step": "sfm", "engine": r.method, "output_dir": r.rel_dir,
                         "registered_images": r.registered_images, "total_images": r.total_images,
                         "sparse_points": r.points, **timing_fields(r.timing)})
    for scene, records in runs.items():
        for r in records:
            base = {"scene": scene, "step": r.kind, "engine": r.engine, "output_dir": r.rel_dir,
                    "active_sfm": r.meta.get("active_sfm"), **timing_fields(r.timing)}
            if r.kind == "train":
                nvs = (r.nvs if r.eval_split else None) or {}
                rows.append({**base, "iterations": r.meta.get("iterations"), "holdout": r.eval_split,
                             "psnr": nvs.get("PSNR"), "ssim": nvs.get("SSIM"), "lpips": nvs.get("LPIPS"),
                             "primitives": r.primitives, "model_mb": r.model_mb})
            elif r.mesh_evals:
                for ev in r.mesh_evals:
                    hyg, acc = ev["hygiene"], ev["accuracy"] or {}
                    rows.append({**base, "mesh": ev["mesh"], "vertices": hyg.get("num_vertices"),
                                 "faces": hyg.get("num_faces"), "watertight": hyg.get("is_watertight"),
                                 "chamfer_l1_m": acc.get("chamfer_l1_m"), "fscore": acc.get("fscore"),
                                 "precision": acc.get("precision"), "recall": acc.get("recall"),
                                 "fscore_threshold_m": acc.get("fscore_threshold_m"), "hausdorff_95_m": acc.get("hausdorff_95_m"),
                                 "gt_coverage_pct": acc.get("gt_coverage_pct"), "align_fitness": ev["alignment"].get("fitness"),
                                 "metrics_version": ev["metrics_version"]})
            else:
                rows.append(base)
    return rows

def write_reports(root: Path, publish: Optional[Path] = None) -> Tuple[Path, Path]:
    """Writes outputs/results_summary.md and outputs/results_runs.csv, optionally copying the Markdown."""
    sfm, runs = collect_results(root)
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    markdown = render_markdown(sfm, runs)
    md_path = outputs_dir / "results_summary.md"
    md_path.write_text(markdown + "\n", encoding="utf-8")

    csv_path = outputs_dir / "results_runs.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in build_csv_rows(sfm, runs):
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})

    if publish is not None:
        publish.parent.mkdir(parents=True, exist_ok=True)
        publish.write_text(markdown + "\n", encoding="utf-8")
    return md_path, csv_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate 3DRC SfM, NVS, mesh, and timing results into one report.")
    parser.add_argument("--root", type=str, default=".", help="Project root containing outputs/ and data/")
    parser.add_argument("--publish", type=str, default=None, help="Also write the Markdown report to this tracked path (e.g. docs/results.md)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    publish = Path(args.publish) if args.publish else None
    md_path, csv_path = write_reports(root, publish)
    print(f"[Results] Wrote {md_path}")
    print(f"[Results] Wrote {csv_path}")
    if publish is not None:
        print(f"[Results] Published Markdown report to {publish}")

if __name__ == "__main__":
    main()
