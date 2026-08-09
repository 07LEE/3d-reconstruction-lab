"""3DRC Dynamic Output Artifact Inspector and Provenance Tracker.

Dynamically scans each scene directory (`outputs/<scene_name>/...` and `data/<scene_name>/sparse/`)
to discover deliverables and inspect execution provenance metadata without static hardcoding.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

def format_size(size_bytes: int) -> str:
    """Formats raw byte sizes into human-readable unit strings.

    Args:
        size_bytes: Size in bytes.

    Returns:
        str: Human-readable size string (e.g., '12.5 MB', '1.24 GB').
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def format_mtime(timestamp: float) -> str:
    """Formats a POSIX timestamp into a standard datetime string.

    Args:
        timestamp: POSIX epoch timestamp in seconds.

    Returns:
        str: Formatted datetime string (YYYY-MM-DD HH:MM:SS).
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

@dataclass
class ArtifactItem:
    scene: str
    stage: str
    algorithm: str
    display_name: str
    rel_path: str
    size: str
    mtime: str
    source_lineage: str = "Unknown"
    details: str = "Default Parameters"

def load_meta_json(target_dir: Path) -> Optional[Dict[str, Any]]:
    """Recursively checks for pipeline_meta.json or sfm_info.json in directory hierarchy."""
    curr = target_dir
    while curr != curr.parent:
        meta_file = curr / "pipeline_meta.json"
        if meta_file.is_file():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        info_file = curr / "sfm_info.json"
        if info_file.is_file():
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        if curr.name in ["outputs", "data", ""]:
            break
        curr = curr.parent
    return None

def extract_artifact_info(file_path: Path, scene_dir: Path, active_sfm: Optional[str] = None) -> ArtifactItem:
    """Dynamically extracts artifact metadata and execution provenance without static assumptions.

    Args:
        file_path: Absolute Path to the discovered output file.
        scene_dir: Base scene outputs directory.
        active_sfm: Active SfM method name (if available).

    Returns:
        ArtifactItem: Populated artifact data structure.
    """
    rel_path = file_path.relative_to(scene_dir)
    rel_parts = [p.lower() for p in rel_path.parts]
    ext = file_path.suffix.lower()

    # Look up dynamic metadata JSON
    meta = load_meta_json(file_path.parent)

    # 1. Infer Stage Dynamically from Directory Hierarchy
    if "sparse" in rel_parts:
        stage = "Stage 1 (SfM)"
    elif "mesh" in rel_parts:
        stage = "Stage 3 (Mesh)"
    elif "eval" in rel_parts or ext == ".json":
        stage = "Stage 5 (Eval)"
    elif any(k in rel_parts for k in ["3dgs", "2dgs", "planargs", "point_cloud"]):
        stage = "Stage 2 (Training)"
    else:
        stage = "Artifact"

    # 2. Extract Algorithm / Engine Name Dynamically
    if meta and "engine" in meta:
        algorithm = meta["engine"]
    elif meta and "algorithm" in meta:
        algorithm = meta["algorithm"]
    elif meta and "method" in meta:
        algorithm = f"SfM ({meta['method']})"
    else:
        # Dynamic fallback based on immediate parent directory name
        if "sparse" in rel_parts:
            method_idx = rel_parts.index("sparse") + 1 if len(rel_parts) > rel_parts.index("sparse") + 1 else -1
            algorithm = f"SfM ({rel_path.parts[method_idx]})" if method_idx != -1 else "SfM Pose Model"
        elif "mesh" in rel_parts:
            mesh_idx = rel_parts.index("mesh") + 1 if len(rel_parts) > rel_parts.index("mesh") + 1 else -1
            algorithm = f"Mesh ({rel_path.parts[mesh_idx]})" if mesh_idx != -1 else "Extracted Mesh"
        elif len(rel_path.parts) > 1:
            algorithm = f"Model ({rel_path.parts[0]})"
        else:
            algorithm = "Scene Deliverable"

    # 3. Extract Provenance / Upstream Source Lineage Dynamically
    if meta and "source_model_path" in meta:
        source_lineage = f"Model: {meta['source_model_path']}"
        if meta.get("active_sfm") and meta["active_sfm"] != "unknown":
            source_lineage += f" [Pose: {meta['active_sfm']}]"
    elif meta and "source_dataset" in meta:
        source_lineage = f"Dataset: {meta['source_dataset']}"
        if meta.get("active_sfm") and meta["active_sfm"] != "unknown":
            source_lineage += f" [Pose: {meta['active_sfm']}]"
    elif "sparse" in rel_parts:
        source_lineage = "Raw Captured Frames (data/<scene>/images)"
    elif "mesh" in rel_parts:
        # Check subpath under mesh/ (e.g. mesh/sugar_3dgs_inria_30k or mesh/2dgs or mesh/milo)
        mesh_subdir = rel_path.parts[1] if len(rel_path.parts) > 1 else ""
        source_lineage = f"Derived from scene pipeline ({mesh_subdir})"
        if active_sfm:
            source_lineage += f" [Active Pose: {active_sfm}]"
    elif stage == "Stage 2 (Training)":
        source_lineage = f"Stage 1 SfM Active Poses ({active_sfm or 'sparse/0'})"
    elif stage == "Stage 5 (Eval)":
        source_lineage = "Reconstructed 3D Mesh + Ground Truth LiDAR PCD"
    else:
        source_lineage = "Scene Root Workspace"

    # 4. Details / Hyperparameters
    details = "Default Pipeline Parameters"
    if file_path.name == "cameras.bin":
        details = "COLMAP Camera Intrinsics Model"
    elif file_path.name == "images.bin":
        details = "COLMAP Camera Extrinsics / Poses"
    elif file_path.name == "points3D.bin":
        details = "COLMAP Sparse 3D Point Cloud Binary"
    elif file_path.name == "points3D.ply":
        details = "COLMAP Sparse Point Cloud (PLY Viewable)"
    elif "iteration_" in str(file_path):
        for part in file_path.parts:
            if part.startswith("iteration_"):
                details = f"Checkpoint {part.replace('iteration_', '')} Iterations"
                break
    elif file_path.name == "mesh_cleaned_largest.ply":
        details = "Cleaned Largest Connected Component Mesh"
    elif file_path.name == "tsdf_mesh.ply":
        details = "Open3D TSDF Voxel Integration Mesh"

    size_str = format_size(file_path.stat().st_size)
    mtime_str = format_mtime(file_path.stat().st_mtime)

    return ArtifactItem(
        scene=scene_dir.name,
        stage=stage,
        algorithm=algorithm,
        display_name=str(rel_path),
        rel_path=str(rel_path),
        size=size_str,
        mtime=mtime_str,
        source_lineage=source_lineage,
        details=details,
    )

def inspect_workspace_outputs(project_root: str | Path = ".") -> Dict[str, List[ArtifactItem]]:
    """Scans outputs/ and data/ directories to collect and record scene artifacts with dynamic provenance.

    Args:
        project_root: Root workspace directory to scan.

    Returns:
        Dict[str, List[ArtifactItem]]: Dictionary mapping scene names to discovered artifact items.
    """
    root = Path(project_root).resolve()
    outputs_dir = root / "outputs"
    data_dir = root / "data"

    all_scene_artifacts: Dict[str, List[ArtifactItem]] = {}

    scene_names: Set[str] = set()
    if outputs_dir.is_dir():
        scene_names.update(d.name for d in outputs_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    if data_dir.is_dir():
        scene_names.update(d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith("."))

    for scene_name in sorted(scene_names):
        scene_dir = outputs_dir / scene_name
        scene_data_dir = data_dir / scene_name
        scene_artifacts: List[ArtifactItem] = []

        # 1. Determine active SfM target dynamically from symlink
        active_sfm: Optional[str] = None
        symlink_0 = scene_data_dir / "sparse" / "0"
        if symlink_0.is_symlink():
            try:
                active_sfm = symlink_0.resolve().name
            except Exception:
                active_sfm = None

        # 2. Discover all SfM Methods inside data/<scene_name>/sparse/
        sparse_base = scene_data_dir / "sparse"
        if sparse_base.is_dir():
            for method_dir in sorted(sparse_base.iterdir()):
                if not method_dir.is_dir() or method_dir.name == "0":
                    continue
                is_active = (active_sfm == method_dir.name)
                for sfm_file in sorted(method_dir.glob("*")):
                    if not sfm_file.is_file() or sfm_file.suffix.lower() not in [".bin", ".ply", ".txt"]:
                        continue
                    if sfm_file.name in ["sfm_info.json", "pipeline_meta.json"]:
                        continue

                    rel_link = f"../../data/{scene_name}/sparse/{method_dir.name}/{sfm_file.name}"
                    item = extract_artifact_info(sfm_file, scene_data_dir, active_sfm=active_sfm)
                    item.rel_path = rel_link
                    item.display_name = f"sparse/{method_dir.name}/{sfm_file.name}"
                    if is_active:
                        item.algorithm += " [Active Poses]"
                    item.source_lineage = "Raw Captured Frames (data/<scene>/images)"
                    scene_artifacts.append(item)

        # 3. Discover all Deliverables inside outputs/<scene_name>/
        if scene_dir.is_dir():
            for file_path in sorted(scene_dir.rglob("*")):
                if not file_path.is_file() or file_path.name.startswith("."):
                    continue
                if file_path.stem.isdigit() and file_path.suffix.lower() in [".png", ".jpg"]:
                    continue
                if file_path.name in ["summary.md", "pipeline_meta.json"]:
                    continue

                ext = file_path.suffix.lower()
                if ext not in [".ply", ".obj", ".splat", ".json", ".png", ".mtl", ".pth"]:
                    continue

                item = extract_artifact_info(file_path, scene_dir, active_sfm=active_sfm)
                scene_artifacts.append(item)

            if scene_artifacts:
                all_scene_artifacts[scene_name] = scene_artifacts

                # Write scene-specific summary.md with Dynamic Lineage Tracking
                scene_summary_md = scene_dir / "summary.md"
                with open(scene_summary_md, "w", encoding="utf-8") as f:
                    f.write(f"# Scene Execution & Artifacts Summary: `{scene_name}`\n\n")
                    f.write(f"Dynamic execution record, artifact inventory, and provenance lineage for scene **{scene_name}**.\n\n")
                    f.write("| Stage | Engine / Algorithm | File Subpath | Upstream Source Lineage | Execution Details | Size | Creation Time |\n")
                    f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
                    for a in scene_artifacts:
                        f.write(f"| {a.stage} | {a.algorithm} | [{a.display_name}]({a.rel_path}) | `{a.source_lineage}` | {a.details} | {a.size} | {a.mtime} |\n")
                print(f"[Inspector] Wrote dynamic scene summary with provenance to: {scene_summary_md}")

    # Terminal Output Table
    print("\n" + "=" * 145)
    print(" 3DRC Dynamic Output Artifacts Inspector & Lineage Tracker")
    print("=" * 145)
    for s_name, s_arts in all_scene_artifacts.items():
        print(f"\n[Scene: {s_name}]")
        print(f"{'Stage':<18} | {'Engine / Algorithm':<26} | {'Subpath / File Name':<34} | {'Upstream Source Lineage':<40} | {'Size':<10}")
        print("-" * 145)
        for a in s_arts:
            print(f"{a.stage:<18} | {a.algorithm:<26} | {a.display_name:<34} | {a.source_lineage:<40} | {a.size:<10}")
    print("=" * 145)

    return all_scene_artifacts

if __name__ == "__main__":
    inspect_workspace_outputs()
