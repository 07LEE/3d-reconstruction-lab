"""3DRC Dynamic Output Artifact Inspector and Summary Generator.

Dynamically scans each scene directory (`outputs/<scene_name>/...` and `data/<scene_name>/sparse/0/`)
to generate a clean, portable `summary.md` file inside `outputs/<scene_name>/summary.md`
without duplicating binary files on disk.
"""

import os
import time
from pathlib import Path

def format_size(size_bytes: int) -> str:
    """Format byte sizes to human-readable units."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def format_mtime(timestamp: float) -> str:
    """Format file modification time to YYYY-MM-DD HH:MM:SS."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

def extract_metadata(file_path: Path, rel_path_str: str) -> tuple[str, str, str, str]:
    """Extract (Stage, Algorithm Name, Display Name, Parameters/Config) for an output file."""
    ext = file_path.suffix.lower()
    parent_parts = [p.lower() for p in file_path.parts]

    # Algorithm Identification
    if "sparse" in parent_parts or "sfm" in parent_parts:
        algorithm = "COLMAP / hloc (SfM Poses)"
    elif "inria_30k" in parent_parts:
        algorithm = "Inria 3DGS (Reference)"
    elif "planargs" in parent_parts:
        algorithm = "PlanarGS (Indoor Planar Priors)"
    elif "milo" in parent_parts:
        algorithm = "MILo (Mesh-in-the-Loop SDF)"
    elif "sugar" in parent_parts:
        algorithm = "SuGaR (Surface-Aligned Mesh)"
    else:
        algorithm = "Standard Pipeline Artifact"

    # Stage Identification
    if "sparse" in parent_parts or "sfm" in parent_parts:
        stage = "Stage 1 (SfM)"
    elif "3dgs" in parent_parts or "point_cloud" in parent_parts:
        stage = "Stage 2 (3DGS)"
    elif "mesh" in parent_parts or ext in [".obj", ".stl", ".ply", ".splat", ".mtl"]:
        stage = "Stage 3 (Mesh)"
    elif "eval" in parent_parts or ext == ".json":
        stage = "Stage 4 (Eval)"
    else:
        stage = "Artifact"

    # Disambiguate Display Name
    if file_path.name == "point_cloud.ply":
        parent_name = file_path.parent.name
        if parent_name.startswith("iteration_"):
            display_name = f"{parent_name}/point_cloud.ply"
        else:
            display_name = f"{file_path.parent.name}/point_cloud.ply"
    elif "sparse" in parent_parts:
        display_name = f"sparse/0/{file_path.name}"
    else:
        display_name = file_path.name

    # Config / Parameter Details
    config = "Default Pipeline Parameters"
    if "sparse" in parent_parts:
        if file_path.name == "cameras.bin":
            config = "COLMAP Camera Intrinsics Model"
        elif file_path.name == "images.bin":
            config = "COLMAP Camera Extrinsics / Poses"
        elif file_path.name == "points3D.bin":
            config = "COLMAP Sparse 3D Point Cloud Binary"
        elif file_path.name == "points3D.ply":
            config = "COLMAP Sparse Point Cloud (PLY Viewable)"
    elif "sugar" in parent_parts:
        if "sugarfine_" in file_path.name:
            name = file_path.name
            parts = name.split("_")
            params = []
            for p in parts:
                if p.startswith("3Dgs"):
                    params.append(f"GS Iter: {p[4:]}")
                elif p.startswith("sdfnorm"):
                    params.append(f"SDF Norm: 0.{p[7:]}")
                elif p.startswith("decim"):
                    params.append(f"Faces: {int(p[5:]):,}")
                elif p.startswith("gaussperface"):
                    params.append(f"Gaussians/Face: {p[12:]}")
            if params:
                config = ", ".join(params)
    elif "milo" in parent_parts:
        if file_path.name == "mesh_cleaned_largest.ply":
            config = "Marching Tetrahedra SDF (Largest Connected Component Cleaned)"
        elif file_path.name == "gaussian.splat":
            config = "32-byte Splat WebGL Viewer Format"
        elif file_path.name == "gaussian_xyz_rgb.ply":
            config = "Universal XYZ+RGB Point Cloud"
    elif "inria_30k" in parent_parts:
        if "iteration_30000" in parent_parts:
            config = "30,000 Iterations Final Checkpoint (PINHOLE single-camera)"
        elif "iteration_7000" in parent_parts:
            config = "7,000 Iterations Coarse Checkpoint"

    return stage, algorithm, display_name, config

def inspect_workspace_outputs(project_root: str = "."):
    root = Path(project_root).resolve()
    outputs_dir = root / "outputs"
    data_dir = root / "data"

    all_scene_artifacts = {}

    scene_names = set()
    if outputs_dir.exists():
        for d in outputs_dir.iterdir():
            if d.is_dir():
                scene_names.add(d.name)
    if data_dir.exists():
        for d in data_dir.iterdir():
            if d.is_dir():
                scene_names.add(d.name)

    for scene_name in sorted(scene_names):
        scene_dir = outputs_dir / scene_name
        scene_artifacts = []

        # Stage 1 SfM Check (Scan all method subfolders inside data/<scene_name>/sparse/)
        sparse_base_dir = data_dir / scene_name / "sparse"
        if sparse_base_dir.exists():
            active_target = None
            symlink_0 = sparse_base_dir / "0"
            if symlink_0.is_symlink():
                try:
                    active_target = symlink_0.resolve().name
                except Exception:
                    active_target = None

            for method_dir in sorted(sparse_base_dir.iterdir()):
                if not method_dir.is_dir() or method_dir.name == "0":
                    continue
                
                method_name = method_dir.name
                is_active = (active_target == method_name)
                
                # Read sfm_info.json metadata if available
                info_file = method_dir / "sfm_info.json"
                sfm_method_name = method_name
                if info_file.exists():
                    try:
                        import json
                        with open(info_file, "r") as f:
                            sfm_info = json.load(f)
                            sfm_method_name = sfm_info.get("method", method_name)
                    except Exception:
                        pass

                # Map algorithm display name
                algorithm_names = {
                    "hloc": "COLMAP / hloc (SuperPoint+SuperGlue)",
                    "vggt": "VGGT-Omega (Feed-Forward Transformer)",
                    "fastmap": "FastMap GPU SfM",
                    "sfm": "COLMAP / SIFT Baseline",
                    "vi_sfm": "Visual-Inertial SfM (RGB+IMU)"
                }
                algorithm_label = algorithm_names.get(sfm_method_name, f"SfM ({sfm_method_name})")
                if is_active:
                    algorithm_label += " [Active Poses]"

                for sfm_file in sorted(method_dir.glob("*")):
                    if not sfm_file.is_file() or sfm_file.suffix.lower() not in [".bin", ".ply", ".txt"]:
                        continue
                    if sfm_file.name == "sfm_info.json":
                        continue

                    rel_link = f"../../data/{scene_name}/sparse/{method_name}/{sfm_file.name}"
                    stage, _, display_name, config = extract_metadata(sfm_file, rel_link)
                    display_name = f"sparse/{method_name}/{sfm_file.name}"
                    mtime = format_mtime(sfm_file.stat().st_mtime)
                    size_str = format_size(sfm_file.stat().st_size)

                    scene_artifacts.append({
                        "scene": scene_name,
                        "stage": "Stage 1 (SfM)",
                        "algorithm": algorithm_label,
                        "config": config,
                        "display_name": display_name,
                        "rel_path": rel_link,
                        "size": size_str,
                        "mtime": mtime
                    })

        # Discover all key deliverables inside outputs/<scene_name>/
        if scene_dir.exists():
            for file_path in sorted(scene_dir.rglob("*")):
                if not file_path.is_file() or file_path.name.startswith("."):
                    continue

                if file_path.stem.isdigit() and file_path.suffix.lower() in [".png", ".jpg"]:
                    continue

                if file_path.name == "summary.md":
                    continue

                ext = file_path.suffix.lower()
                if ext not in [".ply", ".obj", ".splat", ".json", ".png", ".mtl", ".pth"]:
                    continue

                rel_path = file_path.relative_to(scene_dir)
                stage, algorithm, display_name, config = extract_metadata(file_path, str(rel_path))
                mtime = format_mtime(file_path.stat().st_mtime)
                size_str = format_size(file_path.stat().st_size)

                scene_artifacts.append({
                    "scene": scene_name,
                    "stage": stage,
                    "algorithm": algorithm,
                    "config": config,
                    "display_name": str(rel_path),
                    "rel_path": str(rel_path),
                    "size": size_str,
                    "mtime": mtime
                })

            if scene_artifacts:
                all_scene_artifacts[scene_name] = scene_artifacts

                # Write detailed scene-specific summary.md inside outputs/<scene_name>/summary.md
                scene_summary_md = scene_dir / "summary.md"
                with open(scene_summary_md, "w", encoding="utf-8") as f:
                    f.write(f"# Scene Execution & Artifacts Summary: `{scene_name}`\n\n")
                    f.write(f"Detailed execution record and artifact inventory for scene **{scene_name}**.\n\n")
                    f.write("| Stage | Algorithm | Checkpoint / File Subpath | Link | Execution Parameters / Details | Size | Creation Time |\n")
                    f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
                    for a in scene_artifacts:
                        f.write(f"| {a['stage']} | {a['algorithm']} | `{a['display_name']}` | [{a['display_name']}]({a['rel_path']}) | {a['config']} | {a['size']} | {a['mtime']} |\n")
                print(f"[Inspector] Wrote detailed scene summary to: {scene_summary_md}")

    # Terminal Output Table
    print("\n" + "=" * 125)
    print(" 3DRC Scene Output Artifacts Inspector")
    print("=" * 125)
    for s_name, s_arts in all_scene_artifacts.items():
        print(f"\n[Scene: {s_name}]")
        print(f"{'Stage':<15} | {'Algorithm':<26} | {'Subpath / File Name':<38} | {'Size':<10} | {'Creation Time'}")
        print("-" * 125)
        for a in s_arts:
            print(f"{a['stage']:<15} | {a['algorithm']:<26} | {a['display_name']:<38} | {a['size']:<10} | {a['mtime']}")
    print("=" * 125)

if __name__ == "__main__":
    inspect_workspace_outputs()
