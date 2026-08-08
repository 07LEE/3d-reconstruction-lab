"""3DRC Output Artifact Inspector and Summary Generator.

Scans the `outputs/` directory and generates a structured summary of all
generated 3DGS models, extracted meshes, web viewer files, and evaluation reports
organized by scene name (`outputs/<scene_name>/...`).
"""

import os
import json
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

def inspect_workspace_outputs(project_root: str = "."):
    root = Path(project_root).resolve()
    outputs_dir = root / "outputs"
    data_dir = root / "data"

    artifacts = []

    if outputs_dir.exists():
        # Iterate over scene subdirectories inside outputs/
        for scene_dir in sorted(outputs_dir.iterdir()):
            if not scene_dir.is_dir() or scene_dir.name in ["milo_mesh", "sugar_mesh", "gs_final_precision", "milo", "planargs", "eval"]:
                continue
            
            scene_name = scene_dir.name

            # Stage 2 3DGS
            inria_ply = scene_dir / "3dgs" / "inria_30k" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
            if inria_ply.exists():
                artifacts.append({
                    "scene": scene_name,
                    "stage": "Stage 2 (3DGS)",
                    "category": "Inria 3DGS",
                    "name": "Inria 3DGS 30k Iter Point Cloud",
                    "path": str(inria_ply),
                    "file_link": f"file://{inria_ply}",
                    "info": f"Size: {format_size(inria_ply.stat().st_size)}"
                })

            planargs_dir = scene_dir / "3dgs" / "planargs"
            if planargs_dir.exists():
                artifacts.append({
                    "scene": scene_name,
                    "stage": "Stage 2 (3DGS)",
                    "category": "PlanarGS",
                    "name": "PlanarGS Model Checkpoint",
                    "path": str(planargs_dir),
                    "file_link": f"file://{planargs_dir}",
                    "info": "Selective planar-regularized 3DGS"
                })

            # Stage 3 Mesh
            milo_mesh_clean = scene_dir / "mesh" / "milo" / "mesh_cleaned_largest.ply"
            if milo_mesh_clean.exists():
                artifacts.append({
                    "scene": scene_name,
                    "stage": "Stage 3 (Mesh)",
                    "category": "MILo SDF Mesh",
                    "name": "Cleaned MILo SDF Surface Mesh",
                    "path": str(milo_mesh_clean),
                    "file_link": f"file://{milo_mesh_clean}",
                    "info": f"Size: {format_size(milo_mesh_clean.stat().st_size)} (Single Connected Component)"
                })

            sugar_mesh = scene_dir / "mesh" / "sugar"
            if sugar_mesh.exists():
                artifacts.append({
                    "scene": scene_name,
                    "stage": "Stage 3 (Mesh)",
                    "category": "SuGaR Mesh",
                    "name": "SuGaR UV-Textured Mesh Output",
                    "path": str(sugar_mesh),
                    "file_link": f"file://{sugar_mesh}",
                    "info": "UV-textured OBJ polygon mesh"
                })

            # Stage 4 Eval
            eval_json = scene_dir / "eval" / "eval_results.json"
            if eval_json.exists():
                artifacts.append({
                    "scene": scene_name,
                    "stage": "Stage 4 (Eval)",
                    "category": "Evaluation Report",
                    "name": "Quantitative Topology JSON",
                    "path": str(eval_json),
                    "file_link": f"file://{eval_json}",
                    "info": "Topology hygiene & metric report"
                })

    # Legacy Fallback Scanning (outputs/milo_mesh, outputs/gs_final_precision, outputs/milo)
    legacy_milo_clean = outputs_dir / "milo_mesh" / "mesh_cleaned_largest.ply"
    if legacy_milo_clean.exists():
        artifacts.append({
            "scene": "legacy_milo",
            "stage": "Stage 3 (Mesh)",
            "category": "MILo SDF Mesh",
            "name": "Cleaned MILo SDF Surface Mesh (Legacy)",
            "path": str(legacy_milo_clean),
            "file_link": f"file://{legacy_milo_clean}",
            "info": f"Size: {format_size(legacy_milo_clean.stat().st_size)}"
        })

    legacy_inria_ply = outputs_dir / "gs_final_precision" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    if legacy_inria_ply.exists():
        artifacts.append({
            "scene": "legacy_inria",
            "stage": "Stage 2 (3DGS)",
            "category": "Inria 3DGS",
            "name": "Inria 3DGS 30k Iter Point Cloud (Legacy)",
            "path": str(legacy_inria_ply),
            "file_link": f"file://{legacy_inria_ply}",
            "info": f"Size: {format_size(legacy_inria_ply.stat().st_size)}"
        })

    # Terminal Output
    print("\n" + "=" * 90)
    print(" 3DRC Scene-Based Output Artifacts Inspector")
    print("=" * 90)
    print(f"{'Scene Name':<15} | {'Stage':<15} | {'Category':<18} | {'Artifact Name':<30} | {'Info'}")
    print("-" * 90)
    for a in artifacts:
        print(f"{a['scene']:<15} | {a['stage']:<15} | {a['category']:<18} | {a['name']:<30} | {a['info']}")
    print("=" * 90)

    # Write outputs/summary.md
    summary_md = outputs_dir / "summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# 3DRC Scene-Based Output Artifacts Summary\n\n")
        f.write("A structured overview of all generated camera poses, 3DGS models, extracted meshes, web viewer files, and evaluation reports organized by scene.\n\n")
        f.write("| Scene Name | Stage | Category | Artifact Name | Link | Info |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for a in artifacts:
            f.write(f"| {a['scene']} | {a['stage']} | {a['category']} | {a['name']} | [{Path(a['path']).name}]({a['file_link']}) | {a['info']} |\n")
    print(f"\n[Inspector] Generated structured Markdown summary at: {summary_md}")

if __name__ == "__main__":
    inspect_workspace_outputs()
