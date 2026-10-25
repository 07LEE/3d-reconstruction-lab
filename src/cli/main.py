"""3DRC Package Unified Command Line Interface (CLI) Main Module.

Thin wrapper around scripts/run_3drc.sh, which is the canonical entry point. Every shell command
is exposed here with the same positional order, so both entry points accept the same steps.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def run_step(command: str, *args: Optional[str]) -> None:
    """Executes a 3DRC unified pipeline step by invoking scripts/run_3drc.sh.

    Args:
        command: Subcommand identifier passed to run_3drc.sh.
        args: Optional trailing arguments; None entries are skipped.

    Returns:
        None
    """
    script_path = PROJECT_ROOT / "scripts" / "run_3drc.sh"
    if not script_path.is_file():
        print(f"[Error] Unified runner script not found at: {script_path}")
        sys.exit(1)

    cmd = ["bash", str(script_path), command]
    cmd.extend(a for a in args if a)

    print(f"[3DRC CLI] Running: {' '.join(cmd)}")
    try:
        # run_3drc.sh resolves configs/ and scripts/ relative to the project root
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as e:
        print(f"[Error] Pipeline step failed with exit code: {e.returncode}")
        sys.exit(e.returncode)

def main() -> None:
    """Main CLI entrypoint for 3DRC package execution."""
    parser = argparse.ArgumentParser(
        description="3DRC Package Unified Pipeline CLI Launcher (wraps scripts/run_3drc.sh)",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Shared --dataset option (default: scene resolved by configs/default_config.sh)
    dataset_opt = argparse.ArgumentParser(add_help=False)
    dataset_opt.add_argument("--dataset", type=str, default=None, help="Dataset directory, e.g. data/<scene> (default: config scene)")

    subparsers = parser.add_subparsers(dest="step", help="Pipeline steps to execute")

    # data subcommand
    subparsers.add_parser("data", parents=[dataset_opt], help="Step 0b: YUV In-Memory Data Load")

    # sfm subcommand
    sfm_parser = subparsers.add_parser("sfm", parents=[dataset_opt], help="Step 1: Camera Pose Estimation")
    sfm_parser.add_argument("--method", type=str, default="hloc", choices=["hloc", "vggt", "fastmap", "sfm"], help="SfM estimation method")

    # train subcommand
    train_parser = subparsers.add_parser("train", parents=[dataset_opt], help="Step 2: Gaussian Splatting Model Training")
    train_parser.add_argument("--method", type=str, default="3dgs", choices=["3dgs", "planargs", "2dgs", "scaffoldgs", "mipsplatting"], help="Gaussian training backend")

    # Shortcuts
    subparsers.add_parser("planargs", parents=[dataset_opt], help="Step 2b: PlanarGS Training")
    subparsers.add_parser("2dgs", parents=[dataset_opt], help="Step 2c: 2D Gaussian Splatting Surfel Training")
    subparsers.add_parser("scaffoldgs", parents=[dataset_opt], help="Step 2d: Scaffold-GS Training")
    subparsers.add_parser("mipsplatting", parents=[dataset_opt], help="Step 2e: Mip-Splatting Anti-Aliased Training")

    # view subcommand
    view_parser = subparsers.add_parser("view", help="Result Visualization")
    view_parser.add_argument("--type", type=str, default="hloc", choices=["hloc", "fastmap"], help="Reconstruction visualization type")

    # mesh subcommands
    sugar_parser = subparsers.add_parser("sugar", parents=[dataset_opt], help="Step 3: SuGaR Mesh Reconstruction")
    sugar_parser.add_argument("--source-model", type=str, default=None, help="Source Gaussian model subdirectory (default: latest under outputs/<scene>/3dgs/)")
    subparsers.add_parser("milo", parents=[dataset_opt], help="Step 3b: MILo Mesh Reconstruction")
    tsdf_parser = subparsers.add_parser("tsdf", parents=[dataset_opt], help="Step 3c: TSDF Mesh Extraction")
    tsdf_parser.add_argument("--source-model", type=str, default=None, help="Source Gaussian model subdirectory (default: 2dgs)")

    # grouping subcommand
    subparsers.add_parser("grouping", parents=[dataset_opt], help="Step 4: Gaussian Grouping Segmentation")

    # eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Step 5: Mesh Evaluation against LiDAR GT PointCloud")
    eval_parser.add_argument("--mesh", type=str, required=True, help="Path to the mesh to evaluate")
    eval_parser.add_argument("--gt", type=str, default=None, help="Path to ground truth point cloud or mesh")

    # nvs subcommand
    nvs_parser = subparsers.add_parser("nvs", parents=[dataset_opt], help="Step 5b: NVS Evaluation (PSNR/SSIM/LPIPS on held-out views)")
    nvs_parser.add_argument("--model", type=str, default="3dgs/inria_30k", help="Model path or subdirectory under outputs/<scene>/")

    # results subcommand
    results_parser = subparsers.add_parser("results", help="Aggregate SfM/NVS/mesh/timing results into one report")
    results_parser.add_argument("--publish", type=str, default=None, help="Also write the Markdown report to this path (e.g. docs/results.md)")

    # outputs subcommand
    subparsers.add_parser("outputs", help="Inspect generated scene artifacts")

    # pipeline subcommand
    pipeline_parser = subparsers.add_parser("pipeline", parents=[dataset_opt], help="Run End-to-End Pipeline")
    pipeline_parser.add_argument("--method", type=str, default="hloc", help="SfM estimation method for end-to-end pipeline")
    pipeline_parser.add_argument("--train", type=str, default=None, choices=["3dgs", "planargs", "2dgs", "scaffoldgs", "mipsplatting"], help="Gaussian training backend (default: 3dgs)")
    pipeline_parser.add_argument("--mesh", type=str, default=None, choices=["sugar", "tsdf", "milo", "none"], help="Mesh engine (default: sugar)")

    args = parser.parse_args()

    if not args.step:
        parser.print_help()
        sys.exit(0)

    dataset = getattr(args, "dataset", None)
    if args.step in ("sfm", "train"):
        run_step(args.step, args.method, dataset)
    elif args.step == "pipeline":
        run_step("pipeline", args.method, dataset,
                 *(["--train", args.train] if args.train else []),
                 *(["--mesh", args.mesh] if args.mesh else []))
    elif args.step in ("sugar", "tsdf"):
        run_step(args.step, dataset, args.source_model)
    elif args.step == "view":
        run_step("view", args.type)
    elif args.step == "eval":
        run_step("eval", args.mesh, args.gt)
    elif args.step == "nvs":
        run_step("nvs", dataset, args.model)
    elif args.step == "results":
        run_step("results", *(["--publish", args.publish] if args.publish else []))
    elif args.step == "outputs":
        run_step("outputs")
    else:
        # data, planargs, 2dgs, scaffoldgs, mipsplatting, milo, grouping
        run_step(args.step, dataset)

if __name__ == "__main__":
    main()
