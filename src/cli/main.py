"""3DRC Package Unified Command Line Interface (CLI) Main Module."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

def run_step(command: str, arg: Optional[str] = None) -> None:
    """Executes a 3DRC unified pipeline step by invoking scripts/run_3drc.sh.

    Args:
        command: Subcommand identifier passed to run_3drc.sh.
        arg: Optional secondary argument string.

    Returns:
        None
    """
    script_path = Path("scripts/run_3drc.sh").resolve()
    if not script_path.is_file():
        print(f"[Error] Unified runner script not found at: {script_path}")
        sys.exit(1)

    cmd = ["bash", str(script_path), command]
    if arg:
        cmd.append(arg)

    print(f"[3DRC CLI] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Error] Pipeline step failed with exit code: {e.returncode}")
        sys.exit(e.returncode)

def main() -> None:
    """Main CLI entrypoint for 3DRC package execution."""
    parser = argparse.ArgumentParser(
        description="3DRC Package Unified Pipeline CLI Launcher",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="step", help="Pipeline steps to execute")

    # sfm subcommand
    sfm_parser = subparsers.add_parser("sfm", help="Step 1: Camera Pose Estimation")
    sfm_parser.add_argument("--method", type=str, default="hloc", choices=["hloc", "vggt", "fastmap", "sfm"], help="SfM estimation method")

    # train subcommand
    train_parser = subparsers.add_parser("train", help="Step 2: Gaussian Splatting Model Training")
    train_parser.add_argument("--method", type=str, default="3dgs", choices=["3dgs", "planargs", "2dgs", "scaffoldgs"], help="Gaussian training backend")

    # 2dgs shortcut
    subparsers.add_parser("2dgs", help="Step 2c: 2D Gaussian Splatting Surfel Training")

    # view subcommand
    view_parser = subparsers.add_parser("view", help="Result Visualization")
    view_parser.add_argument("--type", type=str, default="hloc", choices=["hloc", "fastmap"], help="Reconstruction visualization type")

    # mesh subcommands
    subparsers.add_parser("sugar", help="Step 3: SuGaR Mesh Reconstruction")
    subparsers.add_parser("milo", help="Step 3b: MILo Mesh Reconstruction")
    subparsers.add_parser("tsdf", help="Step 3c: TSDF Mesh Extraction")

    # grouping subcommand
    subparsers.add_parser("grouping", help="Step 4: Gaussian Grouping Segmentation")

    # outputs subcommand
    subparsers.add_parser("outputs", help="Inspect generated scene artifacts")

    # pipeline subcommand
    pipeline_parser = subparsers.add_parser("pipeline", help="Run End-to-End Pipeline")
    pipeline_parser.add_argument("--method", type=str, default="hloc", help="SfM estimation method for end-to-end pipeline")

    args = parser.parse_args()

    if not args.step:
        parser.print_help()
        sys.exit(0)

    if args.step == "sfm":
        run_step("sfm", args.method)
    elif args.step == "train":
        run_step("train", args.method)
    elif args.step == "2dgs":
        run_step("2dgs")
    elif args.step == "view":
        run_step("view", getattr(args, "type", "hloc"))
    elif args.step in ["sugar", "milo", "tsdf", "grouping", "outputs"]:
        run_step(args.step)
    elif args.step == "pipeline":
        run_step("pipeline", args.method)

if __name__ == "__main__":
    main()
