"""3DRC Package Unified Command Line Interface (CLI) Main Module."""

import argparse
import subprocess
import sys
from pathlib import Path

def run_step(command: str, arg: str = None):
    script_path = Path("scripts/run_3drc.sh")
    if not script_path.exists():
        print(f"[Error] Unified script {script_path} not found.")
        sys.exit(1)
        
    cmd = ["bash", str(script_path), command]
    if arg:
        cmd.append(arg)
        
    print(f"[3DRC CLI] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Error] Pipeline step failed with exit code {e.returncode}")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(
        description="3DRC Package Unified Pipeline CLI Launcher",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="step", help="Pipeline steps to execute")
    
    # sfm subcommand
    sfm_parser = subparsers.add_parser("sfm", help="Step 1: Camera Pose Estimation")
    sfm_parser.add_argument("--method", type=str, default="hloc", choices=["hloc", "vi_sfm", "vggt", "fastmap", "sfm"], help="SfM estimation method")

    # train subcommand
    train_parser = subparsers.add_parser("train", help="Step 2: 3DGS Model Training")
    train_parser.add_argument("--method", type=str, default="3dgs", choices=["3dgs", "planar"], help="3DGS training model method")

    # view subcommand
    subparsers.add_parser("view", help="Step 3: Result Visualization")

    # sugar subcommand
    subparsers.add_parser("sugar", help="Step 4: SuGaR Mesh Reconstruction")

    # grouping subcommand
    subparsers.add_parser("grouping", help="Step 5: Gaussian Grouping Segmentation")

    # pipeline subcommand
    pipeline_parser = subparsers.add_parser("pipeline", help="Run End-to-End Pipeline (Step 1 -> 2 -> 3)")
    pipeline_parser.add_argument("--method", type=str, default="hloc", help="SfM estimation method for end-to-end pipeline")

    args = parser.parse_args()
    
    if not args.step:
        parser.print_help()
        sys.exit(0)
        
    if args.step == "sfm":
        run_step("sfm", args.method)
    elif args.step == "train":
        run_step("train", args.method)
    elif args.step == "view":
        run_step("view")
    elif args.step == "sugar":
        run_step("sugar")
    elif args.step == "grouping":
        run_step("grouping")
    elif args.step == "pipeline":
        run_step("pipeline", args.method)

if __name__ == "__main__":
    main()
