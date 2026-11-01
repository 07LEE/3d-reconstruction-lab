"""3DRC Package Unified Command Line Interface (CLI) Main Module.

The canonical entry point for every pipeline step (ADR 0022). Arguments are parsed here once;
scripts/run_3drc.sh and 3drc.py only forward to this module. Each stage script receives its
arguments in a fixed positional order and keeps its own conda environment activation.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = "configs/default_config.sh"

SFM_METHODS = ("hloc", "vggt", "fastmap", "sfm")
VIEW_TYPES = ("hloc", "fastmap")
MESH_ENGINES = ("sugar", "tsdf", "milo", "none")
# Mesh engines that read a specific model format, with the backends that can feed them
MESH_SOURCES = {
    "sugar": (("3dgs", "planargs"), "SuGaR requires a 3DGS-format model (--train 3dgs or planargs)"),
    "tsdf": (("2dgs",), "TSDF extraction requires a 2DGS model (--train 2dgs)"),
}
# Commands whose arguments go unparsed to a Python tool: (message, command after the 3drc python)
FORWARDED = {
    "eval": ("Executing Step 5: Mesh Quantitative Evaluation...", ["-m", "src.mesh.eval_mesh"]),
    "results": (None, ["src/utils/collect_results.py"]),
}
CONFIG_KEYS = ("DATA_DIR", "SCENE_NAME", "OUTPUT_DIR", "SFM_METHOD", "GS_EVAL_MODE")


class CliError(Exception):
    """A user error reported as '[FATAL] <message>' with exit code 1."""


@dataclass
class Backend:
    """One row of configs/backends.txt."""

    name: str
    script: str
    model_subdir: str
    shortcut: bool
    title: str

    @property
    def header(self) -> str:
        step = self.script.split("_", 1)[0].lstrip("0")
        return f"Step {step}: {self.title} Training"


def load_backends(root: Path = PROJECT_ROOT) -> Dict[str, Backend]:
    """Reads the training backend registry.

    Args:
        root: Project root containing configs/backends.txt.

    Returns:
        Dict[str, Backend]: Backends by name, in registry order.
    """
    backends: Dict[str, Backend] = {}
    for line in (root / "configs" / "backends.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, script, model_subdir, shortcut, title = line.split()[:5]
        backends[name] = Backend(name, script, model_subdir, shortcut == "yes", title.replace("_", " "))
    return backends


def load_config(dataset: Optional[str]) -> Dict[str, str]:
    """Sources configs/default_config.sh in bash and returns the variables the CLI needs.

    The config owns scene resolution (dataset argument, then SCENE_NAME, then the most recent
    dataset with a warning), so it is not reimplemented here.

    Args:
        dataset: Dataset directory given on the command line, if any.

    Returns:
        Dict[str, str]: Values of CONFIG_KEYS after sourcing.
    """
    if dataset and not (PROJECT_ROOT / dataset).is_dir():
        raise CliError(f"Dataset directory not found: '{dataset}'")
    script = 'source "$0" "$@" >&2 || exit 1; printf "%s\\0" ' + " ".join(f'"${{{key}:-}}"' for key in CONFIG_KEYS)
    result = subprocess.run(["bash", "-c", script, CONFIG_PATH, *([dataset] if dataset else [])],
                            cwd=PROJECT_ROOT, stdout=subprocess.PIPE)
    if result.returncode != 0:
        sys.exit(result.returncode)
    return dict(zip(CONFIG_KEYS, result.stdout.decode().split("\0")))


def python_bin() -> str:
    """Python of the 3drc conda environment, or python3 when it is not installed."""
    conda = shutil.which("conda")
    base = None
    if conda:
        out = subprocess.run([conda, "info", "--base"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        base = out.stdout.strip() if out.returncode == 0 else None
    candidate = Path(base or Path.home() / "miniconda3") / "envs" / "3drc" / "bin" / "python"
    return str(candidate) if os.access(candidate, os.X_OK) else "python3"


class Runner:
    """Runs stage scripts and Python modules from the project root, stopping at the first failure."""

    def __init__(self, scene: Optional[str] = None) -> None:
        self.env = dict(os.environ)
        if scene:
            # Stage scripts source the config again; the scene keeps them on the resolved dataset without a second notice
            self.env["SCENE_NAME"] = scene

    def run(self, cmd: Sequence[str]) -> None:
        result = subprocess.run(list(cmd), cwd=PROJECT_ROOT, env=self.env)
        if result.returncode != 0:
            print(f"[FATAL] Step failed with exit code {result.returncode}: {' '.join(cmd)}", file=sys.stderr)
            sys.exit(result.returncode)

    def script(self, name: str, *args: str) -> None:
        self.run([f"./scripts/{name}", *args])


def header(title: str) -> None:
    print(f"[STEP]  {title}", flush=True)


def find_gt(dataset: str, gt: Optional[str]) -> Optional[str]:
    """GT point cloud for mesh evaluation: --gt or PIPELINE_GT, else lidar.pcd or lidar.ply in the dataset."""
    if gt:
        if not (PROJECT_ROOT / gt).is_file():
            raise CliError(f"GT point cloud not found: '{gt}'")
        return gt
    for name in ("lidar.pcd", "lidar.ply"):
        if (PROJECT_ROOT / dataset / name).is_file():
            return f"{dataset}/{name}"
    return None


def mesh_files(engine: str, output_dir: str, scene: str, model_subdir: str) -> List[Path]:
    """Meshes the pipeline's mesh step writes, relative to the project root."""
    base = Path(output_dir) / scene / "mesh"
    if engine == "sugar":
        return sorted((PROJECT_ROOT / base / f"sugar_{model_subdir.replace('/', '_')}" / "refined_mesh" / scene).glob("*.obj"))
    if engine == "tsdf":
        return [PROJECT_ROOT / base / Path(model_subdir).name / "tsdf_mesh.ply"]
    if engine == "milo":
        return [PROJECT_ROOT / base / "milo" / "mesh_cleaned_largest.ply"]
    return []


def run_pipeline(args: argparse.Namespace, backends: Dict[str, Backend]) -> None:
    """SfM -> training -> [NVS] -> mesh -> mesh evaluation -> results, checking the combination first."""
    train = args.train or os.environ.get("PIPELINE_TRAIN") or "3dgs"
    mesh = args.mesh or os.environ.get("PIPELINE_MESH") or "sugar"
    if train not in backends:
        raise CliError(f"Unknown training backend: '{train}' (expected: {', '.join(backends)})")
    if mesh not in MESH_ENGINES:
        raise CliError(f"Unknown mesh engine: '{mesh}' (expected: {', '.join(MESH_ENGINES)})")
    if mesh in MESH_SOURCES and train not in MESH_SOURCES[mesh][0]:
        raise CliError(f"{MESH_SOURCES[mesh][1]}, got '{train}'")

    config = load_config(args.dataset)
    dataset, scene = config["DATA_DIR"], config["SCENE_NAME"]
    if args.dataset:
        dataset, scene = args.dataset, Path(args.dataset).name
    method = args.method or config["SFM_METHOD"]
    backend = backends[train]
    gt = find_gt(dataset, args.gt or os.environ.get("PIPELINE_GT"))

    run = Runner(scene)
    print(f"[3DRC CLI] Starting End-to-End Automated Pipeline (SfM: {method}, Train: {train}, Mesh: {mesh}, Dataset: {dataset})...", flush=True)
    run.script("01_sfm_hloc.sh", method, dataset)
    run.script(backend.script, dataset)
    if (os.environ.get("GS_EVAL_MODE") or config["GS_EVAL_MODE"]) == "true":
        run.script("05_eval_nvs.sh", dataset, backend.model_subdir)
    if mesh == "sugar":
        run.script("03_train_sugar.sh", dataset, backend.model_subdir)
    elif mesh == "tsdf":
        run.script("03c_mesh_tsdf.sh", dataset, backend.model_subdir)
    elif mesh == "milo":
        run.script("03b_train_milo.sh", dataset)

    # Step 5: evaluate the meshes this run produced (topology only when no GT is available)
    if mesh != "none":
        if gt is None:
            print(f"[WARN]  No GT point cloud (--gt, or {dataset}/lidar.pcd); mesh evaluation covers topology only.", flush=True)
        found = [m for m in mesh_files(mesh, config["OUTPUT_DIR"], scene, backend.model_subdir) if m.is_file()]
        if not found:
            raise CliError(f"Mesh evaluation found no {mesh} mesh in {config['OUTPUT_DIR']}/{scene}/mesh")
        for mesh_file in found:
            run.run([python_bin(), "-m", "src.mesh.eval_mesh", "--mesh", str(mesh_file.relative_to(PROJECT_ROOT)), *(["--gt", gt] if gt else [])])

    run.run([python_bin(), "src/utils/collect_results.py"])
    print("[3DRC CLI] End-to-End Automated Pipeline Execution Finished!")


def build_parser(backends: Dict[str, Backend]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="3drc",
        description="3DRC pipeline CLI. ./scripts/run_3drc.sh and python 3drc.py take the same arguments.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    dataset_opt = argparse.ArgumentParser(add_help=False)
    dataset_opt.add_argument("-d", "--dataset", default=None, help="Dataset directory, e.g. data/<scene> (default: scene resolved by configs/default_config.sh)")

    sub = parser.add_subparsers(dest="step", metavar="COMMAND")
    sub.add_parser("data", parents=[dataset_opt], help="Step 0b: YUV in-memory data load")

    sfm = sub.add_parser("sfm", parents=[dataset_opt], help="Step 1: camera pose estimation")
    sfm.add_argument("method", nargs="?", choices=SFM_METHODS, help="SfM method (default: SFM_METHOD in the config)")

    train = sub.add_parser("train", parents=[dataset_opt], help="Step 2: Gaussian training")
    train.add_argument("backend", nargs="?", default="3dgs", help=f"Training backend: {', '.join(backends)} (default: 3dgs)")
    for backend in backends.values():
        if backend.shortcut:
            sub.add_parser(backend.name, parents=[dataset_opt], help=f"{backend.header} (same as: train {backend.name})")

    view = sub.add_parser("view", parents=[dataset_opt], help="COLMAP GUI visualization")
    view.add_argument("type", nargs="?", default="hloc", choices=VIEW_TYPES, help="Reconstruction to show (default: hloc)")

    sugar = sub.add_parser("sugar", parents=[dataset_opt], help="Step 3: SuGaR mesh reconstruction")
    sugar.add_argument("--source-model", default=None, help="Source model under outputs/<scene>/ (default: 3dgs/inria_30k, else the only 3DGS model)")
    sub.add_parser("milo", parents=[dataset_opt], help="Step 3b: MILo mesh reconstruction")
    tsdf = sub.add_parser("tsdf", aliases=["mesh_tsdf"], parents=[dataset_opt], help="Step 3c: TSDF mesh extraction")
    tsdf.add_argument("--source-model", default=None, help="Source model under outputs/<scene>/ (default: 2dgs)")
    sub.add_parser("grouping", parents=[dataset_opt], help="Step 4: Gaussian Grouping segmentation")

    # eval and results forward their arguments unparsed; see FORWARDED
    sub.add_parser("eval", add_help=False, help="Step 5: mesh evaluation, arguments of src.mesh.eval_mesh (--mesh M [--gt G] ...)")

    nvs = sub.add_parser("nvs", parents=[dataset_opt], help="Step 5b: NVS evaluation on held-out views")
    nvs.add_argument("--model", default=None, help="Model path or directory under outputs/<scene>/ (default: 3dgs/inria_30k)")

    sub.add_parser("results", add_help=False, help="Aggregate SfM/NVS/mesh/timing results, arguments of collect_results.py ([--publish PATH])")
    sub.add_parser("outputs", help="Inspect generated scene artifacts")

    pipeline = sub.add_parser("pipeline", parents=[dataset_opt], help="SfM -> train -> [NVS] -> mesh -> mesh eval -> results")
    pipeline.add_argument("--method", default=None, choices=SFM_METHODS, help="SfM method (default: SFM_METHOD in the config)")
    pipeline.add_argument("--train", default=None, help="Training backend (default: PIPELINE_TRAIN or 3dgs)")
    pipeline.add_argument("--mesh", default=None, help=f"Mesh engine: {', '.join(MESH_ENGINES)} (default: PIPELINE_MESH or sugar)")
    pipeline.add_argument("--gt", default=None, help="GT point cloud (default: PIPELINE_GT, else <dataset>/lidar.pcd or lidar.ply)")
    return parser


def dispatch(args: argparse.Namespace, backends: Dict[str, Backend]) -> None:
    step = "tsdf" if args.step == "mesh_tsdf" else args.step

    if step == "outputs":
        Runner().run([python_bin(), "src/utils/inspect_outputs.py"])
        return
    if step == "pipeline":
        run_pipeline(args, backends)
        return

    backend_name = args.backend if step == "train" else step if step in backends else None
    if backend_name is not None and backend_name not in backends:
        raise CliError(f"Unknown training backend: '{backend_name}' (expected: {', '.join(backends)})")

    config = load_config(args.dataset)
    dataset = args.dataset or config["DATA_DIR"]
    run = Runner(Path(dataset).name)

    if step == "data":
        header("Step 0b: YUV In-Memory Data Load")
        run.script("00b_data_yuv.sh", dataset)
    elif step == "sfm":
        method = args.method or config["SFM_METHOD"]
        header(f"Step 1: Camera Pose Estimation (Method: {method})")
        run.script("01_sfm_hloc.sh", method, dataset)
    elif backend_name is not None:
        backend = backends[backend_name]
        header(backend.header)
        run.script(backend.script, dataset)
    elif step == "view":
        print(f"[3DRC CLI] Launching Reconstruction Viewer (Type: {args.type})...", flush=True)
        run.script("utils/view_reconstruction.sh", args.type, dataset)
    elif step == "sugar":
        print("[3DRC CLI] Executing Step 3: SuGaR Mesh Reconstruction...", flush=True)
        run.script("03_train_sugar.sh", dataset, *([args.source_model] if args.source_model else []))
    elif step == "milo":
        print("[3DRC CLI] Executing Step 3b: MILo Mesh Reconstruction...", flush=True)
        run.script("03b_train_milo.sh", dataset)
    elif step == "tsdf":
        print("[3DRC CLI] Executing Step 3c: TSDF Mesh Extraction...", flush=True)
        run.script("03c_mesh_tsdf.sh", dataset, *([args.source_model] if args.source_model else []))
    elif step == "grouping":
        print("[3DRC CLI] Executing Step 4: Gaussian Grouping...", flush=True)
        run.script("04_train_grouping.sh", dataset)
    elif step == "nvs":
        header("Step 5b: NVS Evaluation")
        run.script("05_eval_nvs.sh", dataset, *([args.model] if args.model else []))


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main CLI entrypoint for 3DRC package execution."""
    backends = load_backends()
    parser = build_parser(backends)
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["help"]:
        argv = []
    if argv[:1] and argv[0] in FORWARDED:
        title, target = FORWARDED[argv[0]]
        if title:
            print(f"[3DRC CLI] {title}", flush=True)
        Runner().run([python_bin(), *target, *argv[1:]])
        return
    args = parser.parse_args(argv)
    if not args.step:
        parser.print_help()
        return
    try:
        dispatch(args, backends)
    except CliError as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
