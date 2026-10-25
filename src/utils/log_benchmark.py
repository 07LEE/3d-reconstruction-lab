from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from functools import wraps
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_FILE = PROJECT_ROOT / "outputs" / "benchmark_timings.json"

def _run_quiet(cmd: list[str]) -> Optional[str]:
    """Runs a command and returns stripped stdout, or None on any failure."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip() or None
    except Exception:
        return None

def collect_env_fingerprint(project_root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Collects the environment fingerprint required by the ADR rules for performance numbers.

    Every probe is optional: missing tools (nvidia-smi, git) yield None instead of failing.

    Args:
        project_root: Repository root used for the git commit lookup.

    Returns:
        Dict with GPU, driver, torch build, conda env, and repository commit information.
    """
    gpu_name, driver = None, None
    smi = _run_quiet(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if smi:
        first = smi.splitlines()[0].split(",")
        gpu_name = first[0].strip()
        driver = first[1].strip() if len(first) > 1 else None

    try:
        torch_version: Optional[str] = importlib_metadata.version("torch")
    except importlib_metadata.PackageNotFoundError:
        torch_version = None

    commit = _run_quiet(["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"])
    dirty = _run_quiet(["git", "-C", str(project_root), "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=dirty"])

    return {
        "gpu": gpu_name,
        "driver": driver,
        "torch": torch_version,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "git_commit": commit,
        "git_dirty": bool(dirty) if commit else None,
    }

def _normalize_output_dir(output_dir: Optional[str | Path], project_root: Path = PROJECT_ROOT) -> Optional[str]:
    """Expresses an output directory relative to the project root for stable joins.

    Both the logical path (symlinks kept, as shell scripts see it) and the resolved path are tried,
    so a symlinked project directory still yields the same relative key.
    """
    if not output_dir:
        return None
    path = Path(output_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    logical = Path(os.path.normpath(path))
    for candidate, root in ((logical, project_root), (logical.resolve(), project_root.resolve())):
        try:
            return candidate.relative_to(root).as_posix()
        except ValueError:
            continue
    return logical.as_posix()

def record_stage_timing(stage_name: str, dataset_name: str, elapsed_seconds: float, metadata: Optional[Dict[str, Any]] = None, benchmark_file: Path = BENCHMARK_FILE, output_dir: Optional[str | Path] = None, fingerprint: bool = True, project_root: Path = PROJECT_ROOT) -> None:
    """Logs pipeline stage execution time and hardware metadata to benchmark JSON.

    Args:
        stage_name: Name of the pipeline stage (e.g., 'sfm_hloc', '3dgs_train', 'milo_train').
        dataset_name: Dataset or scene identifier name.
        elapsed_seconds: Elapsed execution duration in seconds.
        metadata: Optional metadata dictionary with hardware/hyperparameter details.
        benchmark_file: Path to target benchmark JSON file.
        output_dir: Output directory produced by the stage, used to join timings with results.
        fingerprint: If True, attaches the environment fingerprint (GPU, torch, commit).
        project_root: Root that output_dir is expressed relative to (relative inputs resolve against the cwd).

    Returns:
        None
    """
    benchmark_file.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    if benchmark_file.is_file():
        try:
            with open(benchmark_file, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception:
            entries = []

    readable_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage_name,
        "dataset": dataset_name,
        "output_dir": _normalize_output_dir(output_dir, project_root),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "formatted_time": readable_time,
        "metadata": metadata or {},
        "env": collect_env_fingerprint() if fingerprint else {},
    }

    entries.append(entry)
    tmp_file = benchmark_file.with_suffix(".json.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    os.replace(tmp_file, benchmark_file)

    print(f"[Benchmark Log] Recorded {stage_name} for '{dataset_name}': {elapsed_seconds:.2f}s ({readable_time})")

def log_execution_time(stage_name: str) -> Callable:
    """Python decorator to automatically measure function duration and record to benchmark log.

    Args:
        stage_name: Pipeline stage name identifier.

    Returns:
        Callable: Wrapped decorated function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            # Infer dataset name from first positional string/Path arg if available
            dataset_name = "default_dataset"
            for arg in args:
                if isinstance(arg, (str, Path)):
                    dataset_name = str(arg)
                    break

            record_stage_timing(stage_name, dataset_name, elapsed, {"function": func.__name__})
            return result
        return wrapper
    return decorator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record pipeline benchmark execution timing.")
    parser.add_argument("--stage", required=True, help="Stage name (e.g. sfm_hloc, 3dgs_train, milo_train)")
    parser.add_argument("--dataset", required=True, help="Dataset name or path")
    parser.add_argument("--seconds", type=float, required=True, help="Elapsed time in seconds")
    parser.add_argument("--meta", type=str, default="{}", help="Metadata JSON string")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory produced by the stage")
    parser.add_argument("--file", type=str, default=str(BENCHMARK_FILE), help="Benchmark JSON file path")
    parser.add_argument("--project-root", type=str, default=None, help="Logical project root used to relativize --output-dir")
    parser.add_argument("--set", dest="kv", action="append", default=[], metavar="KEY=VALUE", help="Metadata field (repeatable); VALUE is parsed as JSON when possible")

    args = parser.parse_args()
    meta_dict = json.loads(args.meta) if args.meta else {}
    for item in args.kv:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            parser.error(f"--set expects KEY=VALUE, got: {item!r}")
        try:
            meta_dict[key] = json.loads(raw)
        except json.JSONDecodeError:
            meta_dict[key] = raw
    project_root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    record_stage_timing(args.stage, args.dataset, args.seconds, meta_dict, benchmark_file=Path(args.file), output_dir=args.output_dir, project_root=project_root)
