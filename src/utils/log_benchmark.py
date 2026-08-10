from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

BENCHMARK_FILE = Path("outputs/benchmark_timings.json")

def record_stage_timing(stage_name: str, dataset_name: str, elapsed_seconds: float, metadata: Optional[Dict[str, Any]] = None, benchmark_file: Path = BENCHMARK_FILE) -> None:
    """Logs pipeline stage execution time and hardware metadata to benchmark JSON.

    Args:
        stage_name: Name of the pipeline stage (e.g., 'sfm_hloc', '3dgs_train', 'milo_train').
        dataset_name: Dataset or scene identifier name.
        elapsed_seconds: Elapsed execution duration in seconds.
        metadata: Optional metadata dictionary with hardware/hyperparameter details.
        benchmark_file: Path to target benchmark JSON file.

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
        "elapsed_seconds": round(elapsed_seconds, 2),
        "formatted_time": readable_time,
        "metadata": metadata or {},
    }

    entries.append(entry)
    with open(benchmark_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

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

    args = parser.parse_args()
    meta_dict = json.loads(args.meta) if args.meta else {}
    record_stage_timing(args.stage, args.dataset, args.seconds, meta_dict)
