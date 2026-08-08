import os
import json
import time
from datetime import datetime
from pathlib import Path

BENCHMARK_FILE = Path("outputs/benchmark_timings.json")

def record_stage_timing(stage_name: str, dataset_name: str, elapsed_seconds: float, metadata: dict = None):
    """Logs pipeline stage execution time and hardware metadata to outputs/benchmark_timings.json."""
    BENCHMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    entries = []
    if BENCHMARK_FILE.exists():
        try:
            with open(BENCHMARK_FILE, "r") as f:
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
        "metadata": metadata or {}
    }
    
    entries.append(entry)
    with open(BENCHMARK_FILE, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"[Benchmark Log] Recorded {stage_name} for '{dataset_name}': {elapsed_seconds:.2f}s ({readable_time})")

def log_execution_time(stage_name: str):
    """Python @decorator to automatically record function execution time and log to JSON."""
    def decorator(func):
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Infer dataset name from first positional string arg if available
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
    import argparse
    parser = argparse.ArgumentParser(description="Record pipeline benchmark execution timing.")
    parser.add_argument("--stage", required=True, help="Stage name (e.g. sfm_hloc, 3dgs_train, milo_train)")
    parser.add_argument("--dataset", required=True, help="Dataset name or path")
    parser.add_argument("--seconds", type=float, required=True, help="Elapsed time in seconds")
    parser.add_argument("--meta", type=str, default="{}", help="Metadata JSON string")
    
    args = parser.parse_args()
    meta_dict = json.loads(args.meta)
    record_stage_timing(args.stage, args.dataset, args.seconds, meta_dict)
