"""3D Mesh extraction, cleaning, and quantitative evaluation module."""

import importlib

# Submodules are imported lazily so the NumPy-only helpers (alignment, metrics)
# can be used without Open3D or Trimesh installed.
_EXPORTS = {
    "clean_mesh": "clean_milo_mesh",
    "run_stereo_fusion": "dense_fusion",
    "compute_mesh_hygiene_and_topology": "eval_mesh",
    "evaluate_gt_accuracy": "eval_mesh",
}

__all__ = list(_EXPORTS)

def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
