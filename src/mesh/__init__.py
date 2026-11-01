"""3D Mesh extraction, cleaning, and quantitative evaluation module."""

import importlib
from typing import TYPE_CHECKING

# Submodules are imported lazily so the NumPy-only helpers (alignment, metrics)
# can be used without Open3D or Trimesh installed.
_EXPORTS = {
    "clean_mesh": "clean_milo_mesh",
    "run_stereo_fusion": "dense_fusion",
    "compute_mesh_hygiene_and_topology": "eval_mesh",
    "evaluate_gt_accuracy": "eval_mesh",
}

# Type checkers cannot follow __getattr__, so they read the exports from these imports instead.
if TYPE_CHECKING:
    from .clean_milo_mesh import clean_mesh
    from .dense_fusion import run_stereo_fusion
    from .eval_mesh import compute_mesh_hygiene_and_topology, evaluate_gt_accuracy

__all__ = ["clean_mesh", "run_stereo_fusion", "compute_mesh_hygiene_and_topology", "evaluate_gt_accuracy"]

def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
