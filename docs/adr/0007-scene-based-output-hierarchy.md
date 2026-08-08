# 0007. Scene-Based Output Hierarchy for Multi-Dataset Pipeline Isolation

Status: Accepted (2026-08-08)

Context:
Pipeline outputs previously used flat global directories under `outputs/` (e.g. `outputs/gs_final_precision`, `outputs/milo_mesh`, `outputs/sugar`). When switching datasets or re-running experiments across different indoor scenes, outputs from prior runs were silently overwritten or intermixed, making dataset-specific tracking and side-by-side comparison impossible.

Decision:
Adopt a scene-based output directory hierarchy (`outputs/<scene_name>/<stage>/`) for all pipeline stages:

- `outputs/<scene_name>/3dgs/inria_30k/`: Inria 3DGS 30k iteration reference checkpoints.
- `outputs/<scene_name>/3dgs/planargs/`: PlanarGS planar-regularized checkpoints.
- `outputs/<scene_name>/mesh/milo/`: MILo SDF isosurface meshes, floater-cleaned PLYs, and WebGL `.splat` files.
- `outputs/<scene_name>/mesh/sugar/`: SuGaR surface-aligned UV-textured OBJ meshes.
- `outputs/<scene_name>/eval/eval_results.json`: Quantitative topology hygiene and 2M-point LiDAR GT evaluation metrics.

`SCENE_NAME` is automatically inferred from the dataset directory basename (e.g., `undistorted`, `nerfstudio_data`, `kitchen`). Legacy flat output directories are preserved as fallbacks in inspection utilities (`src/inspect_outputs.py`).

Alternatives:

- Flat output directories (`outputs/gs_final_precision`): Rejected due to cross-dataset overwriting and artifact pollution.
- Manual output path overrides per script invocation: Rejected due to execution verbosity and maintenance friction across scripts.

Consequences:
Multiple datasets and indoor scenes can run concurrently without output collisions. Inspection tools (`./scripts/run_3drc.sh outputs` and `outputs/summary.md`) automatically organize and display artifacts grouped by scene.

Related:

- 0003 (decoupling buildtime and runtime configurations)
