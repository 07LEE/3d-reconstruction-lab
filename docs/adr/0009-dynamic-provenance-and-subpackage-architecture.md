# 0009. Dynamic Provenance Tracking and Subpackage Architecture

Status: Accepted (2026-08-10)

Context:
3DRC evaluates arbitrary combinations of SfM, Gaussian optimization, and Mesh extraction. Previously, `src/` was flat and scripts assumed fixed upstream paths (e.g., TSDF always from 2DGS), preventing multi-model experimentation and leaving outputs without provenance.

Decision:

1. Subpackage Modularization: Reorganized `src/` into 5 functional packages (`sfm/`, `prep/`, `mesh/`, `viz/`, `utils/`, `cli/`) with explicit exports.
2. Open Model Arguments: Generalized mesh scripts (`03c_mesh_tsdf.sh`, `03_train_sugar.sh`) to accept arbitrary upstream model paths.
3. Dynamic Provenance (`pipeline_meta.json`): Recorded runtime model inputs and active SfM targets (`sparse/0`) automatically during execution.
4. Dynamic Inspector: Updated `inspect_outputs.py` to parse metadata and display live upstream lineage in `summary.md` without hardcoded rules.

Alternatives:

- Fixed pipeline string mappings: Rejected as it blocks cross-model comparisons (e.g. TSDF on PlanarGS).

Consequences:

- Enables arbitrary pipeline combinations with auditable data lineage in `summary.md`.

Related:

- 0007 (scene-based output hierarchy)
- 0008 (2d-gaussian-splatting integration)
