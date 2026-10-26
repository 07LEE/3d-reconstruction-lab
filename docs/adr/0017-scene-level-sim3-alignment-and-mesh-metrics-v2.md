# 0017. Scene-Level Sim(3) Alignment and Mesh Metrics v2

Status: Accepted (2026-10-26)

Context:
`eval_mesh.py` aligned each mesh to the LiDAR GT on its own. Three problems made the mesh comparison unreliable:

1. The ICP initialization matched centroids and bounding-box extents with an identity rotation. SfM frames are arbitrarily rotated relative to LiDAR, so ICP could settle in a wrong local minimum without any error; the only symptom was a large Chamfer value. The docs described the step as "Umeyama RANSAC", which the code did not do.
2. Every mesh was registered by ICP onto the GT, so each mesh was scored in its own frame and the alignment itself optimized the metric being measured.
3. Completeness dropped GT points farther than `--coverage-radius` (0.5 m) from the mesh before averaging, so missing regions lowered only `gt_coverage_pct` and not Chamfer. There was no F-score, the standard thresholded metric for this comparison (Tanks and Temples).

Decision:

1. The Sim(3) transform is estimated once per scene from the SfM sparse points the meshes were trained on, located through `active_sfm` / `source_dataset` in the mesh `pipeline_meta.json` (falling back to `data/<scene>/sparse/0`). It is cached in `outputs/<scene>/eval/alignment/sim3_<sfm>_<gt_stem>.json` with fingerprints of the sparse model and GT, and is recomputed only when either changes or `--compute-transform` is passed. `--align-ref mesh` keeps the per-mesh behavior for meshes without an SfM model.
2. Global initialization enumerates the 24 proper rotations mapping the source PCA axes onto the target PCA axes, refines each with scale-aware point-to-point ICP (Umeyama updates), and keeps the hypothesis with the highest fitness. A result is flagged `ambiguous` when a hypothesis rotated more than 10 degrees from the chosen one fits within 0.02 fitness, which happens in symmetric rooms. The implementation (`src/mesh/alignment.py`) uses NumPy / SciPy only, so it is unit tested in CI without Open3D or a GPU.
3. Metrics v2 (`src/mesh/metrics.py`, `metrics_version: 2` in `eval_results.json`): accuracy and completeness clip each distance at `--coverage-radius` instead of dropping it; F-score, precision, and recall are reported at `--fscore-threshold` (0.05 m) with a curve at 0.01 / 0.02 / 0.05 / 0.1 m. HD95 and GT coverage keep their definitions.

Consequences:
Meshes of one scene are compared in one frame that does not depend on any mesh. Chamfer values from v1 and v2 are not comparable: completeness now rises with holes, and accuracy no longer grows without bound from floaters. `collect_results.py` marks v1 rows with `(v1)` and the report asks for re-evaluation. The alignment adds about 17 s per scene on CPU (synthetic 100k-point GT, 4k sparse points) and is cached afterwards. The SfM sparse points must cover enough of the scene to constrain the alignment; low `fitness` or `ambiguous: true` in `eval_results.json` means the transform should be checked visually or supplied with `--transform`.
