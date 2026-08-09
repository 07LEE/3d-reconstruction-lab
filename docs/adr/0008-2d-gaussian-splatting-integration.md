# 0008. 2D Gaussian Splatting Integration for Unbiased Depth and TSDF Mesh Reconstruction

Status: Accepted (2026-08-09)

Context:
Standard 3D Gaussian Splatting (3DGS) models scene geometry using 3D ellipsoids with non-zero thickness along surface normals. In multi-view indoor reconstruction, this introduces view-dependent depth ambiguities and geometric collapse, leading to noisy, multi-layered, or floater-ridden meshes. While SuGaR extracts textured meshes at the cost of high vertex counts, and MILo extracts compact SDF collision meshes, neither provides direct TSDF integration from exact ray-splat intersections.

Decision:
Integrate 2D Gaussian Splatting (2DGS) into the 3DRC pipeline as an optional Stage 2 representation (`scripts/02c_train_2dgs.sh`) and Stage 3 mesh engine (`scripts/03c_mesh_tsdf.sh`):

1. Surfel Representation: Employs 2D oriented planar disks to restrict Gaussian thickness to zero, computing exact analytical ray-splat intersections to render unbiased depth and surface normal maps.
2. Volumetric TSDF Fusion: Renders depth/normal maps across viewpoints and integrates them via Open3D TSDF voxel volume (`outputs/<scene_name>/mesh/2dgs/tsdf_mesh.ply`).
3. Submodule & Native sm_120 Build: Added `third_party/2d-gaussian-splatting` as a submodule (`ignore = dirty`), patched COLMAP camera models (`SIMPLE_RADIAL` / `RADIAL` support), resolved missing `<cstdint>`, zero-initialized state structs, and fixed backward gradient tensor shape checks for SH coefficients under PyTorch 2.4 / NVCC 12.8.

Alternatives:

- Rely exclusively on PGSR: Rejected. 2DGS surfels compute analytical ray-plane intersections natively across all objects without imposing planar-only priors.
- Rely exclusively on SuGaR / Poisson reconstruction: Rejected due to excessive vertex counts and floater artifacts on low-texture walls.

Consequences:

- Provides clean, floater-free polygon meshes with crisp edge and wall preservation.
- Adds `diff-surfel-rasterization` module to the active environment audit in `scripts/utils/verify_patches.sh`.
- Requires `open3d` and `trimesh` in the runtime environment for TSDF integration and postprocessing.

Verification:

1. `scripts/utils/verify_patches.sh` audits native `sm_120` SASS CUBIN in `diff_surfel_rasterization` (`[ok]`).
2. End-to-end training and TSDF meshing validated on `data/test_multi`.

Related:

- 0002 (Blackwell sm_120 native CUBIN)
- 0004 (submodule patches over fork)
- 0005 (COLMAP camera models)
- 0007 (scene-based output hierarchy)
