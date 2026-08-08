# 3D Mesh Reconstruction and Quantitative Evaluation Guide

This guide details the 3D mesh extraction pipelines (SuGaR and MILo) and the 3-axis quantitative evaluation tool (`src/eval_mesh.py`).

## 3D Mesh Extraction Engines

### SuGaR Engine

- Script: `./scripts/03_train_sugar.sh`
- Output Directory: `outputs/<scene_name>/mesh/sugar/`
- Output Format: Textured Wavefront `.obj` mesh + `.mtl` material file + `.png` UV texture map.
- Characteristics: Surface-aligned 3D Gaussians bound to triangle faces. Ideal for UV-textured rendering.
  - Regularization: `-r dn_consistency` (default)
  - Postprocessing: `--postprocess_mesh True`

### MILo Engine

- Script: `./scripts/03b_train_milo.sh data/<scene_name>`
- Output Directory: `outputs/<scene_name>/mesh/milo/`
- Floater Filter Script: `python src/clean_milo_mesh.py --input outputs/<scene_name>/mesh/milo/mesh_learnable_sdf.ply --output outputs/<scene_name>/mesh/milo/mesh_cleaned_largest.ply`
- Characteristics: Differentiable mesh-in-the-loop optimization using Marching Tetrahedra SDF isosurface extraction. Requires undistorted PINHOLE input dataset.

## Quantitative Evaluation Tool (`src/eval_mesh.py`)

### Evaluation Metrics

1. Topology Hygiene & Quality:
   - Watertight: Surface closure status (`mesh.is_watertight`).
   - Manifold: Winding consistency and manifold edge status.
   - Self-Intersection Count: Number of intersecting triangles.
   - Euler Characteristic & Genus: $\chi = V - E + F$.
2. Geometry Precision & Accuracy:
   - Chamfer Distance (CD): Symmetric 3D point cloud distance (2M points uniform sampling).
   - Hausdorff Distance (HD): Maximum surface deviation metric.

### Command Line Usage

```bash
# 1. Evaluate topology hygiene and self-intersections of a reconstructed mesh
python src/eval_mesh.py --mesh outputs/20260429_140922/mesh/sugar/mesh.obj

# 2. Evaluate Chamfer / Hausdorff distance against ground truth point cloud (e.g., LiDAR PCD)
python src/eval_mesh.py --mesh outputs/20260429_140922/mesh/sugar/mesh.obj --gt data/lidar.pcd

# 3. Evaluate mesh accuracy with SIM3 alignment
python src/eval_mesh.py --mesh outputs/20260429_140922/mesh/sugar/mesh.obj --gt data/lidar.pcd --transform outputs/20260429_140922/eval/sim3_transform.json
python src/eval_mesh.py --mesh outputs/20260429_140922/mesh/milo/mesh_cleaned_largest.ply --gt data/lidar.pcd --transform outputs/20260429_140922/eval/sim3_transform.json
```
