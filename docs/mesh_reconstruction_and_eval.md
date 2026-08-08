# 3D Mesh Reconstruction and Quantitative Evaluation Guide

This guide details the 3D mesh extraction pipelines (SuGaR and MILo) and the 3-axis quantitative evaluation tool (`src/eval_mesh.py`).

## 3D Mesh Extraction Engines

### SuGaR Engine

- Script: `./scripts/03_train_sugar.sh`
- Output Directory: `outputs/sugar_mesh/`
- Characteristics: High-density UV-textured OBJ polygon meshes.
- Key Parameters:
  - Regularization: `-r sdf` (sharp solid surface) or `-r dn_consistency`
  - Surface Level: `--surface_level 0.5`
  - Postprocessing: `--postprocess_mesh True`

### MILo Engine

- Script: `./scripts/03b_train_milo.sh`
- Output Directory: `outputs/milo_mesh/`
- Characteristics: Differentiable mesh-in-the-loop optimization producing compact collision meshes with 10x fewer vertices (100k vs 1M).

## Quantitative Evaluation Tool (`src/eval_mesh.py`)

### Evaluation Metrics

1. Topology Hygiene & Quality:
   - Watertight: Surface closure status (`mesh.is_watertight`).
   - Manifold: Winding consistency and manifold edge status.
   - Degenerate Faces: Count of zero-area invalid faces.
   - Sliver Ratio: Percentage of faces with min angle $< 5^\circ$ (critical for PhysX/Isaac Sim stability).

2. Geometric Accuracy (against LiDAR GT):
   - Sim(3) 7-DoF Alignment: Scale + Rigid alignment via `TransformationEstimationPointToPoint(with_scaling=True)`.
   - Accuracy (Mesh -> GT): Mean and RMSE distance from mesh surface to nearest GT points.
   - Completeness (GT -> Mesh): Mean and RMSE distance from GT points (within coverage mask) to mesh surface.
   - Chamfer L1: Average of Accuracy and Completeness.
   - 95% Hausdorff Distance: 95th percentile error bound.

### Evaluation Execution Commands

```bash
# 1. Evaluate mesh topology hygiene standalone
python src/eval_mesh.py --mesh outputs/sugar_mesh/.../mesh.obj

# 2. Compute and save initial Sim(3) 7-DoF transform against LiDAR GT
python src/eval_mesh.py --mesh baseline.obj --gt data/lidar.pcd --compute-transform

# 3. Reuse precomputed Sim(3) transform for fair 1:1 comparison
python src/eval_mesh.py --mesh outputs/sugar_mesh/.../sugar.obj --gt data/lidar.pcd --transform outputs/eval/sim3_transform.json
python src/eval_mesh.py --mesh outputs/milo_mesh/.../milo.obj --gt data/lidar.pcd --transform outputs/eval/sim3_transform.json
```
