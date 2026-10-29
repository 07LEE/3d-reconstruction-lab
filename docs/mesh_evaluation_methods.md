---
title: Mesh Evaluation Methods in the Literature
description: How benchmarks and reconstruction papers evaluate meshes, how their ground truth was obtained, and how 3DRC compares.
category: guide
last_updated: 2026-10-26
---

# Mesh Evaluation Methods in the Literature

This note records, paper by paper, how meshes are evaluated: where the ground truth (GT) comes from, how the reconstruction is brought into the GT frame, which region is scored, how surfaces are sampled, and which metrics and thresholds are reported. It ends with a comparison against `src/mesh/eval_mesh.py` (metrics v2, [ADR 0017](adr/0017-scene-level-sim3-alignment-and-mesh-metrics-v2.md)).

Provenance markers:

- [code]: read from the official evaluation code at the commit listed in [Sources](#sources).
- [paper]: stated in the paper or dataset description and not re-checked against code for this note.

## Common Evaluation Pipeline

Every protocol below is a choice at each of five steps. Most differences between reported numbers come from steps 2-4, not from the metric formula.

1. GT acquisition: laser scanner, structured light, RGB-D fusion, or synthetic mesh.
2. Frame alignment: GT camera poses (no alignment), trajectory-based Sim(3), ICP, or a provided transform.
3. Region restriction: crop volume, observation mask, camera-frustum and occlusion culling, or free-space modeling.
4. Sampling: uniform surface samples, then optionally voxel downsampling so point density does not weight the score.
5. Metrics: accuracy (prediction to GT), completeness (GT to prediction), Chamfer L1, precision / recall / F-score at a threshold, normal consistency.

## Protocol Summary

| Protocol | GT | Alignment | Region Restriction | Sampling | Metrics (threshold) |
| --- | --- | --- | --- | --- | --- |
| DTU | Structured-light scans [paper] | None: GT camera calibration | Observation mask grid + bounding box; GT above ground plane [code] | Mesh sampled at 0.2 mm, radius downsample [code] | Accuracy, completeness, Chamfer; distances over 20 mm dropped [code] |
| Tanks and Temples | Industrial laser scanner (FARO Focus 3D X330) [paper] | Camera-trajectory RANSAC Sim(3), then 3 ICP stages with scaling [code] | Per-scene crop polygon volume [code] | Voxel downsample at tau / 2 [code] | Precision, recall, F-score at per-scene tau (3 mm to 25 mm) [code] |
| ETH3D | Registered laser scans [code] | GT poses from scan alignment [code] | Laser-beam free-space modeling ignores unobservable space [code] | Scores normalized per 1 cm voxel [code] | Accuracy, completeness, F1 at 1, 2, 5, 10, 20, 50 cm [code] |
| ScanNet (Atlas protocol) | BundleFusion RGB-D mesh `vh_clean_2` [code] | None: dataset poses | Prediction re-fused from depth rendered at GT poses (MonoSDF) [code] | Voxel downsample 2 cm [code] | Acc, comp, precision, recall, F-score at 5 cm [code] |
| Replica (NICE-SLAM protocol) | Synthetic high-quality scene meshes [paper] | Optional rigid ICP (no scale) [code] | GT culled to seen regions; prediction culled to frustums [code] | 200k surface samples [code] | Accuracy, completion, completion ratio at 5 cm; depth L1 [code] |
| MuSHRoom | Laser-scanner reference mesh (FARO) [code, paper] | Provided ICP transform per device [code] | Frustum + occlusion (2 cm) + valid sensor depth; cut to GT extent [code] | Area-proportional surface samples [code] | Acc, comp, precision, recall, F-score at 5 cm, normal consistency [code] |

## Benchmark Protocols

### DTU MVS (Aanæs et al., IJCV 2016)

- GT: structured-light scans of tabletop objects captured with a robot-mounted rig; cameras are calibrated, so reconstructions are already in the GT frame [paper].
- Region: points outside the per-scan observation mask (`ObsMask`, a voxel grid of observed space with bounding box) are removed from the prediction; for completeness, only GT points above the table plane (`Plane.mat`) are used [code].
- Sampling: each triangle is sampled on a grid at 0.2 mm spacing, the points are shuffled, and points closer than 0.2 mm to a kept point are removed [code].
- Metrics: accuracy = mean prediction-to-GT distance, completeness = mean GT-to-prediction distance, Chamfer = their average, in mm. Distances of 20 mm or more are excluded from both means (`max_dist`), so outliers and holes are not penalized beyond that [code].
- Method papers on Gaussian splatting (2DGS, GOF, PGSR, MILo) additionally cull the mesh with object masks dilated by a 24 px disk before running this script, following UNISURF [code, 2DGS `evaluate_single_scene.py`].

### Tanks and Temples (Knapitsch et al., SIGGRAPH 2017)

- GT: laser scans with an industrial scanner (FARO Focus 3D X330, about 2 mm accuracy), registered through overlap; GT for the training scenes is public, test scenes are scored on the website [paper].
- Alignment [code, `run.py`, `registration.py`]:
  1. The submitted camera trajectory is registered to the reference COLMAP trajectory (already aligned to GT) by RANSAC on camera-center correspondences with a Sim(3) estimate.
  2. The reconstruction is refined against the GT with three ICP stages (point-to-point with scaling) inside the crop volume: voxel tau with threshold 80 tau, voxel tau / 2 with threshold 20 tau, then uniform with threshold 2 tau.
- Region: a hand-made polygon crop volume per scene (`<scene>.json`) is applied to both clouds [code].
- Sampling: both clouds are voxel-downsampled at tau / 2 [code].
- Metrics: precision (prediction points within tau of GT), recall (GT points within tau of prediction), F-score. tau per scene: Ignatius 3 mm, Caterpillar and Truck 5 mm, Barn and Meetingroom 10 mm, Church and Courthouse 25 mm [code, `config.py`].
- Mesh methods (2DGS, GOF, PGSR, MILo) cull the mesh by depth-rendered visibility from the training cameras before evaluation [code, 2DGS `eval_tnt/cull_mesh.py`]; MILo states it follows the GOF and RaDe-GS pipeline [code, README].

### ETH3D (Schöps et al., CVPR 2017)

- GT: multiple laser scans registered into one frame (MeshLab project with scan poses); images are registered to the scans [code, paper].
- Region: the scanner's beams are modeled (start radius, divergence) to determine which space was observed; reconstruction points in space the scanner could not observe do not count against accuracy [code, README].
- Sampling: scores are normalized per voxel (default 1 cm) so dense regions do not dominate [code].
- Metrics: accuracy, completeness, and F1 at tolerances 1, 2, 5, 10, 20, 50 cm [code].

### ScanNet with the Atlas Protocol (Murez et al., ECCV 2020)

- GT: the ScanNet mesh fused from RGB-D with BundleFusion (`vh_clean_2`); it is incomplete and noisy compared to laser GT [code, MonoSDF `scannet_eval`].
- Alignment: none; reconstructions use dataset poses [code].
- Sampling: both meshes' vertices voxel-downsampled at 2 cm [code, Atlas `evaluation.py`].
- Metrics: accuracy (`dist1`), completeness (`dist2`), precision, recall, F-score at 5 cm [code].
- Region (MonoSDF variant): the predicted mesh is rendered to depth at every GT camera pose and re-fused with TSDF (1 cm voxel), which removes geometry no camera saw or that is occluded [code, MonoSDF `refuse`].

### Replica with the NICE-SLAM Protocol (Zhu et al., CVPR 2022)

- GT: Replica scene meshes (Straub et al., 2019) [paper]. NICE-SLAM distributes GT meshes with unseen regions already culled [code, README].
- Alignment: optional ICP between prediction and GT, rigid only [code, `eval_recon.py`].
- Region: predicted mesh vertices outside every camera frustum are removed [code, `cull_mesh.py`].
- Metrics: accuracy and completion (cm), completion ratio (percentage of GT within 5 cm), and a 2D depth L1 over 1000 random views that avoid unseen regions [code]. MonoSDF's Replica script adds precision, F-score, and normal consistency from 200k samples [code].

### MuSHRoom (Ren et al., 2023)

- GT: a reference mesh from a FARO laser scanner; images and depth from consumer devices (iPhone, Kinect) [code: DN-Splatter `eval_faro.py` "reference faro scanner"; paper].
- Alignment: a provided ICP transform per device (`icp_iphone.json`, `icp_kinect.json`) [code].
- Region [code, `evaluate_mesh.py`]:
  1. Prediction culled per camera: inside the frustum, not occluded (depth within 2 cm of the rendered depth), and with valid sensor depth.
  2. Prediction cut to the GT extent by projecting onto the xy, xz, and yz planes.
- Metrics: accuracy, completeness, precision, recall, F-score at 5 cm, and normal consistency (mean absolute dot product of matched face normals, averaged both ways) [code].
- Relevance to 3DRC: same setting (consumer camera images, laser-scanned room), so it is the closest reference protocol.

## Method Papers

### MonoSDF (Yu et al., NeurIPS 2022)

Monocular depth and normal priors for neural SDF. Ships evaluation scripts for ScanNet (Atlas metrics with depth re-fusion culling), Replica (NICE-SLAM metrics plus F-score and normal consistency), and DTU, and training configs for Tanks and Temples [code]. Its scripts are the source most indoor papers copy.

### 2D Gaussian Splatting (Huang et al., SIGGRAPH 2024)

DTU Chamfer with mask culling and the official DTU script; Tanks and Temples F-score with visibility culling and the official toolbox (pinned to Open3D 0.10.0 for comparability) [code, README]. Reports NVS on Mip-NeRF 360 separately, which is where 3DGS still scores higher.

### Gaussian Opacity Fields (Yu et al., SIGGRAPH Asia 2024) and PGSR (Chen et al., TVCG 2024)

Both reuse DTUeval-python and the Tanks and Temples toolbox [code, README]. Their numbers are directly comparable with 2DGS because the culling and evaluation scripts are shared.

### MILo (Guédon et al., 2025)

- Geometry: Tanks and Temples following GOF and RaDe-GS; DTU with scripts adapted from GOF and RaDe-GS, using regular-grid extraction for DTU [code, README].
- Mesh-based NVS [code, `eval/mesh_nvs/render_mesh_nvs.py`]: the mesh is frozen, a color field (TensoRF VM) is trained on the mesh surface from the training views (5000 iterations by default), and test views are rendered and scored with PSNR / SSIM / LPIPS. Every mesh gets the same appearance model, so the score reflects geometry rather than each method's texturing.
- Also reports vertex counts, since compactness is its claim [paper].

### DN-Splatter (Turkulainen et al., 2025)

Depth and normal priors for indoor Gaussian splatting, with data parsers for MuSHRoom, ScanNet++, and Replica [code]. Mesh metrics are accuracy, completeness, Chamfer L1, normal consistency, and F-score, computed after culling the predicted mesh to training-camera visibility; a MuSHRoom-specific variant of the script exists [code, `eval_instructions.md`, `eval_mesh_vis_cull.py`].

### SuGaR (Guédon and Lepetit, CVPR 2024)

The official evaluation script computes PSNR, SSIM, and LPIPS on held-out test images of the refined SuGaR model [code, README]; it has no GT geometry evaluation. Mesh quality is argued through rendering quality and editability.

## 3DRC Compared With These Protocols

| Step | 3DRC (`eval_mesh.py`, metrics v2) | Closest reference | Gap |
| --- | --- | --- | --- |
| GT | Own LiDAR point cloud | MuSHRoom, ETH3D, Tanks and Temples (laser) | - |
| Alignment | Sim(3) from SfM sparse points: 24 PCA hypotheses + scale-aware ICP, shared per scene | Tanks and Temples (trajectory Sim(3) + ICP) | Uses points instead of camera centers; no ICP refinement on each mesh within a crop |
| Region | None; GT and mesh used whole | MuSHRoom / DN-Splatter visibility culling, DTU masks, ETH3D free space | Largest gap: LiDAR areas no camera saw lower recall and completeness |
| Sampling | 2M uniform surface samples, no density normalization | Tanks and Temples voxel at tau / 2, Atlas 2 cm, ETH3D per-voxel | LiDAR density varies with range, so near-scanner regions weigh more |
| Metrics | Clipped accuracy / completeness, Chamfer, F-score at 5 cm with curve, HD95, coverage | MuSHRoom (5 cm F-score + normal consistency) | No normal consistency |
| Mesh NVS | None | MILo color-field protocol | Optional; would compare SuGaR, TSDF, MILo meshes on equal appearance |

Suggested order of work, by expected effect on comparability:

1. Visibility culling: remove mesh and GT parts outside every training camera's frustum or occluded (MuSHRoom procedure), so recall measures what the cameras could reconstruct.
2. Voxel downsampling of both sides before distances (for example at tau / 2), so LiDAR density does not weight the score.
3. Normal consistency, which separates flat-wall quality from object detail better than Chamfer in rooms.
4. Mesh-based NVS with a shared color field (MILo protocol), if appearance of exported meshes matters.

## Sources

Official repositories read for the [code] items (shallow clones on 2026-10-26):

| Repository | Commit | Files |
| --- | --- | --- |
| [isl-org/TanksAndTemples](https://github.com/isl-org/TanksAndTemples) | `2a0d1b25df` | `python_toolbox/evaluation/{run,registration,evaluation,config}.py` |
| [jzhangbs/DTUeval-python](https://github.com/jzhangbs/DTUeval-python) | `45b4fecbe0` | `eval.py`, `README.md` |
| [ETH3D/multi-view-evaluation](https://github.com/ETH3D/multi-view-evaluation) | `0daa4f4de0` | `README.md` |
| [magicleap/Atlas](https://github.com/magicleap/Atlas) | `bee3ddbdb0` | `atlas/evaluation.py` |
| [cvg/nice-slam](https://github.com/cvg/nice-slam) | `7af15cc337` | `src/tools/{eval_recon,cull_mesh}.py`, `README.md` |
| [autonomousvision/monosdf](https://github.com/autonomousvision/monosdf) | `12513009cd` | `scannet_eval/evaluate.py`, `replica_eval/{eval_recon,cull_mesh}.py` |
| [TUTvision/MuSHRoom](https://github.com/TUTvision/MuSHRoom) | `3f36dbfe24` | `evaluate_mesh.py`, `README.md` |
| [maturk/dn-splatter](https://github.com/maturk/dn-splatter) | `97588b4290` | `dn_splatter/eval/{eval_instructions.md,eval_mesh_vis_cull.py}`, `dn_splatter/data/mushroom_utils/eval_faro.py` |
| [hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting) | `f3e3b9fa67` | `scripts/eval_dtu/evaluate_single_scene.py`, `scripts/eval_tnt/cull_mesh.py`, `README.md` |
| [autonomousvision/gaussian-opacity-fields](https://github.com/autonomousvision/gaussian-opacity-fields) | `5245b20e5d` | `README.md` |
| [zju3dv/PGSR](https://github.com/zju3dv/PGSR) | `de24f1a38b` | `README.md` |
| [Anttwo/MILo](https://github.com/Anttwo/MILo) | `0c38aa65f8` | `README.md`, `milo/eval/mesh_nvs/render_mesh_nvs.py` |
| [Anttwo/SuGaR](https://github.com/Anttwo/SuGaR) | `7c10c4ae4a` | `README.md` |

The [paper] items (scanner models, DTU capture rig, Replica origin, venues) come from the papers and were not re-verified here; the arXiv and benchmark websites were not reachable from the environment used to write this note.
