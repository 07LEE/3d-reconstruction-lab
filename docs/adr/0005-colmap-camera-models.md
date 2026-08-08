# 0005. Expand COLMAP Camera Model Support (SIMPLE_RADIAL / RADIAL / OPENCV)

Status: Accepted (2026-08-07)

Context:
Inria 3DGS readColmapCameras only supported PINHOLE and SIMPLE_PINHOLE models, aborting on other models. Because COLMAP/hloc defaults to SIMPLE_RADIAL for smartphone video sequences, dataset loading failed.

Decision:
Add parser branches for SIMPLE_RADIAL, RADIAL, and OPENCV in dataset_readers.py (Patch 0001).

Alternatives:

- Pre-undistorting images via COLMAP image_undistorter: Rejected due to extra storage, resolution cropping, and loss of original camera field-of-view.

Consequences:
Distortion parameters are approximated via effective focal length f = (f_x + f_y)/2 pinhole models rather than non-linear projection during rasterization.

Verification:
Successfully parsed 1,112 SIMPLE_RADIAL cameras, initialized 219,372 point cloud points, and completed training pass (Loss 0.1384535).
