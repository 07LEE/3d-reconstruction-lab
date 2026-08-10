# 0010. Multi-Video Merged SfM via Hybrid Sequential-Retrieval Matching and Cache Integrity

Status: Accepted (2026-08-10)

Context:
Reconstructing a unified 3D collision mesh or Gaussian representation from multiple handheld/robot video trajectories of the same space is essential for robotic simulations. Previously, `hloc_pipeline.py` supported only `sequential` (frame index proximity) and `exhaustive` ($O(N^2)$) pairing. When multiple videos (`v1_*.png`, `v2_*.png`) were placed in a single dataset:

1. Sequential matching sorted alphabetically, generating zero pairs between separate videos except across arbitrary temporal boundaries, causing COLMAP to disconnect or discard trajectories.
2. Exhaustive pairing was computationally prohibitive for >2,000 frames (~2 million SuperGlue pairs).
3. H5 matching cache (`matches.h5`) was checked only by file existence, silently skipping matching when pairing strategy or overlap changed.

Decision:

1. Prefix-Aware Sequential Partitioning: Implemented `group_images_by_prefix` to group frames by video identifier (e.g. `v1_`, `v2_`) and generate sequential pairs strictly within each video trajectory, preventing cross-video boundary contamination.
2. NetVLAD Global Descriptor Retrieval: Added `sequential+retrieval` strategy in `hloc_pipeline.py`. Extracted global descriptors (`global_features.h5`) via NetVLAD to retrieve top-k ($k=30$) visually similar pairs across distinct video trajectories.
3. Canonical Pair Deduplication: Implemented `merge_and_deduplicate_pairs` normalizing pair ordering `(min(p1, p2), max(p1, p2))` and filtering redundant pairs between sequential and retrieval steps.
4. Pair Hash Cache Invalidation: Stored the SHA-256 hash of the generated pair list in `matches.pairs.sha256`. Re-executed `match_features` only when the pair list hash changed (`stale == True`), while safely reusing `features.h5`.
5. Shared Single Camera Intrinsics: Maintained `CameraMode.SINGLE` and `SIMPLE_RADIAL` across merged videos from the same capture device to constrain bundle adjustment degrees of freedom.
6. Provenance & Parameter Extraction: Extracted `SFM_STRATEGY`, `SFM_OVERLAP`, and `SFM_RETRIEVAL_K` to `configs/default_config.sh` and recorded video prefix counts in `sfm_info.json`.

Consequences:

- Enables scalable end-to-end multi-video merging into a single unified COLMAP sparse model and downstream 2DGS/MILo representation.
- Prevents silent cache invalidation bugs when switching pairing configurations.

Related:

- 0005 (colmap-camera-models)
- 0007 (scene-based-output-hierarchy)
- 0009 (dynamic-provenance-and-subpackage-architecture)
