# 0019. Defer In-Memory YUV Ingestion, Use Disk Extraction as Interim Path

Status: Accepted (2026-08-22)

Context:
`src/data/yuv_pipeline.py` (`YUVInMemoryDataset`, `YUVInMemoryPipelineManager`) wraps the
`yuv_sensor` package to decode a raw YUV_420_888 capture session directly into normalized
PyTorch tensors in memory, and is reachable via the CLI (`3drc.py data`). Its docstring states
the goal of "zero-disk-writing" ingestion straight into the SfM / 3DGS pipelines, but nothing
downstream currently consumes it.

`src/sfm/hloc_pipeline.py`'s `run_hloc_pipeline()` requires a real on-disk `image_dir` and
iterates files directly (`third_party/hloc`'s own feature extraction also reads images from a
directory path, not from in-memory tensors). Every training backend
(`gaussian-splatting`, `MILo`, `SuGaR`, `Scaffold-GS`, `Mip-Splatting`) loads images through a
COLMAP-style dataset reader that expects `images/` + `sparse/` to already exist on disk.
Wiring in-memory tensors into any of these means patching each submodule's loader individually
— there is no single central integration point.

Decision:
For now, keep the existing disk-extraction path as the only supported way to feed a raw YUV
session into the pipeline: run the `yuv-sensor` CLI (`yuv-sensor --session_dir ... [--undistort]`)
to extract frames to `extracted_frames_raw` / `extracted_frames_undistorted`, then point `raw_images`
at that output for the normal `sfm` / `train` steps. Full in-memory ingestion is deferred, not
abandoned.

If revisited, it requires (per ADR-0004, via `patches/`) at minimum:

1. Patching `third_party/hloc`'s feature-extraction image loading to accept an in-memory
   tensor/array source instead of a directory listing.
2. Patching each training submodule's COLMAP-based dataset reader to accept the same
   in-memory source (`gaussian-splatting`, `MILo`, `SuGaR`, `Scaffold-GS`, `Mip-Splatting`
   each have their own loader — this is per-submodule work, not one change).

Consequences:
`yuv_pipeline.py`'s "zero-disk-writing" docstring describes the target design, not current
behavior — noted here so a future maintainer doesn't assume it's already wired up. Disk
extraction remains the supported path until the loader patches above land.
