# 0004. Manage Submodule Changes via patches/ Instead of Fork Branches

Status: Accepted (2026-08-07)

Context:
Camera model patches existed only on a remote fork branch (3drc-patch), while gitlink in the parent repository pointed to upstream commit 54c035f. Running git checkout wiped local patches without indication because .gitmodules ignore = dirty suppressed working tree diffs.

Decision:
Track all submodule modifications as versioned patch files under patches/<submodule>/NNNN-<name>.patch in the parent repository. Apply via apply_patches.sh and verify via verify_patches.sh (which returns exit 1 on failure).

Alternatives:

- Fork branch + gitlink pinning: Rejected due to silent patch loss if gitlinks are untracked or reset.

Consequences:
Upstream file updates require manual patch rebasing.

Patch Index:

- gaussian-splatting/0001-colmap-camera-models.patch (See ADR-0005)
- gaussian-splatting/0002-distcuda2-scipy-fallback.patch
- diff-gaussian-rasterization/0001-zero-init-state-structs.patch
- SuGaR gs_model.py CPU data_device OOM patch
- MILo nvdiffrast/torch/ops.py TORCH_CUDA_ARCH_LIST initialization fix

Related:

- 0001 (split Conda environments — patch series enforces module correctness)
- 0006 (complete removal of personal fork repositories)
