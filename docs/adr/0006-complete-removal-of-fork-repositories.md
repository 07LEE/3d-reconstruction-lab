# 0006. Complete Removal of Personal Fork Repositories in Favor of Direct Upstream Submodules and Patches

Status: Accepted (2026-08-08)

Context:
Submodule modifications (such as COLMAP camera model extensions, KDTree fallback, and C++ rasterizer zero-initialization) previously relied on custom remote fork commits or branches. This introduced external dependencies on personal GitHub fork repositories and posed risks of untracked divergence or patch loss when submodule pointers were reset.

Decision:
Configure all submodules under `third_party/` (including `third_party/gaussian-splatting`) to reference official upstream repositories (e.g., `graphdeco-inria/gaussian-splatting.git`) directly at official main or release commits. Completely eliminate personal fork repository dependencies. Maintain all custom project modifications as versioned `.patch` files under `patches/third_party/` and apply them automatically via `scripts/utils/apply_patches.sh`.

Alternatives:

- Fork branch + gitlink pinning: Rejected due to external repository maintenance overhead and silent patch loss risks.

Consequences:
Personal fork repositories can be safely deleted on GitHub. Upstream updates to submodule repositories require manual patch rebasing when updating submodule pointers.

Patch Index:

- gaussian-splatting/0001-colmap-camera-models.patch
- gaussian-splatting/0002-distcuda2-scipy-fallback.patch
- diff-gaussian-rasterization/0001-zero-init-state-structs.patch
- diff-gaussian-rasterization/0002-cstdint-include.patch
- sugar/0001-sugar-extension-dup-and-cpu-device.patch
- milo/0001-cstdint-and-cmake-fixes.patch

Related:

- 0004 (submodule patches over fork branches)
