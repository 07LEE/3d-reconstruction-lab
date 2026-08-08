# 0001. Split Conda Environments by Component

Status: Accepted (2026-08-07)

Context:
In a single monolithic gs_original environment, SuGaR's bundled rasterizer (2023 release) displaced site-packages' Inria dr_aa rasterizer variant. Because both packages shared the name diff_gaussian_rasterization, they silently displaced each other, surfacing as an unexpected antialiasing keyword TypeError while source trees remained unmodified and invisible to git status.

Decision:
Partition execution into dedicated Conda environments (gs_train, gs_sugar, gs_group, gs_milo) and install component-specific rasterizers from submodules in editable mode (pip install -e .).
Migration path: Created gs_train by cloning baseline gs_original (conda create -n gs_train --clone gs_original) and replacing the rasterizer with third_party/gaussian-splatting/submodules/diff-gaussian-rasterization, while preserving dedicated environments for SuGaR, MILo, and Gaussian Grouping.

Alternatives:

- Single environment with strict installation ordering: Rejected due to package name collision; pip lacks package-level isolation so on any reinstall of either component, one silently overwrites the other.
- Fork submodules to rename packages: Rejected due to ongoing upstream maintenance costs.

Consequences:

- Editable installs bind module correctness to the submodule working tree state; environment isolation alone is insufficient without the patch series applied.
- Environment drift across 4 envs is the practical risk. Requires per-env lock files (envs/*.yml). Note: PyTorch 2.12.0.dev nightly builds can disappear from channels — pinning in yml alone may not restore the environment without local wheel caching.
- Mitigated duplicate CUDA module build time by storing pre-compiled wheels in ~/wheels/sm120/ (reducing re-installation from 40 minutes to seconds).

Verification:
verify_patches.sh inspects GaussianRasterizationSettings._fields for dr_aa presence. Verified cross-environment failure in gs_original (EXIT=1) vs clean pass in gs_train (EXIT=0).

Related: 0004 (patch series — editable installs depend on it)
