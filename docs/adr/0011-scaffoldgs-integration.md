# 0011. Scaffold-GS Integration for Anchor-Based Scene Reconstruction

Status: Accepted (2026-08-11)

Context:
Standard 3DGS uses unconstrained 3D ellipsoids. To evaluate anchor-guided Gaussian representation for view-dependent object detail within the 3DRC testbed, a Scaffold-GS training option is required.

Decision:
1. Add official `city-super/scaffold-gs` repository as a git submodule under `third_party/scaffold-gs`.
2. Add `--method scaffoldgs` to CLI (`src/cli/main.py`) and script routing (`scripts/run_3drc.sh`, `scripts/02d_train_scaffoldgs.sh`).
3. Configure `gs_scaffold` conda environment profile with fallback to `gs_train`.

Consequences:
Adds Scaffold-GS training backend to Stage 2 pipeline choices, requiring `wandb`, `einops`, `lpips`, `laspy`, `colorama`, and `gcc-12` host compiler.
