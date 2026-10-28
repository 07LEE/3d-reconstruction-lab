# 0018. Unified Gaussian Environment with Per-Backend Rasterizer Builds

Status: Accepted (2026-10-27), opt-in until verified on hardware; Amends 0001

Context:
ADR 0001 split the Gaussian backends into six Conda environments (gs_train, gs_sugar, gs_group, gs_milo, gs_scaffold, gs_mipsplatting) because several packages install a CUDA extension under the same module name. Since ADR 0012 all six use Python 3.10 and torch 2.11.0+cu128 with nearly identical pip dependencies, so every environment repeats the same multi-GB torch/CUDA install and the same extension builds. Comparing the extension sources (2026-10-27):

- `simple_knn`: the copies in scaffold-gs, mip-splatting, milo, gaussian-grouping, and PlanarGS have identical sources (blob `e72e4c9`); patches only add `<cstdint>` / `<cfloat>` includes.
- `diff_gaussian_rasterization`: five different kernels under one name. Settings fields differ: Inria (`antialiasing`), Mip-Splatting (`kernel_size`, `subpixel_offset`), MILo (`kernel_size`, `require_depth`, `require_coord`), Scaffold-GS and Gaussian Grouping (original fields, different kernels).
- `diff_surfel_rasterization`, `diff_plane_rasterization`, `diff_gaussian_rasterization_ms`, `diff_gaussian_rasterization_gof`, `fused_ssim`: unique names, no collision.

The collision is therefore limited to one module name, which does not require separate interpreters.

Decision:

1. Add a `gs` environment (`envs/gs.yml`, the union of the six gs_* dependency lists) built with `./scripts/00_setup_environment.sh --env gs`.
2. Extensions with unique names, and one `simple_knn` build, are installed into `gs` site-packages in editable mode as before.
3. Each `diff_gaussian_rasterization` variant (inria, scaffold, mipsplatting, milo, grouping) is installed non-editable with `pip install --target .ext/<variant>` (`GS_EXT_ROOT` overrides `.ext`). None is installed into site-packages, so a backend that does not select a variant fails to import instead of silently using another one.
4. Stage scripts call `activate_backend <backend>` (`scripts/utils/activate_backend.sh`). With `GS_ENV_MODE=unified` it activates `gs` and prepends `.ext/<variant>` to `PYTHONPATH`; with `GS_ENV_MODE=split` it activates the ADR 0001 environment, unchanged.
5. `verify_patches.sh` checks, per variant, that `diff_gaussian_rasterization` loads from `.ext/<variant>` and exposes that variant's settings fields, and audits the `.ext` builds for sm_120 CUBIN.

Alternatives:

- Rename packages through patches: rejected for the same reason as in 0001; every import site in every backend would need patching.
- Migrate backends to gsplat: removes the collision at the source, but changes results relative to the reference implementations and does not cover SuGaR, MILo, or Gaussian Grouping.
- Keep split environments and only share a base yml: reduces duplication in files but not disk use or build time.

Consequences:
One torch/CUDA install replaces six, `simple_knn` is built once instead of five times, and the Inria rasterizer once instead of twice. The default stays `split` until each backend has been run in `unified` mode on hardware; then the default flips and the gs_* environments and yml files can be removed in a follow-up. Editable shared extensions build in the submodule trees, so building `gs` rebuilds the same `.so` files used by the split environments; this is safe while both use the same Python and torch versions. PlanarGS NVS evaluation now runs in its training environment (gs_sugar in split mode), which also provides `diff_plane_rasterization`; it previously ran in gs_train.

Verification:
GPU-free tests (`tests/test_activate_backend.py`) cover the backend-to-environment and backend-to-variant mapping, PYTHONPATH ordering, and failure messages. Building `gs` and running each backend in unified mode has not been verified on hardware yet.

Related: 0001 (split environments), 0004 (patches applied before builds), 0012 (torch 2.11 cu128 standardization), 0014 (Mip-Splatting isolation)
