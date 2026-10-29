# 0021. Remove the Split gs_* Environment Mode

Status: Accepted (2026-10-29); Supersedes 0001; Amends 0018

Context:
ADR 0018 added the unified `gs` environment next to the six split `gs_*` environments of ADR 0001 and kept `GS_ENV_MODE=split` as a fallback. ADR 0020 moved `gs` to CUDA 13.2 / torch cu130 because the host (glibc 2.43) cannot compile CUDA 12.x code. The split environments are cu128, so they can no longer be rebuilt on this host, and only `gs` has been verified from a clean rebuild: all nine backends passed on one scene (2026-10-29). The split path therefore stayed in the scripts as code that could be run only against stale binaries.

Decision:
1. `gs` is the only Gaussian environment. `GS_ENV_MODE`, `--all`, the per-backend `gs_*` mapping in `activate_backend.sh`, the `gs_*` extension lists in `00_setup_environment.sh`, and the `gs_*` checks in `verify_patches.sh` are removed. `GS_ENV_NAME` still renames the environment.
2. `envs/gs_*.yml` and `envs/archive/*.bak` are deleted, and the seven `gs_*` Conda environments (including `gs_original`) are removed from the host.
3. Before removal, each environment was exported with `conda env export` to `~/conda_env_backups/` (outside the repository).

Alternatives:

- Keep split mode as a fallback: it would need CUDA 12.x builds that this host cannot produce, so the fallback could not be rebuilt or tested.
- Keep the code and only delete the environments: leaves a mode whose first step fails with "environment not found".

Consequences:
The cu128 setup can be restored only from the exported yml files, and its extensions would not build on this host. Past results in `outputs/` that record `gs_train` or another `gs_*` as `conda_env` stay valid as history. Per-backend isolation is now provided only by `.ext/<variant>/` and `PYTHONPATH` (ADR 0018), so a backend that needs a different torch version would require a new environment.

Related: 0001 (split environments), 0018 (unified environment), 0020 (CUDA 13.2)
