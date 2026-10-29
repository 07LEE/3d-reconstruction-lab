# 0020. Build the Unified gs Environment on CUDA 13.2 (torch cu130)

Status: Accepted (2026-10-29); Amends 0012, 0018

Context:
The host moved to Ubuntu 26.04 (glibc 2.43). glibc 2.41+ declares `cospi`, `sinpi` and `rsqrt` `noexcept`, and the CUDA 12.x `crt/math_functions.h` does not, so no `.cu` file compiles on this host with CUDA 12.8 or 12.9. CUDA 13.0 still fails on `rsqrt`/`rsqrtf`; CUDA 13.2 compiles cleanly. The driver (595) supports up to CUDA 13.2, and torch 2.11.0 is published as `+cu130`. Existing binaries built before the upgrade keep working, but nothing can be rebuilt, including extensions that JIT-compile during training (nvdiffrast).

Decision:

1. `envs/gs.yml` installs torch/torchvision/torchaudio 2.11.0 `+cu130`. The split gs_* environments stay on `+cu128` (ADR 0012).
2. The build toolkit is a separate Conda environment, `envs/cuda132.yml` (`cuda-toolkit=13.2`), created by `00_setup_environment.sh` when missing. `BUILD_TOOLKIT_ENV` selects it; the default is `cuda132` for the unified environment and `gs_train` (12.8) for split builds. Unified and split environments are built in separate runs because torch refuses a toolkit with a different CUDA major version.
3. `setup_build_env.sh` puts the toolkit's `targets/x86_64-linux/include` on `CPATH`, so host-side g++ does not bake in a system CUDA from `/usr/include` (`torch_scatter` recorded CUDA 12.4 otherwise).
4. The tetranerf C++ extension is built with CMake against the environment's torch with the matching C++ ABI, and with the system linker first on `PATH` and via `-Xcompiler -B/usr/bin`, because the toolkit environment's `ld` is older than the system glibc.
5. nvdiffrast (no Python 3.10 wheel), `torch_scatter`, `pytorch3d` (pinned to the commit used by gs_sugar), `simple_knn` and the shared extensions are installed non-editable from the build script; the CGAL stack comes from conda-forge in `gs.yml`. `open3d` is pinned to 0.19.0 because 0.20.0 produces an empty TSDF mesh.
6. Source patches: milo `diff-gaussian-rasterization_ms` uses a local max functor instead of `cub::Max` (removed in CUDA 13's CCCL); SuGaR passes `antialiasing=False` to the Inria rasterizer settings and accepts its extra `invdepths` output.
7. `GS_ENV_NAME` (default `gs`) and `GS_EXT_ROOT` allow a second unified environment next to the first. `camera_utils.py` converts the sensor-frame calibration to the rotated, undistorted frames before SfM (separate fix, same verification run).

Alternatives:

- Hand-edit `noexcept(true)` into the CUDA 12.9 `math_functions.h`: worked on this host, but the file lives outside the repository and cannot be reproduced by the setup script.
- Downgrade glibc or the OS: not acceptable for a general-purpose workstation.
- Keep CUDA 12.x and stay on prebuilt binaries: blocks every rebuild and JIT extension.

Consequences:
The split gs_* environments cannot be rebuilt on this host. Moving to cu130 changes the torch build used for training, so results are not bit-identical to the cu128 runs. Editable installs of shared extensions would clobber the other torch's `.so` (this broke the old cu128 `gs` while `gs132` was built), hence item 5.

Verification:
Environment `gs132` (same content as `gs.yml`) was built with the scripts and `verify_patches.sh --verbose` passed. On a 99/288-image hloc reconstruction of one scene, 3dgs, mipsplatting, scaffoldgs, 2dgs, planargs, grouping (downsample 2), milo, tsdf and sugar completed with 3000 iterations. The environment was then rebuilt from scratch with the final scripts (the `cuda132` toolkit environment was created automatically, `verify_patches.sh` passed) and all nine backends completed again on the same scene.

Related: 0004 (patches), 0012 (cu128 standardization), 0018 (unified environment)
