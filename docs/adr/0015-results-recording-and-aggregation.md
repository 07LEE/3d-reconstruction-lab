# 0015. Results Recording and Aggregation for Cross-Engine Comparison

Status: Accepted (2026-10-25)

Context:
The lab runs four SfM engines, five Gaussian training backends, and three mesh extractors, but no comparison results were retained. `outputs/` is untracked, `record_stage_timing` existed without callers, NVS metrics (PSNR / SSIM / LPIPS) had no pipeline step, and `eval_mesh.py` wrote every mesh evaluation to the shared `outputs/eval/` directory, so each run overwrote the previous one.

Decision:

1. Every Step 1-3 script measures its main workload with `bench_start` / `bench_record` (`scripts/utils/common_logger.sh`) and appends an entry to `outputs/benchmark_timings.json`. Each entry stores the stage output directory relative to the project root (the join key), stage parameters, `resumed_from` when training resumed from a checkpoint, and an environment fingerprint (GPU, driver, torch build, conda env, git commit, dirty flag) as required by the ADR rules for performance numbers.
2. Step 5b (`scripts/05_eval_nvs.sh`) computes NVS metrics with each backend's own `render.py --skip_train` and `metrics.py`, so the numbers match the upstream evaluation protocol. It refuses models whose `cfg_args` lacks `eval=True`: without the holdout split there are no test views, and training-view fits are not comparable.
3. `eval_mesh.py` defaults its output to `<mesh_dir>/eval/<mesh_stem>/` so results stay next to the mesh and do not overwrite each other.
4. `src/utils/collect_results.py` (standard library only) joins SfM statistics, NVS metrics, mesh metrics, model sizes, and timings into `outputs/results_summary.md` and `outputs/results_runs.csv`. `--publish docs/results.md` writes a tracked copy for results worth keeping in the repository.

Consequences:
Comparison tables can be regenerated from `outputs/` at any time. NVS metrics require training with `GS_EVAL_MODE="true"`, which holds out every 8th view; full reconstructions (the default) still train on all views and show `-` in the NVS columns. Times recorded after a resume cover only the resumed portion and are marked `(resumed)`. Gaussian Grouping (Step 4) is not timed because its outputs stay inside the submodule directory.
