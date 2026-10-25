# 0016. Gaussian Grouping Output Under Scene Outputs

Status: Accepted (2026-10-25)

Context:
`scripts/04_train_grouping.sh` wrote models to `third_party/gaussian-grouping/output/<scene>/` inside the submodule. Unlike every other stage, the results were outside `outputs/<scene>/` (ADR 0007), had no `pipeline_meta.json`, and were excluded from stage timing (ADR 0015).

Decision:

1. Gaussian Grouping trains into `outputs/<scene>/grouping/` by passing an absolute `-m` path to the upstream `train.py`.
2. The script records `pipeline_meta.json` (Stage 4) and a `grouping_train` timing entry like the other stages.
3. Results already in the legacy submodule location are left in place; the script warns when it finds them.

Consequences:
All stage outputs for a scene live under `outputs/<scene>/`, and `inspect_outputs.py` lists Grouping artifacts as Stage 4. Legacy results must be moved manually if they are still needed.
