# 0022. Python CLI as the Canonical Entry Point

Status: Accepted (2026-11-01)

Context:
Command arguments were interpreted in three places. `scripts/run_3drc.sh` and `scripts/01_sfm_hloc.sh` each guessed whether an argument was a dataset (an existing directory or a path starting with `data/`) or a method, model or backend, and accepted them in either order; `src/cli/main.py` parsed options with argparse and then rebuilt a `run_3drc.sh` command line. The guessing rules differed slightly between the copies, which once moved a SuGaR source model into the dataset slot. The pipeline command's combination checks and mesh evaluation loop lived in shell, where they could be tested only through stub workspaces, and the Python CLI kept its own copies of the backend lists.

Decision:
1. `src/cli/main.py` parses every command once. `scripts/run_3drc.sh` and `3drc.py` only forward their arguments to it.
2. The dataset is always `--dataset` (`-d`); a method, backend or view type is the one positional of its command. No argument is classified by its value.
3. Scene resolution stays in `configs/default_config.sh`. The CLI sources it once in bash, passes the resolved dataset to every stage script, and sets `SCENE_NAME` for the stage scripts so they resolve the same scene without a second fallback notice.
4. Stage scripts take fixed positional arguments; `01_sfm_hloc.sh` is `[method] [dataset]`. Conda activation stays in the stage scripts because each step runs in its own environment.
5. The pipeline command (combination checks, step order, mesh evaluation, results) moves to Python. Backends are read from `configs/backends.txt`; the bash lookups the old runner used are removed.
6. `eval` and `results` forward their arguments unparsed to `src.mesh.eval_mesh` and `collect_results.py`, so their options are defined in one place.

Alternatives:

- Keep `run_3drc.sh` canonical and make the Python CLI a thin wrapper: keeps the shell guessing and leaves the pipeline logic untestable outside stub workspaces.
- Reimplement scene resolution in Python: two copies of the fallback rules would drift like the argument rules did.

Consequences:
Commands that passed the dataset as a positional (`run_3drc.sh train 3dgs data/<scene>`, `pipeline data/<scene> hloc`) now fail with an argparse error and must use `--dataset` (and `--method` for the pipeline). `eval` takes `--mesh` and `--gt` instead of positionals. Unknown commands exit with an error instead of printing the help. `run_3drc.sh` needs a `python3` on PATH; the CLI itself uses only the standard library and runs the 3drc environment's Python for evaluation and reports.

Related: 0007 (scene-based outputs), 0015 (results aggregation)
