# 0013. Clean Output Hierarchy and Subdirectory Isolation Policy

Status: Accepted (2026-08-14)

Context:
During training and evaluation of 3D Gaussian Splatting, 2DGS, PlanarGS, and Scaffold-GS models, multiple intermediate artifacts (`chkpnt10000.pth`, `chkpnt20000.pth`, `events.out.tfevents.*`) and automated source code backup folders (`scaffoldgs/backup/`) were historically created directly in the output folder of each scene (`outputs/<scene_name>/<model>/`). This created root directory clutter and dumped hundreds of template PNG/documentation files into `outputs/<scene_name>/summary.md`.

Decision:

1. All `torch.utils.tensorboard.SummaryWriter` instances MUST output to `os.path.join(args.model_path, "events")`.
2. All training checkpoints (`chkpnt*.pth`) MUST be saved to `os.path.join(scene.model_path, "checkpoints")`. Pipeline auto-resume scripts MUST scan both `${MODEL_OUTPUT}/checkpoints/chkpnt*.pth` and legacy `${MODEL_OUTPUT}/chkpnt*.pth` for backward compatibility.
3. `src/utils/inspect_outputs.py` MUST exclude all files located inside `backup/` directories from `summary.md`.

Consequences:
`outputs/<scene_name>/<model>/` root directories remain clean and easy to navigate. `summary.md` reports provide concise artifact inventories. Full backward compatibility for auto-resume remains intact.
