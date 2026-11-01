"""SfM reconstruction report: camera trajectory plots and per-image quality statistics.

Writes <model_dir>/report/sfm_report.md with trajectory.png and per_image.png, or opens the
cameras, trajectory and sparse points in an Open3D window with --show.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
# A step longer than this multiple of the median step between consecutive frames is flagged as a pose jump.
JUMP_FACTOR = 5.0
MAX_PLOT_POINTS = 50000


@dataclass
class CameraTrack:
    """Registered cameras in capture order."""

    names: List[str]
    centers: np.ndarray  # (N, 3) camera centers in world coordinates
    forward: np.ndarray  # (N, 3) viewing directions in world coordinates
    up: np.ndarray  # (N, 3) image-up directions in world coordinates
    reproj_errors: np.ndarray  # (N,) mean reprojection error in pixels
    num_points: np.ndarray  # (N,) observed 3D points
    groups: List[str] = field(default_factory=list)


@dataclass
class SparseModel:
    track: CameraTrack
    points: np.ndarray  # (M, 3)
    colors: np.ndarray  # (M, 3) uint8
    cameras: List[Dict[str, Any]]
    mean_reproj_error: float
    mean_track_length: float
    mean_observations: float


def capture_order_key(name: str) -> List[Any]:
    """Natural sort key, so frame_10 follows frame_9 and timestamp names sort by time."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", name)]


def image_group(name: str) -> str:
    """Video prefix of an image name, using the same rule as sfm_info.json."""
    stem = Path(name).stem
    if "_" not in stem:
        return "single"
    head, tail = stem.rsplit("_", 1)
    return head if tail.isdigit() else stem.split("_", 1)[0]


def sort_track(track: CameraTrack) -> CameraTrack:
    order = sorted(range(len(track.names)), key=lambda i: capture_order_key(track.names[i]))
    names = [track.names[i] for i in order]
    return CameraTrack(
        names=names,
        centers=track.centers[order],
        forward=track.forward[order],
        up=track.up[order],
        reproj_errors=track.reproj_errors[order],
        num_points=track.num_points[order],
        groups=[image_group(n) for n in names],
    )


def load_model(model_dir: str | Path) -> SparseModel:
    """Reads a COLMAP model with pycolmap and computes per-image reprojection errors."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(model_dir))
    xyz_by_id = {pid: p.xyz for pid, p in rec.points3D.items()}

    names, centers, forward, up, errors, counts = [], [], [], [], [], []
    for image in rec.images.values():
        if not image.has_pose:
            continue
        cam_from_world = image.cam_from_world().matrix()
        world_from_cam_rot = cam_from_world[:, :3].T
        names.append(image.name)
        centers.append(image.projection_center())
        forward.append(world_from_cam_rot[:, 2])
        up.append(-world_from_cam_rot[:, 1])

        obs = [(p.xy, p.point3D_id) for p in image.points2D if p.has_point3D()]
        counts.append(len(obs))
        if obs:
            xy = np.array([o[0] for o in obs])
            world = np.array([xyz_by_id[o[1]] for o in obs])
            cam_pts = world @ cam_from_world[:, :3].T + cam_from_world[:, 3]
            proj = np.asarray(image.camera.img_from_cam(cam_pts))
            errors.append(float(np.linalg.norm(proj - xy, axis=1).mean()))
        else:
            errors.append(float("nan"))

    track = sort_track(
        CameraTrack(
            names=names,
            centers=np.array(centers).reshape(-1, 3),
            forward=np.array(forward).reshape(-1, 3),
            up=np.array(up).reshape(-1, 3),
            reproj_errors=np.array(errors),
            num_points=np.array(counts, dtype=int),
        )
    )
    points = np.array([p.xyz for p in rec.points3D.values()]).reshape(-1, 3)
    colors = np.array([p.color for p in rec.points3D.values()], dtype=np.uint8).reshape(-1, 3)
    cameras = [
        {"camera_id": cid, "model": cam.model.name, "width": cam.width, "height": cam.height, "params": [round(float(v), 4) for v in cam.params]}
        for cid, cam in rec.cameras.items()
    ]
    return SparseModel(
        track=track,
        points=points,
        colors=colors,
        cameras=cameras,
        mean_reproj_error=float(rec.compute_mean_reprojection_error()),
        mean_track_length=float(rec.compute_mean_track_length()),
        mean_observations=float(rec.compute_mean_observations_per_reg_image()),
    )


def list_images(image_dir: Optional[str | Path]) -> List[str]:
    if not image_dir or not Path(image_dir).is_dir():
        return []
    names = [f.name for f in Path(image_dir).iterdir() if f.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(names, key=capture_order_key)


def trajectory_stats(track: CameraTrack, jump_factor: float = JUMP_FACTOR) -> Dict[str, Any]:
    """Path length and pose jumps between consecutive frames of the same video."""
    jumps, steps = [], []
    for i in range(1, len(track.names)):
        if track.groups[i] != track.groups[i - 1]:
            continue
        steps.append((i, float(np.linalg.norm(track.centers[i] - track.centers[i - 1]))))
    lengths = np.array([s for _, s in steps])
    median = float(np.median(lengths)) if len(lengths) else 0.0
    if median > 0:
        for i, step in steps:
            if step > jump_factor * median:
                jumps.append({"from": track.names[i - 1], "to": track.names[i], "step": step, "ratio": step / median, "index": i})
    return {
        "path_length": float(lengths.sum()) if len(lengths) else 0.0,
        "median_step": median,
        "max_step": float(lengths.max()) if len(lengths) else 0.0,
        "jumps": jumps,
        "groups": {g: track.groups.count(g) for g in dict.fromkeys(track.groups)},
    }


def plot_axes(track: CameraTrack) -> np.ndarray:
    """Rows are (horizontal 1, horizontal 2, vertical) in world coordinates.

    Vertical is the mean image-up direction of the cameras, which assumes upright images;
    the horizontal axes are the principal directions of the camera centers in that plane.
    """
    vertical = track.up.mean(axis=0)
    norm = np.linalg.norm(vertical)
    vertical = vertical / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
    centered = track.centers - track.centers.mean(axis=0)
    flat = centered - np.outer(centered @ vertical, vertical)
    _, _, vt = np.linalg.svd(flat, full_matrices=False) if len(flat) > 1 else (None, None, np.eye(3))
    h1 = vt[0] - (vt[0] @ vertical) * vertical
    if np.linalg.norm(h1) < 1e-6:
        h1 = np.cross(vertical, [1.0, 0.0, 0.0]) if abs(vertical[0]) < 0.9 else np.cross(vertical, [0.0, 1.0, 0.0])
    h1 /= np.linalg.norm(h1)
    h2 = np.cross(vertical, h1)
    return np.stack([h1, h2, vertical])


def _subsample(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if len(points) <= max_points:
        return np.arange(len(points))
    return np.random.default_rng(seed).choice(len(points), max_points, replace=False)


def _crop_to_cameras(points: np.ndarray, centers: np.ndarray, margin: float = 1.0) -> np.ndarray:
    """Keeps sparse points within the camera extent plus a margin, so far outliers do not shrink the plot."""
    lo, hi = centers.min(axis=0), centers.max(axis=0)
    pad = margin * max(float((hi - lo).max()), 1e-6)
    inside = np.all((points >= lo - pad) & (points <= hi + pad), axis=1)
    return points[inside]


def plot_trajectory(model: SparseModel, stats: Dict[str, Any], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    track = model.track
    axes_w = plot_axes(track)
    origin = track.centers.mean(axis=0)
    cams = (track.centers - origin) @ axes_w.T
    fwd = track.forward @ axes_w.T
    pts = _crop_to_cameras(model.points, track.centers)
    pts = (pts[_subsample(pts, MAX_PLOT_POINTS)] - origin) @ axes_w.T
    order = np.arange(len(cams))
    arrow_every = max(1, len(cams) // 40)
    arrow_len = 3 * stats["median_step"] if stats["median_step"] > 0 else 0.1

    fig, panels = plt.subplots(1, 2, figsize=(16, 7.5))
    for ax, (a, b), title in ((panels[0], (0, 1), "Top view"), (panels[1], (0, 2), "Side view")):
        ax.scatter(pts[:, a], pts[:, b], s=0.3, c="0.75", linewidths=0, rasterized=True)
        for group in stats["groups"]:
            idx = [i for i, g in enumerate(track.groups) if g == group]
            if len(idx) > 1:
                seg = np.stack([cams[idx[:-1]][:, [a, b]], cams[idx[1:]][:, [a, b]]], axis=1)
                lc = LineCollection(list(seg), cmap="viridis", linewidths=2)
                lc.set_array(order[idx[:-1]])
                lc.set_clim(0, max(len(cams) - 1, 1))
                ax.add_collection(lc)
        sel = order[::arrow_every]
        ax.quiver(
            cams[sel, a],
            cams[sel, b],
            fwd[sel, a] * arrow_len,
            fwd[sel, b] * arrow_len,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
            color="tab:orange",
            alpha=0.8,
        )
        ax.scatter(cams[0, a], cams[0, b], s=90, marker="o", c="tab:green", zorder=5, label="start")
        ax.scatter(cams[-1, a], cams[-1, b], s=90, marker="s", c="tab:red", zorder=5, label="end")
        for jump in stats["jumps"]:
            i = jump["index"]
            ax.plot(cams[[i - 1, i], a], cams[[i - 1, i], b], "r--", linewidth=1.5)
            ax.scatter(cams[i, a], cams[i, b], s=60, marker="x", c="red", zorder=6)
        ax.set_title(title)
        ax.set_xlabel("horizontal axis 1")
        ax.set_ylabel("horizontal axis 2" if b == 1 else "vertical (mean camera up)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linewidth=0.3)
    panels[0].legend(loc="best")
    sm = ScalarMappable(cmap="viridis", norm=Normalize(0, max(len(cams) - 1, 1)))
    fig.colorbar(sm, ax=panels, shrink=0.8, label="capture order (frame index)")
    fig.suptitle("Camera trajectory (orange arrows: viewing direction, red: pose jumps)")
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_per_image(model: SparseModel, all_images: Sequence[str], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    track = model.track
    names = list(all_images) if all_images else track.names
    position = {n: i for i, n in enumerate(names)}
    x = np.array([position.get(n, -1) for n in track.names])
    keep = x >= 0
    missing = [i for i, n in enumerate(names) if n not in set(track.names)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True)
    ax1.plot(x[keep], track.reproj_errors[keep], ".-", markersize=3, linewidth=0.8)
    ax1.axhline(model.mean_reproj_error, color="0.4", linestyle="--", linewidth=0.8, label="model mean")
    ax1.set_ylabel("reprojection error (px)")
    ax1.legend(loc="upper right")
    ax2.plot(x[keep], track.num_points[keep], ".-", markersize=3, linewidth=0.8, color="tab:green")
    ax2.set_ylabel("observed 3D points")
    ax2.set_xlabel("frame index in capture order")
    for ax in (ax1, ax2):
        for i in missing:
            ax.axvline(i, color="red", alpha=0.4, linewidth=1)
        ax.grid(True, linewidth=0.3)
    fig.suptitle("Per-image quality (red lines: images not registered)")
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


def _fmt_list(items: Sequence[str], limit: int = 20) -> str:
    shown = ", ".join(f"`{i}`" for i in items[:limit])
    return shown + (f" and {len(items) - limit} more" if len(items) > limit else "")


def build_report(model: SparseModel, stats: Dict[str, Any], all_images: Sequence[str], title: str, sfm_info: Optional[Dict[str, Any]] = None) -> str:
    track = model.track
    registered = set(track.names)
    unregistered = [n for n in all_images if n not in registered]
    total = len(all_images) if all_images else len(track.names)
    lines = [
        f"# SfM Report: {title}",
        "",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Registered images | {len(track.names)} / {total} ({100.0 * len(track.names) / max(total, 1):.1f}%) |",
        f"| 3D points | {len(model.points)} |",
        f"| Mean reprojection error | {model.mean_reproj_error:.3f} px |",
        f"| Mean track length | {model.mean_track_length:.2f} |",
        f"| Mean observations per image | {model.mean_observations:.1f} |",
    ]
    if sfm_info:
        for key in ("method", "strategy", "overlap", "retrieval_k"):
            if key in sfm_info:
                lines.append(f"| SfM {key} | {sfm_info[key]} |")
    for cam in model.cameras:
        lines.append(f"| Camera {cam['camera_id']} | {cam['model']} {cam['width']}x{cam['height']}, " f"params {cam['params']} |")

    lines += [
        "",
        "## Trajectory",
        "",
        "Distances are in SfM units (arbitrary scale).",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Path length | {stats['path_length']:.3f} |",
        f"| Median step between frames | {stats['median_step']:.4f} |",
        f"| Largest step | {stats['max_step']:.4f} |",
        f"| Videos (frames) | {', '.join(f'{g} ({n})' for g, n in stats['groups'].items())} |",
        f"| Pose jumps (step > {JUMP_FACTOR:g}x median) | {len(stats['jumps'])} |",
        "",
        "![Camera trajectory](trajectory.png)",
        "",
    ]
    if stats["jumps"]:
        lines += ["| From | To | Step | x median |", "| --- | --- | --- | --- |"]
        lines += [f"| `{j['from']}` | `{j['to']}` | {j['step']:.4f} | {j['ratio']:.1f} |" for j in stats["jumps"]]
        lines.append("")

    valid = ~np.isnan(track.reproj_errors)
    worst = [i for i in np.argsort(-np.where(valid, track.reproj_errors, -np.inf))[:5] if valid[i]]
    sparse = list(np.argsort(track.num_points)[:5])
    lines += [
        "## Per-image quality",
        "",
        "![Per-image quality](per_image.png)",
        "",
        "Highest reprojection error: " + ", ".join(f"`{track.names[i]}` ({track.reproj_errors[i]:.2f} px)" for i in worst),
        "",
        "Fewest observed 3D points: " + ", ".join(f"`{track.names[i]}` ({track.num_points[i]})" for i in sparse),
        "",
        "## Unregistered images",
        "",
        (f"{len(unregistered)}: " + _fmt_list(unregistered)) if unregistered else "None.",
        "",
    ]
    return "\n".join(lines)


def show_open3d(model: SparseModel, stats: Dict[str, Any]) -> None:
    """Opens cameras, the capture-order trajectory and sparse points in an Open3D window."""
    import open3d as o3d
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    track = model.track
    geometries = []
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(_crop_to_cameras(model.points, track.centers, margin=2.0)))
    geometries.append(pcd.paint_uniform_color([0.6, 0.6, 0.6]))

    mapper = ScalarMappable(norm=Normalize(0, max(len(track.names) - 1, 1)), cmap="viridis")
    order_colors = np.asarray(mapper.to_rgba(np.arange(len(track.names))))[:, :3]
    lines = [[i - 1, i] for i in range(1, len(track.names)) if track.groups[i] == track.groups[i - 1]]
    traj = o3d.geometry.LineSet(o3d.utility.Vector3dVector(track.centers), o3d.utility.Vector2iVector(lines))
    traj.colors = o3d.utility.Vector3dVector(order_colors[[i for _, i in lines]])
    geometries.append(traj)

    size = 2 * stats["median_step"] if stats["median_step"] > 0 else 0.05
    for i in range(len(track.names)):
        right = np.cross(track.forward[i], track.up[i])
        apex, f, u = track.centers[i], track.forward[i] * size, track.up[i] * size * 0.75
        r = right * size
        corners = [apex + f + r + u, apex + f - r + u, apex + f - r - u, apex + f + r - u]
        frustum = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector([apex] + corners),
            o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]),
        )
        frustum.paint_uniform_color(order_colors[i])
        geometries.append(frustum)
    # draw_geometries comes from open3d's compiled bindings, which Pylance cannot see
    o3d.visualization.draw_geometries(geometries, window_name="SfM cameras and trajectory")  # pyright: ignore[reportAttributeAccessIssue]


def resolve_paths(target: str | Path, method: Optional[str], image_dir: Optional[str]) -> Tuple[Path, Optional[Path]]:
    """Accepts a model directory or a dataset directory (data/<scene>) and returns (model_dir, image_dir)."""
    target = Path(target)
    if (target / "images.bin").is_file() or (target / "images.txt").is_file():
        model_dir = target
        dataset_dir = target.parent.parent if target.parent.name == "sparse" else None
    else:
        model_dir = target / "sparse" / (method or "0")
        dataset_dir = target
        if not model_dir.is_dir():
            raise FileNotFoundError(f"No SfM model at {model_dir}")
    if image_dir:
        return model_dir, Path(image_dir)
    if dataset_dir:
        for name in ("raw_images", "images"):
            if (dataset_dir / name).is_dir():
                return model_dir, dataset_dir / name
    return model_dir, None


def write_report(model_dir: Path, image_dir: Optional[Path], out_dir: Optional[Path] = None) -> Path:
    import json

    model = load_model(model_dir)
    if len(model.track.names) == 0:
        raise RuntimeError(f"No registered images in {model_dir}")
    stats = trajectory_stats(model.track)
    all_images = list_images(image_dir)
    out_dir = out_dir or model_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    info_path = model_dir / "sfm_info.json"
    sfm_info = json.loads(info_path.read_text()) if info_path.is_file() else None
    dataset = model_dir.parent.parent.name if model_dir.parent.name == "sparse" else model_dir.name
    title = f"{dataset} ({model_dir.resolve().name})"

    plot_trajectory(model, stats, out_dir / "trajectory.png")
    plot_per_image(model, all_images, out_dir / "per_image.png")
    report_path = out_dir / "sfm_report.md"
    report_path.write_text(build_report(model, stats, all_images, title, sfm_info), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="SfM report: camera trajectory and per-image quality")
    parser.add_argument("target", help="Dataset directory (data/<scene>) or COLMAP model directory")
    parser.add_argument("--method", default=None, help="SfM method subdirectory under sparse/ (default: active sparse/0)")
    parser.add_argument("--image_dir", default=None, help="Input image directory, used to list unregistered images")
    parser.add_argument("--out_dir", default=None, help="Report directory (default: <model_dir>/report)")
    parser.add_argument("--show", action="store_true", help="Open an interactive Open3D window instead of writing the report")
    args = parser.parse_args()

    model_dir, image_dir = resolve_paths(args.target, args.method, args.image_dir)
    if args.show:
        model = load_model(model_dir)
        show_open3d(model, trajectory_stats(model.track))
        return
    report_path = write_report(model_dir, image_dir, Path(args.out_dir) if args.out_dir else None)
    print(f"SfM report written to {report_path}")


if __name__ == "__main__":
    main()
