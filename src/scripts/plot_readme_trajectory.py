"""Draw the M4 test trajectory shown in the repository README.

Run from src/: PYTHONPATH=. python3 scripts/plot_readme_trajectory.py
Requires the local VKITTI2 subset, NumPy and Matplotlib. No checkpoint is needed.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mvo.common.metrics.ate import umeyama


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    extrinsics = root / "data/vkitti2/Scene20/clone/extrinsic.txt"
    predictions = root / "results/m4/pred_Scene20_clone.txt"
    metrics = root / "results/m4/metrics.json"
    output = root.parent / "assets/m4_trajectory.png"

    camera_to_world = []
    with extrinsics.open() as file:
        next(file)
        for line in file:
            values = line.split()
            if int(values[1]) == 0:
                world_to_camera = np.array(values[2:18], dtype=float).reshape(4, 4)
                camera_to_world.append(np.linalg.inv(world_to_camera))
    ground_truth = np.array(camera_to_world)[:, :3, 3]
    predicted = np.loadtxt(predictions)[:, 1:4]
    if ground_truth.shape != predicted.shape:
        raise ValueError("Ground truth and prediction must have one pose per frame")

    rotation, translation, scale = umeyama(predicted, ground_truth, with_scale=True)
    aligned = (scale * (rotation @ predicted.T)).T + translation
    rmse = np.sqrt(np.mean(np.sum((aligned - ground_truth) ** 2, axis=1)))
    reported_ate = json.loads(metrics.read_text())["ate_sim3"]
    if abs(rmse - reported_ate) > 0.05:
        raise ValueError(f"Trajectory does not match the recorded M4 run: ATE={rmse:.3f} m")

    plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.plot(ground_truth[:, 0], ground_truth[:, 2], color="#087f8c", lw=2.8,
            label="Ground truth")
    ax.plot(aligned[:, 0], aligned[:, 2], color="#d48224", lw=2.5,
            label="M4 with CBAM")
    ax.scatter(ground_truth[0, 0], ground_truth[0, 2], color="#253548",
               s=65, zorder=4, label="Start")
    ax.scatter(ground_truth[-1, 0], ground_truth[-1, 2], marker="X",
               color="#253548", s=85, zorder=4, label="End")
    ax.set(xlabel="World X (m)", ylabel="World Z (m)",
           title="Scene20/clone · camera trajectory")
    ax.grid(alpha=0.18)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", frameon=False)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.20)
    fig.text(0.5, 0.055,
             f"Full test sequence · Sim(3) aligned for display · ATE {rmse:.2f} m",
             ha="center", color="#56616f", fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Saved {output} (ATE {rmse:.3f} m)")


if __name__ == "__main__":
    main()
