# M3 verification and experiment record

Owner: Suryansh Bajaj. Date: 11 October 2026.

## What was actually run

Local repository virtual environment; CPU; seed 42; no depth PNGs; random
encoder weights. Three optimizer steps with batch 2 were run through the
existing shared Trainer on real training RGB, followed by a small validation
check. These checks validate integration, not trained model performance.

| CPU smoke configuration | Train photometric | Train active pixels | Depth encoder max update | Pose encoder max update |
|---|---:|---:|---:|---:|
| One scale | 0.1552946 | 61.89% | 0.000300229 | 0.000300478 |
| Four scales | 0.1570179 | 61.93% | 0.000300258 | 0.000300470 |

The two checks are not an accuracy ablation. They use different initialized
decoder heads and only three steps. All losses/gradients were finite, both
encoders updated, and exported reconstruction/mask/depth images were inspected.
The nearly constant depth in these smoke images is expected for untrained
encoders; it is not depth-evaluation evidence. Disposable checkpoints and plots
were written under /tmp, outside the committed results tree.

The standalone M3 CPU suite passed 204 tests, with one CUDA-only check skipped;
the focused notebook suite passed 46 tests. Ruff and git diff --check passed.
Run from src/ using the repository virtual environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider tests/test_notebooks.py tests/test_m3_notebooks.py
python -m ruff check mvo scripts tests
```

Separately, a temporary combined-integration tree previously passed 191 tests
with one CUDA-only skip, including CPU epoch-boundary resume and synthetic depth
evaluation of 837 unique frames. Those shared changes are outside this model PR;
these historical checks do not establish those features on the standalone branch.

## Pending measured experiments

No pretrained GPU pilot, 60-epoch baseline, depth evaluation, robustness run,
latency benchmark or matched-budget ablation has been run. The user will run
the Kaggle notebook. Depth acquisition is also a user-run step.

| Experiment | Budget | Only change from matched baseline | Status |
|---|---:|---|---|
| Canonical M3 | 60 epochs, step at 45 | None | Pending GPU |
| Ablation control | 20 epochs, step at 15 | Shorter matched budget | Pending GPU |
| No automasking | 20 epochs, step at 15 | automask=false | Pending GPU |
| Mean reprojection | 20 epochs, step at 15 | min_reproj=false | Pending GPU |
| Single scale | 20 epochs, step at 15 | num_scales=1 | Pending GPU |

Hold pretrained initialization, split, seed, batch size, augmentation and
precision fixed. Compare each 20-epoch variant against the 20-epoch control.
Do not select the configuration using test depth/poses or robustness variants.

## Evidence to capture from Kaggle

Pin the reviewed M3 implementation and explicitly set RUN_PILOT=True for the
200-step pilot. Inspect finite updates, active/valid coverage, reconstruction
images and GPU memory, then freeze the full-run recipe. Record the exact
checkpoint epoch/hash, training/evaluation revisions, cfg, library versions and
hardware. Save history.csv, metrics.json, TUM trajectory, loss/coverage curves
and fixed depth/reprojection examples. Reliable resume, depth evaluation and
automatic run.json provenance require the separately reviewed shared integrations.
Keep checkpoints outside Git.

After validation selects best.pt, evaluate Scene20/clone once for the frozen
recipe, followed by the held-out robustness split in a separate output directory.
Depth: no crop, 128x416, valid GT (0.001,80) metres, image-level averaging, primary
per-image median scaling and secondary raw scaling. Pose trajectories never
inherit GT depth scales. Report Sim(3) ATE and scale-fixed metrics separately.

Inspect moving vehicles, occlusions, static intervals, texture-poor road/sky,
depth saturation and camera-angle failures. Published KITTI scores and this
VKITTI2 experiment are not directly comparable. Existing team latency figures
have different run provenance; measure the final comparison on the same hardware.
