# M3 verification and experiment record

Owner: Suryansh Bajaj. Created: 11 October 2026. Updated: 27 October 2026.

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

## Canonical Kaggle run, 27 October 2026

The fresh 200-step GPU pilot and its backup were checked separately; pilot
checkpoints were not used for training. The canonical run used the unchanged
`configs/m3.yaml`, seed 42, public RGB subset version 1 (18 sequences and
8,642 Camera_0 frames), and reviewed training commit
`785e5ad933b3a0dd2ac96d940ebb981f0ab01709`. The saved [Kaggle Version 2](https://www.kaggle.com/code/suryanshbajaj/m3-canonical-pilot-2026-10-27?scriptVersionId=362807950)
completed 60 epochs on a Tesla T4. The full output archive passed its ZIP
integrity check; `best.pt`, `last.pt`, history, metrics, trajectories and plots
were backed up outside Git. The [manifest](../../results/m3/manifest.json)
records the archive and checkpoint hashes.

The [fresh pilot Version 1](https://www.kaggle.com/code/suryanshbajaj/m3-canonical-pilot-2026-10-27?scriptVersionId=362802043)
ran 200 optimizer steps at the same training commit. Its train total was
0.089903, validation total 0.163761, and both encoder maximum updates were
nonzero (depth 0.019785, pose 0.010995). Peak Torch allocation was 2.34 GiB
on a Tesla T4. Its separate downloaded archive passed integrity checking and
has SHA-256 `cb399dd5c33437705bbecae52a999e013400d31be64fe5b85b5f9ab667de5466`.
These are integration diagnostics, not benchmark results.

The best checkpoint is epoch 40 (zero-based), selected by the minimum
validation Sim(3) ATE of **2.0232 m**. The checkpoint records the full
training commit, `selection_key=val_ate`, epoch 40 and 60-epoch configuration;
its SHA-256 matches the copy inside the output archive. `summary.json` names
this value `best_val_loss`, but the selected value is ATE, not loss. All 60
history rows have finite loss and coverage values, with positive active
coverage. Training loss fell from 0.07037 to 0.02699; validation loss
fluctuated and ended at 0.06039.

| Evaluation with best checkpoint | Sim(3) ATE (m) | SE(3) ATE (m) |
|---|---:|---:|
| Test: Scene20/clone | 25.082 | 227.123 |
| Robustness aggregate | 91.524 | 227.971 |
| Scene20/rain | 33.296 | — |
| Scene20/fog | 78.211 | — |
| Scene20/15-deg-left | 48.469 | — |
| Scene20/30-deg-right | 206.118 | — |

The test [pose metrics](../../results/m3/metrics.json),
[robustness metrics](../../results/m3/robustness/metrics.json),
[history](../../results/m3/history.csv), [trajectory](../../results/m3/trajectory.png),
and [loss/coverage curves](../../results/m3/loss_curves.png) are copied from
the saved version. The trajectory is Sim(3)-aligned for display; monocular M3
does not recover absolute scale. The especially large 30-degree viewpoint
error is a measured robustness limitation. Real depth scores remain pending
official VKITTI 2.0.3 depth and a separate evaluation-only version pinned to
the depth-evaluation merge commit `1228bf151def45319a890aa2f4972de18a4d8831`.

## Earlier pilot and remaining matched experiments

A previous 200-step Kaggle GPU pilot completed at
`3486a66fe528385497a7f02996d0e6eac7fa9490`. The owner reports train loss
0.089903, validation loss 0.163761 and nonzero updates to both encoders, and
has verified local copies of its checkpoints, logs and three diagnostic PNGs.
Those owner-reported artifacts have not been independently rechecked for this
record. The pilot predates the merged mask-rendering fix and is not a canonical
benchmark. The fresh pilot and canonical training reported above supersede
that pending status. Official depth acquisition and evaluation remain pending.

| Experiment | Budget | Only change from matched baseline | Status |
|---|---:|---|---|
| Canonical M3 | 60 epochs, step at 45 | None | Complete; best epoch 40 |
| Ablation control | 20 epochs, step at 15 | Shorter matched budget | Pending GPU |
| No automasking | 20 epochs, step at 15 | automask=false | Pending GPU |
| Mean reprojection | 20 epochs, step at 15 | min_reproj=false | Pending GPU |
| Single scale | 20 epochs, step at 15 | num_scales=1 | Pending GPU |

Hold pretrained initialization, split, seed, batch size, augmentation and
precision fixed. Compare each 20-epoch variant against the 20-epoch control.
Do not select the configuration using test depth/poses or robustness variants.

## Evidence retained and remaining work

The saved Kaggle versions, downloaded archives, checkpoint hashes, history,
metrics, TUM trajectories and plots identify the pilot and full run separately.
Keep checkpoints outside Git. The depth-evaluation code merged in PR #22, but
has not been run on real depth. That evaluation must use official VKITTI 2.0.3
ground truth, no crop, 128x416, valid GT `(0.001,80)` metres, image-level
averaging, primary per-image median scaling and secondary raw scaling. Pose
trajectories never inherit GT depth scales.

Inspect moving vehicles, occlusions, static intervals, texture-poor road/sky,
depth saturation and camera-angle failures. Published KITTI scores and this
VKITTI2 experiment are not directly comparable. Existing team latency figures
have different run provenance; measure the final comparison on the same hardware.
