# Does whitening the pose targets help?

Motivation, measured on the training split by `scripts/measure_pose_stats.py`:
the per-axis spread of the ground-truth relative pose is 424x, so a plain MSE
sends **99.89% of its gradient to tz** and trains forward translation alone.

## Setup

M4, ImageNet-pretrained ResNet18, identical seed (42), identical data order,
120 optimiser steps at batch 16, Adam 1e-4, CPU. `whiten=False` uses the
measured `rot_weight=5900`; `whiten=True` uses `rot_weight=1.0`, since whitening
already makes the axes commensurate.

Both variants are scored by **RMSE in metric units** on held-out validation
samples, so the numbers are comparable even though their training losses are
on different scales.

## Result

| axis | unwhitened @0 | @120 | change | whitened @0 | @120 | change |
|---|---|---|---|---|---|---|
| tx | 0.00817 | 0.00880 | +7.6% | 0.00878 | 0.00846 | −3.6% |
| ty | 0.00647 | 0.00404 | −37.6% | 0.00449 | 0.00472 | +5.1% |
| tz | 0.77465 | 0.39372 | −49.2% | 0.50147 | 0.34873 | −30.5% |
| rx | 0.00110 | 0.00151 | **+37.4%** | 0.00110 | 0.00110 | −0.0% |
| ry | 0.00232 | 0.00294 | **+26.8%** | 0.00424 | 0.00225 | **−46.9%** |
| rz | 0.00096 | 0.00097 | +1.9% | 0.00096 | 0.00096 | −0.2% |

Aggregate at step 120:

| | unwhitened | whitened | |
|---|---|---|---|
| translation RMSE (m) | 0.39383 | **0.34887** | 11.4% better |
| rotation RMSE (rad) | 0.003446 | **0.002686** | 22.1% better |

## Reading

Whitening wins on both aggregates, but the per-axis rows matter more. Unwhitened
training **degraded rotation** — rx by 37.4% and ry by 26.8% — while cutting tz
by half. That is the predicted failure mode observed directly: rotation supplies
about 0.04% of the gradient, so the optimiser trades rotation accuracy away for
forward-translation accuracy. Whitened training improved ry by 46.9% instead and
degraded nothing materially.

The whitened model also starts better (tz RMSE 0.50 vs 0.77 at step 0) because
its zero-initialised head predicts the mean training motion rather than the
identity.

## Caveats

- 120 steps, one seed, one model. Indicative, not conclusive. Confirm over a
  full training run before quoting it as a headline result.
- The wall-clock difference (487s vs 225s) is CPU contention from a concurrent
  test run, not a property of either variant. Do not report it as a speedup.
- Scored on validation samples, not on integrated trajectory metrics. A lower
  per-frame RMSE should reduce ATE and t_rel, but drift compounds non-linearly
  and that has yet to be measured.

## Reproduce

Configs are `whiten: true|false` in `configs/m4.yaml`; the harness used here is
not committed, being a throwaway. The meaningful replication is a full training
run of each variant, which is the ablation for the final report.
