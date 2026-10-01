# M1 contributor instructions after the shared pose-objective update

M1 remains owned and implemented by its assigned contributor. Do not copy the
old six-axis MSE from early experiment notes.

## Required interface

- Subclass `BaseVOModel`, keep `seq_len = 2`, and return metric
  `[tx, ty, tz, rx, ry, rz]` poses with shape `(B, 1, 6)`.
- Use the shared dataset, frozen transform convention, train/validation split,
  training entry point and evaluation script. Do not make an M1-only loader or
  training loop.
- If the learned MLP directly regresses pose, call
  `mvo.common.train.pose_loss.geometric_pose_loss`. This gives Euclidean metric
  translation, SO(3) geodesic rotation, group-level scale balancing and the
  same drift-bias penalty as M2/M4.
- Call the output transform *per-axis standardisation*, not whitening. Full
  whitening would also decorrelate axes using a covariance matrix.

## Classical geometry and scale

- Report the pure classical essential-matrix result separately from any MLP
  regressor. Essential-matrix translation has direction but no monocular metric
  scale.
- Never use test ground-truth scale and present the result as deployable. If an
  oracle GT scale is shown for diagnosis, label it explicitly as oracle.
- A fair learned-scale option is to estimate scale only from training data or
  regress metric translation from training features. Validation chooses
  hyperparameters; test labels are evaluation-only.
- Include failure handling for too few correspondences, failed essential
  matrix recovery and near-zero translation. Record the failure rate.

## Bias and forward-motion checks

- Evaluate signed mean errors for all six axes in addition to ATE/RPE. A small
  consistent yaw or translation bias can create large integrated drift.
- The official split already uses all five VKITTI2 scenes. Weather/lighting
  variants improve appearance robustness but repeat the same underlying scene
  trajectories; do not count them as independent routes.
- Do not manually multiply the `tz` loss. Metric Euclidean translation already
  treats one metre equally in every direction, and the data naturally supplies
  more forward-motion signal.

## Minimum tests

- Synthetic known transform recovers the correct pose direction and convention.
- Horizontal flip changes the expected pose signs.
- Degenerate correspondence input returns a controlled failure, not NaNs.
- Loss is finite at identity and rotation is geodesic.

