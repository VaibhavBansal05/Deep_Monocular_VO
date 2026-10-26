# Final comparison protocol

Use the frozen split in `mvo/common/data/splits.py`, 128 x 416 Camera_0 RGB,
seed 42, validation-selected checkpoints, and the common `scripts/eval.py`
metrics. Keep the training commit, complete config, checkpoint epoch/hash,
dataset version, and test artifacts with every row. Do not select a model or
hyperparameters using the test route.

## Required for the four-model result

1. Compare one verified baseline for each of M1, M2, M3 and M4 on the same
   `Scene20/clone` test split. Include Sim(3) and SE(3) absolute trajectory
   error, relative pose error, KITTI drift, parameter count, and runtime.
   Interpret M3's scale ambiguity separately from the supervised models.
2. Compare M4 CBAM against `configs/m4_none.yaml` with the same seed, data,
   schedule, precision and evaluation procedure. This is the controlled
   attention comparison; the existing CBAM result alone cannot establish an
   attention benefit.
3. Evaluate M3 depth against the paired official VKITTI2 depth PNGs after the
   opt-in evaluator has been implemented and tested. This supplies a depth
   result, not metric scale for the pose trajectory.

## Additional comparisons, as time and compute allow

- M4 SE (`configs/m4_se.yaml`) against the matched no-attention and CBAM runs.
- M2 LSTM (`configs/m2.yaml`) against a matched GRU run. A no-recurrence M2
  run is useful but must be trained under the same M2 protocol; M4 with no
  attention is not a substitute for a controlled M2 recurrence ablation.
- M3 loss ablations from the matched 20-epoch control and variants in
  `reports/experiments/m3_analysis.md`.
- M1 learned regressor against a classical geometry baseline, where the same
  data and evaluation protocol can be used.
- Held-out weather and camera-angle robustness across models, only after
  every compared model is evaluated on the same variants.

The existing M2 60-epoch LSTM artifact is already a valid baseline run; it
does not need retraining for the four-model comparison. The older 25-epoch
loose result is archived under `results/m2/previous_25_epoch/`. The current
loose M4 result is still from an older run and must be reconciled with the
60-epoch CBAM artifact before the final table is generated.
