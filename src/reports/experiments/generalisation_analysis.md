# Why the models drift: diagnosis after the whitened runs

> Historical analysis of earlier 25-epoch runs. Its checkpoint-selection
> observations and recommendations describe those runs. The current verified
> 60-epoch M2 and M4 results are in `src/results/m2/` and `src/results/m4/`;
> M4 now selects on validation trajectory error.

## 1. Whitening helped M4, and the M2 comparison is confounded

|  | ATE sim(3) | ATE SE(3) | t_rel | r_rel |
|---|---|---|---|---|
| M4 unwhitened | 24.46 | 92.67 | 42.54 | 8.46 |
| **M4 whitened** | **20.32** | **86.41** | **37.91** | **5.68** |
| M2 unwhitened | 25.26 | 85.56 | 37.83 | 8.68 |
| M2 whitened | 60.73 | **76.11** | 44.69 | 17.65 |

M4 improves on every trajectory metric. M2 improves on ATE SE(3) and per-frame
RPE translation but regresses on ATE sim(3) and rotational drift.

**These pairs are not a clean ablation.** Augmentation was inert in the earlier
runs and active in the later ones, so two things changed at once. Do not report
them as an ablation; the controlled comparison must be re-run.

## 2. The bottleneck is generalisation, not optimisation

M2's whitened run, in whitened units where the loss at initialisation
corresponds to predicting the training mean:

| epoch | train | val |
|---|---|---|
| 0 | 1.4694 | 1.5523 |
| 4 | 0.4815 | 2.0279 |
| 12 | 0.0513 | 1.2258 |
| 22 | — | **0.8734** (best) |
| 24 | 0.0226 | 0.9171 |

Variance explained: **98.5% on train, ~44% on validation**. The model fits the
three training scenes almost perfectly and transfers weakly. Nothing about the
loss, the optimiser or the schedule will fix that.

## 3. Most of the drift is systematic bias

Per-axis residual on the test sequence, and the effect of subtracting a constant
offset from every prediction:

| | bias/rmse worst axis | t_rel as predicted | t_rel bias removed |
|---|---|---|---|
| M4 | tz 0.69 | 37.91% | **17.72%** |
| M2 | ry 0.63 | 44.69% | **20.53%** |

M2's rotational drift falls from 17.65 to 5.56 the same way. Drift is dominated
by a near-constant per-frame offset, which accumulates linearly, rather than by
random error, which partially cancels.

The cause is **shrinkage toward the training mean**. An MSE-trained model that
cannot generalise hedges toward the average motion it saw. Scene20 travels
faster than the training average, so almost every frame is under-predicted by a
similar amount.

## 4. The bias cannot be calibrated away

A per-axis affine correction `g ~ a*p + b` was fitted on validation (Scene02)
and applied to test (Scene20):

| M4 | ATE sim(3) | ATE SE(3) | t_rel | r_rel |
|---|---|---|---|---|
| uncalibrated | 20.315 | 86.409 | 37.91% | 5.680 |
| val-calibrated | 36.082 | 146.171 | **65.78%** | 8.925 |

Substantially **worse**. The fitted coefficients show why: `rx` scale 0.0037 and
`rz` scale −0.83 mean those predictions are essentially uncorrelated with truth
on validation, so least squares drives them toward a constant and flips a sign.

The bias is therefore **scene-specific**, not a fixed property of the model, and
a correction learned on one scene does not transfer to another. This is a
negative result worth reporting: it confirms the bias is a symptom of failed
generalisation rather than a miscalibration that can be post-hoc corrected.

## 5. Checkpoint selection is picking the wrong epoch

M4's best checkpoint is **epoch 2 of 25**. Validation is Scene02/clone, 232
samples in 29 batches, so the per-frame validation loss is noisy enough that an
early lucky epoch wins over 23 later ones. M4's reported numbers therefore come
from a barely-trained model — and it still beat M2 on most metrics.

Selecting on a trajectory-level metric (validation ATE), or on a smoothed
validation loss, would very likely improve both models at zero training cost.

## Ranked recommendations

1. **Fix checkpoint selection** — select on validation ATE rather than noisy
   per-frame loss. Free, and M4 is currently reporting an epoch-2 model.
2. **Add regularisation** — `weight_decay` is 0.0. With a 98.5%/44% train/val
   split that is the obvious lever: weight decay, backbone freezing, or a
   smaller head.
3. **Reduce capacity** — 12.5M and 15.4M parameters against 1,056 unique frames.
   Freezing conv1–layer2 and training only layer3/layer4 plus the head is worth
   an ablation.
4. **More appearance diversity** — the six unused variations of the training
   scenes give 2.5x the data with no leakage, and appearance is exactly what is
   being overfitted.
5. **Accept the ceiling.** Three training trajectories, 1,056 unique frames.
   Published monocular VO trains on nine or more KITTI sequences with tens of
   thousands of frames. 1–2% t_rel is not reachable here, and the project's
   deliverable is the controlled comparison, not absolute accuracy.
