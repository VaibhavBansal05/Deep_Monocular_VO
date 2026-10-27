# M4 attention comparison

The M4 CBAM and no-attention runs use the same Virtual KITTI 2 subset,
train/validation/test split, code revision
`b67d37a070d383f0a52186b2e4e02f572c98b58a`, seed 42, batch size 16,
60 epochs, full-precision training, and validation trajectory-error checkpoint
selection. Their saved checkpoint configurations differ only in
`model.attention`. CBAM selected epoch 18; no attention selected epoch 16.
Both are independent 60-epoch runs. The locally held checkpoint archives are
`m4_cbam_60_artifacts.zip` and `m4_none_60_artifacts.zip`; Git ignores them.

## Test: Scene20/clone

Lower is better for every metric below. Absolute trajectory error (ATE)
measures the reconstructed path; Sim(3) alignment can adjust scale, while
SE(3) alignment keeps scale fixed. Relative pose error (RPE) measures local
frame-to-frame motion.

| Metric | CBAM | No attention | Lower result |
| --- | ---: | ---: | --- |
| ATE, Sim(3) (m) | 25.471 | 26.886 | CBAM |
| ATE, SE(3) (m) | 75.273 | 53.391 | No attention |
| RPE, translation RMSE (m) | 0.364 | 0.288 | No attention |
| RPE, rotation RMSE (degrees) | 0.196 | 0.179 | No attention |
| Relative translation error (%) | 35.672 | 33.662 | No attention |
| Relative rotation error (degrees/100 m) | 7.888 | 9.362 | CBAM |

CBAM reduces scale-aligned test ATE by **1.416 m** (5.3% relative to no
attention), but no attention has **21.882 m** lower scale-fixed ATE and lower
local translation and rotation RPE. Thus the attention change helps some
metrics and hurts others; this run does not support a blanket improvement
claim. The forward-translation bias is 0.239 m/frame with CBAM and
0.145 m/frame without attention on this test sequence.

## Robustness: held-out Scene20 conditions

The aggregate below is the unweighted mean of the four per-sequence scores
computed by `scripts/eval.py` at the recorded revision.

| Condition | CBAM Sim(3) ATE (m) | No attention Sim(3) ATE (m) |
| --- | ---: | ---: |
| Rain | 44.275 | 36.309 |
| Fog | 29.919 | 70.856 |
| 15-degree-left camera | 55.792 | 86.853 |
| 30-degree-right camera | 189.036 | 214.495 |
| **Four-condition mean** | **79.756** | **102.128** |

CBAM lowers the four-condition mean Sim(3) ATE by **22.372 m** and the mean
SE(3) ATE from 112.916 m to 93.916 m. No attention performs better on rain;
CBAM performs better on fog and both camera-view changes. Both models are
weak on the 30-degree-right view.

The runs use one seed and one test scene, so these are observed differences,
not a statistical estimate of attention's general benefit. The recorded
milliseconds per frame differ greatly between the two evaluation sessions;
their timing values should not be used as a speed comparison without a
matched hardware and measurement setup.

## Source files

- CBAM: `metrics.json`, `robustness/metrics.json`, `history.csv`, and
  `manifest_cbam_60.json` in this directory.
- No attention: corresponding files under `no_attention/`, including
  `effective_config.yaml`.
- Both full archives also include `best.pt` and predicted trajectories.
  No-attention checkpoint SHA-256:
  `3056ba8a0ceebc28d43ca2f57ff006e2d4e26bc73cd2672c107ca66267d7ba52`.
