# Part C draft sections for M4 and the common results

This text is ready to place in the supplied Part C report template. The M4
methodology subsection belongs to Vaibhav. The four-model results use the
verified files listed below. The other model owners' methodology sections,
the integrated literature review, references, and signed contribution page
must be assembled separately.

## IV D M4 Convolutional attention model

M4 estimates relative camera motion from two consecutive RGB frames. We
concatenate the frames into a six-channel input and pass it through a
ResNet18 encoder whose first convolution is adapted for the image pair.
Convolutional Block Attention Modules (CBAM) follow the encoder's four
residual stages. Each module uses channel attention followed by spatial
attention. A learned residual gate, initialised to 0.1, limits disruption
to the pretrained encoder while allowing attention weights to receive
gradients. A pooled feature vector feeds a six-value pose head: three
translation values in metres and three axis-angle rotation values in
radians. The head predicts a standardised pose; the reported pose is
converted back to physical units.

Translation error is measured in Euclidean space and rotation error by
the geodesic angle between the predicted and reference rotations. Each
group is scaled by its spread in the training poses, and the objective also
penalises mean signed residual across a mini-batch to discourage systematic
drift. We trained for 60 epochs with seed 42, batch size 16, learning rate
0.0001, and full-precision arithmetic. The checkpoint with the lowest
validation trajectory error was from epoch 18 (counting from zero). M4 has
11.23 million parameters.

To isolate the effect of attention, we trained a second M4 model with the
same code revision, data, seed, training schedule, and precision, replacing
the four attention blocks with identity operations. Its selected checkpoint
was from epoch 16 and it has 11.19 million parameters. The two runs differ
in their attention setting; their saved training configurations otherwise
match.

## V Experimental setup for the shared comparison

All four models use the same Virtual KITTI 2 RGB subset at 128 by 416 pixels
and the same scene-based split. Scene01, Scene06, and Scene18 provide the
training sequences with clone, morning, overcast, and sunset appearances.
Scene02/clone is validation; Scene20/clone is the test route. Rain, fog,
15-degree-left, and 30-degree-right variants of Scene20 are reserved for
robustness evaluation. The table reports the evaluation outputs already
saved for each model, not a new training run.

## VI Results and discussion

Table I compares the selected checkpoints on Scene20/clone. Absolute
trajectory error (ATE) measures path error after alignment; Sim(3) can
adjust scale and SE(3) keeps scale fixed. Relative pose error (RPE) measures
local motion error. The KITTI-style relative translation and rotation
errors are `t_rel` and `r_rel`. Lower values are better for all error
columns. The full precision and parameter-count values are in
`COMPARISON_TABLE.md`.

**Table I. Four-model comparison on Scene20/clone**

| Model | Sim(3) ATE m | SE(3) ATE m | RPE trans m | RPE rot deg | t_rel % | r_rel deg/100 m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 ORB/LK and MLP | **24.209** | **68.320** | 0.3878 | 0.2182 | 36.61 | 8.4002 |
| M2 CNN and LSTM | 61.646 | 82.835 | **0.3474** | 0.2508 | 54.52 | 19.7597 |
| M3 PoseNet and DepthNet | 25.082 | 227.123 | 0.9363 | **0.1883** | 97.39 | 8.7570 |
| M4 CNN and CBAM | 25.471 | 75.273 | 0.3637 | 0.1965 | **35.67** | **7.8879** |

M1 has the lowest trajectory error on this test route. M4 has the lowest
relative translation and rotation drift in the table, while M2 has the
lowest local translation RPE and M3 the lowest local rotation RPE. M3 is
self-supervised from RGB and has unresolved metric scale; its relatively
small Sim(3) ATE and much larger SE(3) ATE illustrate why both alignments
must be reported. Sim(3) scale is solved using the reference trajectory
for evaluation. Predicted monocular depth does not establish metric pose
scale by itself. A measured depth score remains pending official depth
ground truth and a separate evaluation run.

The controlled M4 comparison shows a mixed effect of attention. On the
test route, CBAM has 25.471 m Sim(3) ATE versus 26.886 m without attention,
but 75.273 m SE(3) ATE versus 53.391 m without attention. Local translation
and rotation RPE are also lower without attention. Across the four held-out
weather and camera variants, CBAM has lower mean Sim(3) ATE (79.756 m
versus 102.128 m), mainly helped by fog and camera-view changes. The
no-attention model performs better on rain. Both have large error on the
30-degree-right view; CBAM reaches 189.036 m and no attention 214.495 m
Sim(3) ATE there. This view change is a concrete failure case. The runs
use one seed and one test route, so these differences describe this
experiment rather than a general statistical advantage.

The saved `ms/frame` values in the complete comparison table came from
different evaluation sessions. They should be treated as descriptive
measurements, not a hardware-controlled speed comparison.

### Result provenance

| Model | Test metrics source | Recorded training or evaluation revision |
| --- | --- | --- |
| M1 | `src/results/m1/metrics.json` | `009426b` |
| M2 | `src/results/m2/metrics.json` | `b67d37a070d383f0a52186b2e4e02f572c98b58a` |
| M3 | `src/results/m3/metrics.json` and `manifest.json` | `785e5ad933b3a0dd2ac96d940ebb981f0ab01709` |
| M4 CBAM | `src/results/m4/metrics.json` and `manifest_cbam_60.json` | `b67d37a070d383f0a52186b2e4e02f572c98b58a` |

M2 and M4 CBAM use the verified 60-epoch results. M4 no-attention metrics,
history, and robustness trajectories are under `src/results/m4/no_attention/`.
The archived checkpoints stay outside Git. No depth score or SE-attention
score is included.
