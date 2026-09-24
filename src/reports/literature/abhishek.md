# Paper summaries — Abhishek Laxman Jadhav

The two recurrent-VO papers M2 is built from and ablates against.

---

## [3] DeepVO: Towards End-to-End VO with Deep Recurrent Convolutional Neural Networks — Wang, Clark, Wen, Trigoni (Oxford), ICRA 2017

**Method.** The first fully end-to-end deep monocular VO. A CNN (FlowNet-pretrained, so it
begins with motion-sensitive features rather than object-recognition features) extracts
features from stacked consecutive frames; a two-layer LSTM then models sequential dynamics
and emits 6-DoF pose per step. **No component of the classical pipeline is retained** — no
feature detection, matching, RANSAC, triangulation or bundle adjustment.

**Dataset.** KITTI odometry benchmark.

**Key result.** Competitive with state-of-the-art classical methods. The significant part is
what disappears: monocular VO normally cannot recover absolute scale without external
information such as camera height, yet DeepVO learns scale implicitly from training data.

**Relevance.** M2's blueprint. Our architecture follows it — pretrained CNN over a stacked
pair, then recurrence over the sequence.

**The detail that decides whether M2 works.** The loss is
`L = ‖t̂ − t‖² + k·‖r̂ − r‖²` with a large `k`. Rotations between adjacent frames are tiny
in radians while translations are ~1 m, so an unweighted sum is dominated by translation
and the network learns to predict a straight line. **We measured the imbalance on our own
data rather than copying the published constant**: E[t²] = 2.00e-1 against E[r²] = 3.41e-5,
a ratio of ~5869:1. DeepVO's k=100 was tuned for KITTI's scale convention and would leave
rotation ~59× under-weighted here, so our configs use k=5900. Reproduce with
`scripts/measure_pose_stats.py`.

**Limitation.** Supervised, so it needs ground-truth poses — the axis on which it differs
from M3. And being end-to-end, it offers no interpretable intermediate state, which is
precisely LKN-VO's [6] criticism.

---

## [7] MINI-DROID-SLAM: Improving Monocular SLAM Using MINI-GRU RNN — Albukhari, El-Sayed et al., *Sensors* 25(17):5448, 2025

**Method.** Takes DROID-SLAM and replaces its ConvGRU recurrent update operator with a
**Mini-GRU**, a reduced-gate variant. Dense pixel-wise depth and bundle-adjustment
refinement are retained, so the change is isolated to the recurrent cell. Trained on a
single GPU.

**Dataset.** TartanAir. *(Note: unlike every other paper in our review, this one is not
evaluated on KITTI, so its numbers are not comparable to the rest of the table.)*

**Key result.** ATE comparable to the original DROID-SLAM at **reduced training time and
computational complexity**, with the authors emphasising trainability on a single GPU.

**Relevance.** Direct empirical support for M2's `rnn: lstm | gru | none` ablation. It is
evidence that the expensive recurrent cell is not necessarily where the accuracy comes
from — if a Mini-GRU matches a ConvGRU inside a full SLAM system, a GRU plausibly matches
an LSTM in our simpler pose regressor.

**Why the `none` setting matters most.** Setting `rnn: none` yields a pairwise model with
the same backbone as M4 without attention. Those two numbers are produced by two members
from independent code, so **if they agree it is mutual verification of both
implementations** — and a strong thing to be able to say in the viva.

**Citation fix.** The synopsis lists this as "PMC Open Access, 2024". The PDF is
*Sensors* 2025, 25, 5448. Correct it in the final reference list.
