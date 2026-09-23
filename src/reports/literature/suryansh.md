# Paper summaries — Suryansh Bajaj

M3's architecture and its loss function.

---

## [4] Unsupervised Learning of Depth and Ego-Motion from Video — Zhou, Brown, Snavely, Lowe (UC Berkeley / Google), CVPR 2017

**Method.** The paper that made M3's category possible. Two networks — a single-view
**DepthNet** and a multi-view **PoseNet** — are trained jointly with **no labels at all**.
The supervisory signal is view synthesis: predicted depth and predicted pose are used to
warp a nearby frame onto the target, and the photometric difference is the loss. An
**explainability mask** is learned alongside, down-weighting pixels that violate the
static-scene assumption. The two networks are coupled only through the loss, so at test
time each runs independently.

**Dataset.** KITTI (odometry split for pose, Eigen split for depth).

**Key result.** ATE over 5-frame snippets, lower is better:

| Method | Seq. 09 | Seq. 10 |
|---|---|---|
| ORB-SLAM (full) | 0.014 ± 0.008 | 0.012 ± 0.011 |
| ORB-SLAM (short) | 0.064 ± 0.141 | 0.064 ± 0.130 |
| Mean Odometry | 0.032 ± 0.026 | 0.028 ± 0.023 |
| **SfMLearner** | **0.021 ± 0.017** | **0.020 ± 0.015** |

It beats both baselines sharing its input setting, and falls short only of ORB-SLAM (full),
which uses strictly more data through loop closure and relocalisation. Depth is comparable
to supervised methods trained with ground-truth depth or pose.

**Relevance.** M3's architecture, essentially unchanged. Also the method the Virtual KITTI 2
authors [1] chose to validate their dataset, via the same PyTorch port we borrow geometry
utilities from.

**Limitation that defines the comparison.** Photometric loss cannot recover absolute scale —
scaling the scene and the translation together leaves the reprojection identical. M3
therefore needs median scaling at evaluation while M1/M2/M4 use ground-truth scale
alignment. **This asymmetry must be stated in the results table**, which is why our protocol
reports ATE both with scale solved (sim3) and fixed (SE3).

---

## [5] Digging Into Self-Supervised Monocular Depth Estimation — Godard, Mac Aodha, Firman, Brostow (UCL/Caltech/Niantic), ICCV 2019

**Method.** Deliberately *not* a new architecture. The authors argue the field was adding
complexity where simple fixes would do, and contribute three:

1. **Minimum reprojection loss** — take the *minimum* photometric error over reference
   frames rather than the average. Averaging blurs occlusions into the loss; the minimum
   selects whichever view actually saw the pixel.
2. **Full-resolution multi-scale sampling** — upsample lower-scale depth to input
   resolution before computing loss, removing texture-copy artifacts and depth holes.
3. **Auto-masking** — drop pixels whose warped error exceeds their *unwarped* error. This
   removes static frames and objects moving with the camera, which otherwise teach the
   network to predict infinite depth.

**Dataset.** KITTI, Eigen split.

**Key result.** Abs Rel **0.115** monocular at 640×192; 0.132 without ImageNet pretraining.
Ablations isolate each component separately.

**Relevance.** M3's loss function, item for item — `min_reproj`, `automask` and
`num_scales` in `configs/m3.yaml` are these three ideas. Pretraining is not optional: the
0.132-vs-0.115 gap is on full KITTI, and our training split is ~4.2k samples, so from
scratch it will not converge in the time available.

**Also a model for how to write our results section.** This paper's influence comes from
disciplined ablation rather than a novel architecture — the same structure M4's
attention-on/off comparison follows.

**Licence note.** Monodepth2's code is released for **non-commercial research use only**.
Anything adapted from it must be recorded in `THIRD_PARTY.md` with the header convention,
per guidelines Sec. 9.
