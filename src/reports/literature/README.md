# Literature Review

ICT-4442 Interim submission (Part B). Nine papers, summarised and compared.

Each row is expanded in the per-member files in this directory. Members own the papers
that bear on their own model, so the review doubles as viva preparation.

> **Before submission:** every figure quoted below is taken from the PDFs in
> `REFERENCES/`. Re-check the ones you own against the source — table extraction from
> PDFs is error-prone, and a wrong number in the report is worse than no number.

## Summary table

| # | Paper | Method | Dataset | Key result | Relevance to this project |
|---|---|---|---|---|---|
| 1 | Cabon et al., *Virtual KITTI 2*, arXiv 2020 | Unity-rendered photo-realistic clones of 5 KITTI tracking sequences; 10 variants each (weather, camera rotation); RGB, depth, flow, segmentation, 6-DoF poses | vKITTI 2 (synthetic), compared against real KITTI | Re-ran depth/pose/segmentation baselines on both; synthetic data is a usable proxy but a domain gap remains | **Our dataset.** Dense 100% depth and exact poses give an error floor no real dataset offers |
| 2 | Xie et al., *CoProU-VO*, GCPR 2025 | Combines target-frame uncertainty with **projected** reference-frame uncertainty in one probabilistic mask; ViT backbone; learns depth, uncertainty and pose jointly | KITTI, nuScenes | Outperforms prior unsupervised two-frame end-to-end methods, notably on highway scenes where others fail | Most recent work in our family; the natural extension to M3, and evidence that attention-style backbones help VO |
| 3 | Wang et al., *DeepVO*, ICRA 2017 | CNN (FlowNet-pretrained) → 2-layer LSTM → 6-DoF pose, fully end-to-end; no module of the classical pipeline retained | KITTI odometry | Competitive with classical methods; showed end-to-end VO is viable and learns scale implicitly | **M2's blueprint**, including the rotation loss weighting our models inherit |
| 4 | Zhou et al., *SfMLearner*, CVPR 2017 | DepthNet + PoseNet trained only by photometric view-synthesis loss; explainability mask for non-rigid regions | KITTI odometry | ATE 0.021±0.017 (Seq 09), 0.020±0.015 (Seq 10) over 5-frame snippets — beats ORB-SLAM (short) 0.064, behind ORB-SLAM (full) 0.014/0.012 | **M3's blueprint.** Establishes that pose is learnable with no labels at all |
| 5 | Godard et al., *Monodepth2*, ICCV 2019 | Three targeted fixes: minimum-over-reference reprojection loss (occlusions), full-resolution multi-scale sampling (artifacts), auto-masking (static pixels) | KITTI (Eigen split) | Abs Rel 0.115 monocular at 640×192; ablations isolate each component's contribution | **M3's loss design.** Also a model for how to structure our own ablations |
| 6 | Zhao et al., *LKN-VO*, Robotics & Autonomous Systems 2019 | Learning Kalman Network: CNN observation model plus a learned transition model, fused by the Kalman update rather than an RNN | KITTI | Learns absolute scale without geometric constraints; sparse representation for state correlations | The **alternative to M2's recurrence** — a principled filter instead of a learned hidden state |
| 7 | Albukhari et al., *MINI-DROID-SLAM*, Sensors 2025 | Replaces DROID-SLAM's ConvGRU with a Mini-GRU; retains dense depth and bundle adjustment | TartanAir | Comparable ATE to DROID-SLAM at lower training time and computational cost | Direct evidence for **M2's GRU-vs-LSTM ablation**: a cheaper recurrent cell can suffice |
| 8 | Zhi et al., *Unsupervised MVO with Multi-Scale Modelling*, Sensors 2022 | Densely-linked atrous convolutions to widen the receptive field without losing resolution, plus **non-local self-attention** for long-range dependencies | KITTI | Competitive with unsupervised SOTA; **state-of-the-art rotation estimation**, especially in rotating scenes | **Closest published analogue to M4.** Its rotation-specific gain is the hypothesis M4 tests |
| 9 | Messikommer et al., *Reinforcement Learning Meets VO*, ECCV 2024 | RL agent inside a classical VO pipeline choosing keyframes and grid size online; reward from pose error and runtime | Public VO benchmarks | Improves accuracy and robustness while removing weeks of manual heuristic tuning | Shows **learning can augment rather than replace** classical VO — the framing for M1 |

## What the comparison shows

**The field moved from replacing the pipeline to repairing it.** DeepVO [3] and SfMLearner [4]
argued the whole geometric pipeline could be learned end-to-end. The strongest later work is
narrower: Monodepth2 [5] keeps the SfMLearner architecture almost unchanged and wins on three
specific loss fixes, and RL-VO [9] leaves a classical pipeline intact while learning only its
heuristics. That trajectory is why our four models span architecture families rather than
chasing a single best design.

**Supervision level and scale handling are entangled.** Supervised models [3, 6] learn absolute
scale from pose labels. Unsupervised models [4, 5, 8, 2] recover structure without labels but
only up to an unknown global scale, so they are evaluated after median or trajectory alignment.
Any comparison that ignores this flatters the supervised side — which is exactly why our
protocol reports ATE both with scale solved (sim3) and with scale fixed (SE3).

**Temporal modelling is contested.** DeepVO [3] uses an LSTM, MINI-DROID-SLAM [7] shows a much
cheaper Mini-GRU suffices, and LKN-VO [6] argues a learned Kalman filter is the better-posed
choice because the state has an explicit meaning. M2's `rnn: lstm | gru | none` ablation tests
the first two directly.

**Attention is the least-settled axis, and the most recent.** Only [8] and [2] use it, [8] via
non-local self-attention and [2] via a ViT backbone. Neither isolates the attention module from
its other changes. **M4's controlled ablation — identical backbone, data, schedule and seed,
with CBAM/SE on or off — measures something these papers leave confounded.** That is this
project's clearest contribution.

**Everything here is evaluated on real KITTI except [7].** None of the nine evaluates on
Virtual KITTI 2 as the primary benchmark, so our numbers are not directly comparable to theirs.
Synthetic data removes sensor noise and gives 100% dense depth, so our results should be read as
an optimistic upper bound. We state this as a limitation rather than claiming parity.

## Gaps this project addresses

1. **No controlled attention ablation exists** in monocular VO. [8] and [2] both change several
   things at once. M4 isolates it.
2. **Cross-family comparison under one protocol is rare.** Published papers compare against
   reported numbers from different splits and resolutions. Our four models share a split,
   an input size, a metric set and a seed.
3. **Synthetic-data behaviour is under-reported.** vKITTI 2 [1] exists precisely to enable this,
   yet the VO literature still evaluates almost entirely on real KITTI.

## Citation corrections needed in the synopsis

- Reference [7] is cited as "PMC Open Access, 2024". The PDF is **Sensors 2025, 25, 5448**
  (received 15 Jul 2025, published 3 Sep 2025). Fix before the final report.
- Reference [8] is correct: **Sensors 2022, 22, 5193**.

## Ownership

| Member | Papers | Why these |
|---|---|---|
| Pranav Joshi | [6] LKN-VO, [9] RL Meets VO | Both keep classical geometry and add learning on top — M1's design question |
| Abhishek Jadhav | [3] DeepVO, [7] MINI-DROID-SLAM | The two recurrent-VO papers M2 is built from and ablates against |
| Suryansh Bajaj | [4] SfMLearner, [5] Monodepth2 | M3's architecture and its loss function, respectively |
| Vaibhav Bansal | [1] vKITTI 2, [2] CoProU-VO, [8] Multi-Scale MVO | The dataset, plus the two attention-based works M4 is positioned against |
