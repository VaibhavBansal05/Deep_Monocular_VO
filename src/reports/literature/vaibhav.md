# Paper summaries — Vaibhav Krishna Bansal

Covering the dataset and the two attention-based works M4 is positioned against.

---

## [1] Virtual KITTI 2 — Cabon, Murray, Humenberger (NAVER LABS Europe), arXiv:2001.10773, 2020

**Method.** Photo-realistic re-creation of five KITTI tracking sequences in the Unity
engine, updated from the original Virtual KITTI with improved lighting and
post-processing. Each of the five clones is rendered in ten variants — weather (fog,
rain, morning, overcast, sunset) and camera rotations (15°/30°, left/right). Every frame
carries RGB, depth, class and instance segmentation, optical flow, scene flow, camera
intrinsics/extrinsics and vehicle locations, for two cameras.

**Dataset.** Itself a dataset paper; validated against the real KITTI tracking benchmark.

**Key result.** The authors re-ran the original Virtual KITTI experiments plus new ones on
stereo matching, monocular depth, camera pose estimation and semantic segmentation —
including SfMLearner [4] via the ClementPinard PyTorch port. Synthetic data is shown to be
a usable proxy for real data, while a measurable domain gap remains.

**Relevance.** This is our dataset. Two properties matter for us. Depth is **100% dense**,
where LiDAR ground truth covers roughly 5% of pixels, so M3 can be evaluated against depth
it never trained on. And poses are exact by construction rather than GPS/INS-derived.

**Limitation we inherit, and must state.** Results on vKITTI 2 are an optimistic upper
bound. No sensor noise, no rolling shutter, no exposure variation. The paper is explicit
that a domain gap persists, so our numbers should not be compared directly against
published KITTI results.

**Measured, not assumed.** The `extrinsic.txt` files store ~7 significant digits, so their
rotation matrices are orthonormal only to ~5e-7. Chaining ground-truth relative poses
across Scene20's 837 frames reproduces the ground-truth trajectory to ~2e-3 m. That is the
precision floor of the dataset itself, and no model can be meaningfully scored below it.

---

## [8] An Unsupervised Monocular VO Based on Multi-Scale Modelling — Zhi, Yin, Li, Pang, *Sensors* 22(14):5193, 2022

**Method.** An unsupervised depth-and-pose framework built on two ideas. **Densely-linked
atrous convolutions** widen the receptive field without downsampling, so large and small
objects are both representable without losing image detail. **Non-local self-attention**
then models long-range dependencies across the feature map.

**Dataset.** KITTI.

**Key result.** Competitive with unsupervised state of the art and comparable to supervised
and model-based methods. The headline claim is **state-of-the-art rotation estimation**,
with the gain concentrated in rotating scenes.

**Relevance.** The closest published analogue to M4, and the source of its central
hypothesis. The claim that attention helps *rotation specifically* is testable: rotation is
the harder component precisely because inter-frame angles are tiny (measured mean |r| =
0.0044 rad on our training split against mean |t| = 0.61 m), so the signal is easily
swamped. If CBAM helps M4, the rotation metrics should improve more than translation.

**What it leaves confounded — and what M4 fixes.** Atrous convolutions and non-local
attention are introduced together, so the paper cannot attribute the gain to either. M4
changes exactly one thing: `attention: cbam | se | none` over an identical backbone, data,
schedule and seed.

---

## [2] CoProU-VO — Xie et al., GCPR 2025

**Method.** Unsupervised two-frame VO whose contribution is uncertainty handling. Prior
work masks unreliable pixels using single-frame uncertainty; CoProU **projects the
reference frame's uncertainty into the target frame** and combines it with the target's own
under a probabilistic formulation, so uncertainty propagates across time. Built on a vision
transformer backbone, learning depth, uncertainty and pose jointly. Inherits Monodepth2's
auto-masking [5] and builds on SC-Depth.

**Dataset.** KITTI and nuScenes.

**Key result.** Significant improvement over prior unsupervised monocular two-frame
end-to-end methods, with the largest margin on **highway scenes where other approaches
fail** — the setting where dynamic objects most often violate the static-scene assumption.
Ablations confirm cross-frame propagation is the active ingredient.

**Relevance.** The most recent work in our family and the natural extension to M3, listed as
a stretch goal. It is also the second piece of evidence that attention-based backbones suit
VO, supporting M4's premise.

**Why the problem barely applies to us — and why that is worth saying.** CoProU exists
because dynamic objects break the static-scene assumption. vKITTI 2 contains moving
vehicles, so the issue is present, but synthetic rendering removes the photometric noise
that makes uncertainty estimation hard in practice. A useful honest line for the
discussion: our setting understates the problem this paper solves.
