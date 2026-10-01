# M3 contributor instructions after the shared training-protocol update

M3 remains owned and implemented by its assigned contributor. It is
self-supervised for training, so do not force the supervised M2/M4 pose loss
into its main objective.

## Required interface

- Subclass `BaseVOModel`, return metric-contract poses with shape
  `(B, S-1, 6)`, and return predicted depth/disparity needed by the loss.
- Train with RGB photometric reconstruction, not ground-truth pose or depth.
  Ground-truth depth is for evaluation only.
- Use the shared dataset and split. Training does not require depth PNGs;
  depth evaluation does.

## Geometry and losses

- Use differentiable back-projection and reprojection with intrinsics scaled to
  the actual image or pyramid resolution.
- Combine SSIM and L1 photometric residuals, minimum reprojection across source
  frames, automasking for stationary pixels and edge-aware disparity
  smoothness. Mask invalid/out-of-view samples.
- Verify the exact pose direction expected by the warping function. The shared
  contract is `T_{t->t+1}`; an inverse used silently will produce plausible but
  incorrect warps.
- Axis-angle coordinate MSE is not appropriate for a diagnostic supervised
  pose ablation. If such an ablation is added, use
  `geometric_pose_loss`; keep it clearly separate from the self-supervised
  result.

## Monocular scale and depth evaluation

- Monocular photometric training has an inherent joint depth/translation scale
  ambiguity. Report Sim(3)-aligned ATE for shape and also clearly label any raw
  SE(3) metric trajectory result.
- For depth, report at least AbsRel, SqRel and RMSE on valid pixels, with the
  chosen median-scaling or metric-scaling protocol stated explicitly. Do not
  use test depth to tune or train the network.
- The local RGB subset alone has no depth files. Download and prepare VKITTI2's
  official dense depth archive before final M3 depth evaluation; the original
  dataset does provide 16-bit depth PNGs in centimetres.
- `needs_depth=True` currently makes the loader require depth during training.
  Split this concern before implementing M3: training should use
  `need_depth=False`, while the depth-evaluation loader uses
  `need_depth=True`. Do not waste training I/O on GT depth that the objective
  must not consume.

## Data diversity and evaluation

- VKITTI2 contains five unique scenes (01, 02, 06, 18 and 20), all already used
  by the project split. Variants provide appearance/domain robustness but not
  additional independent routes.
- Evaluate the clone test route plus weather and camera-angle robustness
  variants, while avoiding claims that these are independent trajectory tests.
- Add tests for identity warp, known pixel translation, depth validity masks,
  multi-scale intrinsics and finite gradients near zero pose.

