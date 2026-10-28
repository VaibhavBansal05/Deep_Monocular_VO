# Individual Contribution Log

Required at Interim (0.5 marks, *"verified via GitHub commits / task log"*) and at Final
as a signed statement. **Update this in the same PR as the work.** Cross-checked against
`git shortlog -sne`.

## Task matrix

| Task | Pranav | Abhishek | Suryansh | Vaibhav |
|---|:--:|:--:|:--:|:--:|
| **Base — shared infrastructure** | | | | |
| Repo scaffold, CI, model contract, training engine | | | | ● |
| vKITTI 2 acquisition + hosting for the team (ops, not code) | | | | ● |
| vKITTI 2 dataset class + preprocessing pipeline | ● | | | |
| SE(3) geometry + projection/warping | | ● | | |
| Metrics (ATE/RPE/depth), trajectory IO, compare.py | | | ● | |
| **Models** | | | | |
| M1 — Classical + MLP | ● | | | |
| M2 — CNN + GRU/LSTM | | ● | | |
| M3 — Self-supervised PoseNet/DepthNet | | | ● | |
| M4 — CNN + CBAM/SE attention | | | | ● |
| **Reports** | | | | |
| Literature review (2–3 papers each) | ○ | ○ | ○ | ○ |
| Methodology subsection (own model) | ○ | ○ | ○ | ○ |
| Results, ablations, error analysis | ○ | ○ | ○ | ○ |
| Slides + presentation | ○ | ○ | ○ | ○ |

● = owner · ○ = to be filled in as completed

## Weekly log

Append one line per member per week. Terse is fine; consistency is the point.

| Week | Member | What was done | PRs |
|---|---|---|---|
| 2 Sep | Vaibhav | Repo scaffold, base contracts, CI, plan | #1 |
| 16 Sep | Suryansh | Implemented ATE/RPE/KITTI/depth metrics, TUM trajectory IO, comparison table and tests | #2 |
| 21 Sep | Abhishek | Implemented the SE(3) geometry and projection/warping utilities with tests; added the M2 CNN and recurrent backbone | #4, M2 backbone commit a78a82d |
| 22 Sep | Vaibhav | Reviewed and merged base PRs; fixed M4 pose scaling and rotation weight from measured statistics | m4/fix-pose-scaling |
| 22 Sep | Vaibhav | Applied the same measured rotation weight to M2's config | fix/m2-rot-weight |
| 24 Sep | Vaibhav | Literature review: 9-paper comparison table and per-member summaries | docs/literature-review |
| 2 Oct | Vaibhav | Developed and evaluated M4 attention model and supervised training fixes; compared available M1, M2 and M4 interim results | #3, #5, #11, #12, #13 |
| 3 Oct | Pranav | Implemented M1 feature extraction, cache, pose regressor and tests; recorded its evaluation results | #14, #15, #16 |
| 11 Oct | Suryansh | Implemented M3 ResNet18 DepthNet/PoseNet, RGB-only multi-scale loss, CPU tests/smokes, diagnostics, matched ablation configs and GPU runners; prepared separately tested shared integration patches; GPU results pending | #17 |
| 11 Oct | Suryansh | Shared checkpoint-resume integration: scheduler/scaler/RNG restoration, exact CPU epoch-boundary replay, CSV/schema validation, atomic checkpoint writes and consistent best/last state; tests cover shuffled loaders and failed resumes/writes | #18 |
| 24 Oct | Vaibhav | Fixed M3 diagnostic image rendering after the Kaggle pilot exposed a boolean-mask plotting error; added a rendering regression test | fix/m3-visualization-mask |
| 27 Oct | Suryansh | Prepared the fresh M3 Kaggle runner for the public RGB subset and implemented opt-in unique-frame depth evaluation, provenance, tests and scale disclosure | #22 (merged) |
| 27 Oct | Suryansh | Ran a fresh 200-step GPU pilot and the 60-epoch canonical M3 training on his Kaggle account; selected epoch 40 by validation Sim(3) ATE, evaluated test and robustness poses, backed up outputs and recorded measured pose evidence | Kaggle Version 2; codex/m3-canonical-results |
| 28 Oct | Suryansh | Verified official VKITTI 2.0.3 depth against Naver's MD5, prepared a private attributed evaluation subset, ran the merged M3 depth evaluator using his validation-selected checkpoint, and recorded 837 unique test-frame scores and provenance | Kaggle depth Version 1; codex/m3-official-depth-results |
| 28 Oct | Vaibhav | Ran the 60-epoch M4 no-attention comparison on Kaggle; archived its checkpoint and verified test and robustness results against the matched CBAM run | m4/kaggle-none-runner |
| 28 Oct | Vaibhav | Promoted the verified M2 result, checked M4 archive provenance, generated the four-model comparison, and drafted the M4 results and discussion for Part C | final/verified-comparison-report |

## Sign-off

Interim (25 Sep 2026): ______  ______  ______  ______

Final (31 Oct 2026): ______  ______  ______  ______
