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
| 3 Sep | Vaibhav | vKITTI2 acquisition + shared archive; Colab template | main |
| 3 Sep | Vaibhav | M4 backbone (ResNet18 6-ch + pose decoder) + tests | m4/backbone |

## Sign-off

Interim (25 Sep 2026): ______  ______  ______  ______

Final (31 Oct 2026): ______  ______  ______  ______
