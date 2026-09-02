# Deep_Monocular_VO — Deep Learning-Based Monocular Visual Odometry for Autonomous Spatial Mapping

[![ci](https://github.com/VaibhavBansal05/Deep_Monocular_VO/actions/workflows/ci.yml/badge.svg)](https://github.com/VaibhavBansal05/Deep_Monocular_VO/actions/workflows/ci.yml)

ICT-4442 Deep Learning Mini Project · Manipal Institute of Technology · Aug–Oct 2026

Four deep-learning architectures for 6-DoF monocular ego-motion estimation, trained and
evaluated on **Virtual KITTI 2** under one identical protocol.

| Model | Architecture | Owner |
|---|---|---|
| M1 | Classical ORB/Lucas-Kanade features + MLP regressor | Pranav Joshi |
| M2 | CNN (ResNet18) + GRU/LSTM recurrent pose regression | Abhishek Laxman Jadhav |
| M3 | Self-supervised PoseNet + DepthNet, photometric reprojection loss | Suryansh Bajaj |
| M4 | CNN + CBAM/SE convolutional attention | Vaibhav Krishna Bansal |

## Repository layout

```
DOCS/          course-issued documents — synopsis, guidelines, report template
REFERENCES/    the 9 reference papers (gitignored: 77 MB of PDFs, kept local)
src/           everything we build
  mvo/common/  shared base — split, geometry, metrics, training engine
  mvo/models/  one folder per model, single owner each
  configs/     one YAML per model, inheriting configs/base.yaml
  scripts/     train.py · eval.py · compare.py · prepare_data.py
  results/     metrics.json + trajectories + plots (committed — evidence)
  reports/     literature review, methodology sections, contribution log
```

See [PLAN.md](PLAN.md) for the full implementation plan and
[src/reports/CONTRIBUTIONS.md](src/reports/CONTRIBUTIONS.md) for the task matrix.

## Quick start

```bash
git clone <repo-url> && cd Deep_Monocular_VO/src
python -m venv .venv && source .venv/bin/activate
# EVERY member must do this once, with the email on their GitHub account.
# Wrong email -> commits show as an unlinked ghost author and cannot be verified.
git config user.name  "Your Name"
git config user.email "your-github-email@example.com"

cd .. && bash tools/setup.sh && cd src   # hooks + identity check + tests
python scripts/prepare_data.py --root data/vkitti2   # ~6-8 GB subset
python scripts/train.py --model m4 --config configs/m4.yaml
python scripts/eval.py  --model m4 --ckpt checkpoints/m4/best.pt
python scripts/compare.py                            # -> the comparison table
```

All commands run from `src/`.

## Repository rules

1. **`mvo/common/` is shared.** Changing it needs a PR reviewed by the whole team,
   because it changes everyone's results. Your model lives in `mvo/models/mX_*/`
   and nobody else touches it.
2. **The pose convention is frozen**: 6-vector `[tx,ty,tz,rx,ry,rz]`, translation
   first, axis-angle rotation, meaning `T_{t->t+1}`. Defined in
   `mvo/common/train/base_model.py`.
3. **The split is frozen**: `mvo/common/data/splits.py` is the only place scenes are
   listed. Changing it invalidates every result in `results/`.
4. **Merge with `--no-ff`, never squash.** Squash-merge collapses individual commits
   and destroys the per-member history that is being graded. Squash merging is
   disabled in repo settings.
5. **Never commit** data, checkpoints or `.npy` caches. Small artefacts
   (`results/*/metrics.json`, trajectory `.txt`, plots) *are* committed.
6. **Any borrowed code goes in `mvo/common/`** with an attribution header and an
   entry in [THIRD_PARTY.md](THIRD_PARTY.md).

## Evaluation protocol

- **Split** (by scene, no geometric leakage): train `Scene01/06/18` × {clone, morning,
  overcast, sunset} · val `Scene02/clone` · test `Scene20/clone`.
  `rain / fog / 15-deg / 30-deg` are held out entirely for the robustness ablation.
- **Metrics**: ATE RMSE (reported both `sim3`, scale-solved, and `se3`, scale-fixed),
  RPE translation/rotation at δ=1, KITTI `t_rel` / `r_rel`, plus depth AbsRel/RMSE/δ<1.25
  for M3. Params (M) and ms/frame for the compute trade-off.
- **Scale**: M1/M2/M4 use GT scale alignment; M3 self-scales via median depth. This
  asymmetry is why both `ate_sim3` and `ate_se3` are always reported.
