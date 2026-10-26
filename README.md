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

## Where the work happens

**Code locally, train on Colab.** Everything except GPU training is CPU work — editing,
the contract tests (~1 s), lint, ORB feature extraction, metrics, plots. Only M2/M3/M4
training needs a GPU. M1 precomputes its features once and then trains an MLP on 192-d
vectors, so it runs fine on a laptop CPU.

For GPU runs, copy `notebooks/colab_train_template.ipynb` and change `MODEL`. It mounts
Drive, unpacks the shared 1.5 GB dataset archive, clones this repo at your branch, trains
with checkpoints written to Drive (so a disconnect costs one epoch, not the run), and
evaluates.

Two rules that matter:

- **Never edit model code in a Colab cell.** Edit locally, commit, push, re-clone. Code
  written only in a notebook is not in your commit history — which is what the project is
  graded on.
- **Never commit checkpoints.** `*.pt` stays in Drive. Do commit `results/<model>/metrics.json`,
  trajectories and plots; they are small and they are the evidence trail.

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
- **Scale**: M1/M2/M4 learn metric translation from pose labels. M3 trains from
  RGB alone and has unresolved absolute-scale ambiguity; its own depth prediction
  cannot recover metric scale. Every model reports both Sim(3)-aligned
  (`ate_sim3`, scale solved for evaluation) and SE(3)-aligned (`ate_se3`, scale
  fixed) trajectory error. Ground-truth median scaling, if used for M3 depth
  evaluation, does not change pose trajectories.
