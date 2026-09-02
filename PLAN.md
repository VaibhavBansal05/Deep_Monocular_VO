# Deep_Monocular_VO — Implementation Plan

Deep Learning-Based Monocular Visual Odometry for Autonomous Spatial Mapping (ICT-4442)

## 0. Constraints this plan is written against

From `DOCS/ICT_4442_Mini_Project_Guidelines.pdf`:

- Commit history is a **graded artifact**: 0.5 marks at Interim ("verified via GitHub
  commits / task log") and it cross-checks the Individual Contribution Statement.
  The guidelines explicitly say: *"teams should use a shared repository with
  per-member commits, not a single push at the end."*
- Final phase (6 marks) can swing **±20% per member** on individual viva. Each member
  is questioned on their own model. So each member must *write* their own model.
- Sec. 9: external code / tutorials / LLM use **must be cited in the report**, with the
  exact place stated. Turnitin ≤15% plagiarism, ≤20% AI.
- Interim needs: 8–10 paper lit-review table, preprocessing pipeline, **≥2 models**
  with preliminary results, contribution log.

From `DOCS/MVO_Synopsis.pdf`: 4 models, one owner each, one dataset (vKITTI 2),
**identical split and metrics**, ATE + RPE (+ depth RMSE for Model 3), 128x416 input.

## 1. Repo layout

Repo root holds `DOCS/` (course documents), `REFERENCES/` (papers, gitignored)
and `src/` (everything we build). All commands below run from `src/`.

The whole design goal: a **thin shared base** everyone depends on, and **four fat model
folders nobody else touches**. Merge conflicts are what destroy per-member commit
attribution, so ownership is enforced by directory.

```
src/
  mvo/common/                      # the "base" — frozen by 10 Sep, changes only via PR
    data/
      vkitti.py                # Dataset: returns the standard batch dict
      splits.py                # THE canonical train/val/test split (single source of truth)
      transforms.py            # resize 128x416, ImageNet norm, K rescaling
    geometry/
      se3.py                   # euler/quat/so3 <-> 4x4, compose, invert, relative<->absolute
      projection.py            # backproject, project, inverse-warp (needed by M3)
    metrics/
      ate.py                   # Umeyama sim(3)/scale alignment + ATE RMSE
      rpe.py                   # RPE trans/rot, KITTI-style t_rel/r_rel
      depth.py                 # AbsRel, SqRel, RMSE, delta<1.25 (M3 only)
    train/
      base_model.py            # BaseVOModel ABC — the contract
      engine.py                # generic train/val loop, AMP, ckpt, seeding
      registry.py              # @register("m4") so scripts/train.py --model m4 works
    io/
      trajectory.py            # read/write TUM-format .txt (evo-compatible)
      logging.py
  mvo/models/
    m1_classical_mlp/          # OWNER: Pranav Joshi
    m2_cnn_rnn/                # OWNER: Abhishek Laxman Jadhav
    m3_selfsup_posedepth/      # OWNER: Suryansh Bajaj
    m4_cnn_attention/          # OWNER: Vaibhav Krishna Bansal
  configs/                     # one YAML per model, shared base.yaml
  scripts/
    prepare_data.py            # download + subset + cache
    train.py                   # --model {m1,m2,m3,m4}
    eval.py                    # -> results/<model>/metrics.json (fixed schema)
    compare.py                 # reads all metrics.json -> the ONE comparison table
    plot_trajectories.py
  results/<model>/             # metrics.json, pred_traj.txt, plots (COMMITTED — small)
  notebooks/                   # Colab runners, one per member
  reports/
    literature/<member>.md     # each member writes their own 2-3 paper summaries
    methodology/<model>.md     # each member writes their own report subsection
    CONTRIBUTIONS.md           # the task matrix, updated continuously
  tests/
  data/                        # GITIGNORED
  checkpoints/                 # GITIGNORED
  CODEOWNERS
```

`CODEOWNERS` doubles as machine-readable evidence of who owns what:

```
/src/mvo/common/    @vaibhav @pranav @abhishek @suryansh
/src/mvo/models/m1_*/  @pranav
/src/mvo/models/m2_*/  @abhishek
/src/mvo/models/m3_*/  @suryansh
/src/mvo/models/m4_*/  @vaibhav
```

## 2. The base contracts (freeze these first — everything else depends on them)

### 2.1 Batch dict — what `vkitti.py` yields

```python
{
  "imgs":      FloatTensor (S, 3, 128, 416),   # S=2 for M1/M4, S=5..10 for M2/M3
  "K":         FloatTensor (3, 3),             # RESCALED to 128x416
  "K_inv":     FloatTensor (3, 3),
  "rel_pose":  FloatTensor (S-1, 4, 4),        # T_{t -> t+1}, SE(3), GT
  "abs_pose":  FloatTensor (S, 4, 4),          # world->cam, GT
  "depth":     FloatTensor (S, 1, 128, 416),   # metres; only loaded if cfg.need_depth
  "seq_id":    str,                            # "Scene20/clone"
  "frame_idx": LongTensor (S,)
}
```

Every model consumes this. Nobody writes their own loader.

### 2.2 Model contract

```python
class BaseVOModel(nn.Module):
    def forward(self, batch) -> dict:
        # MUST contain "pose": (B, S-1, 6) = [tx,ty,tz, rx,ry,rz] axis-angle,
        # meaning T_{t->t+1}. May contain "depth", "uncertainty", etc.
    def compute_loss(self, batch, out) -> dict:
        # MUST contain "total"; other keys are logged as-is
```

Fixing the pose parameterisation as **6-vector, translation-first, axis-angle,
t→t+1** is the single most important decision here. Every sign/convention bug in VO
projects comes from four people picking four conventions. `common/geometry/se3.py`
owns the conversion and is unit-tested.

### 2.3 The split (single source of truth, `common/data/splits.py`)

vKITTI 2 has 5 scenes (Scene01/02/06/18/20) x 10 variations x 2 cameras.
Split **by scene** so there is zero geometric leakage:

| Split | Scenes | Variations | Approx frames (Camera_0) |
|---|---|---|---|
| train | Scene01, Scene06, Scene18 | clone + morning + overcast + sunset | ~4,200 |
| val   | Scene02 | clone | ~230 |
| test  | Scene20 | clone | ~840 |

Test is **clone only** so all four models are scored on identical trajectories.
Fog / rain / 15-deg / 30-deg variations are held out entirely as an optional
**robustness ablation** (great material for Results & Discussion, and for the viva).

### 2.4 Metrics + scale (`common/metrics/`)

- **ATE RMSE** after Umeyama alignment. Two numbers reported per model:
  `ate_sim3` (7-DoF, scale solved) and `ate_se3` (6-DoF, scale fixed).
- **RPE** trans (m) / rot (deg) at delta=1 frame, plus KITTI `t_rel` (%) and
  `r_rel` (deg/100m) over 100..800m windows.
- **Depth**: AbsRel, SqRel, RMSE, delta<1.25 — Model 3 only.
- **Scale handling** (as the synopsis states): M1/M2/M4 use per-sequence GT scale
  alignment; M3 uses median depth scaling. This asymmetry must be stated explicitly
  in the report or the comparison is unfair — flag it in the table footnote.
- Cost: params (M), FLOPs, ms/frame on a fixed device. Rubric wants "computational
  trade-offs".

### 2.5 Output schema (`results/<model>/metrics.json`)

```json
{"model":"m4","owner":"Vaibhav Krishna Bansal","split":"test",
 "ate_sim3":0.0,"ate_se3":0.0,"rpe_trans":0.0,"rpe_rot_deg":0.0,
 "t_rel_pct":0.0,"r_rel_deg_per_100m":0.0,
 "params_m":0.0,"ms_per_frame":0.0,"seed":42,"commit":"<sha>"}
```

`compare.py` globs these into the one comparison table the rubric demands. Nobody
hand-types results into the report — that is how fabricated-result accusations start.

## 3. Reference code: what to borrow from where

The papers in `REFERENCES/` map to these public repos. **Borrow geometry and eval
utilities** (boring, easy to get subtly wrong, not what you are graded on).
**Write the model architectures yourself** (that is exactly what the viva tests).

| Repo | Paper | Take | License |
|---|---|---|---|
| `ClementPinard/SfmLearner-Pytorch` | [4] Zhou 2017 (cited *in* the vKITTI2 paper) | `inverse_warp.py` (pose_vec2mat, projective warp), `loss_functions.py`, `kitti_eval/pose_evaluation_utils.py` | MIT — safest |
| `nianticlabs/monodepth2` | [5] Godard 2019 | `layers.py` (BackprojectDepth, Project3D, SSIM, disp_to_depth), min-reprojection + auto-mask logic, ResNet encoder/decoder shape | **Non-commercial research only** — fine for coursework, MUST cite |
| `ChiWeiHsiao/DeepVO-pytorch` | [3] Wang 2017 | Structural reference only for M2 (CNN→LSTM stacking, loss weighting k=100 for rotation). Read it, don't paste it. | MIT |
| `jchao-xie.github.io/CoProU` | [2] Xie 2025 | Uncertainty-masked photometric loss — an M3 *stretch goal* / future-work section | check on release |
| `MichaelGrupp/evo` | — | Use as an **independent check** that our ATE/RPE match a standard tool | GPLv3 — use as CLI, do not vendor |
| `uzhrpg/rl_vo` | [9] | Future Work discussion only | GPLv3 |

Rules to stay clean under Sec. 9:
1. Anything adapted goes in `common/` (not in a model folder) with a header comment:
   `# Adapted from <repo> @ <commit>, <license>. Modified: <what>.`
2. Maintain `THIRD_PARTY.md` at repo root listing every borrowed file — this becomes a
   report appendix directly.
3. Never vendor GPL code into the repo (evo, rl_vo). Call evo as a CLI in a notebook.
4. Same file logs LLM usage per Sec. 9.

## 4. Bootstrapping — the part most teams get wrong

If one person builds the base, that person has 40 commits and everyone else has 3 by
Interim, and the 0.5 contribution mark suffers. So **the base itself is split four
ways**, in parallel, in week 1:

| Member | Base component | Their model |
|---|---|---|
| Vaibhav | repo scaffold, `train/engine.py`, `registry.py`, CI, CODEOWNERS | M4 attention |
| Pranav | `scripts/prepare_data.py`, `data/vkitti.py`, `transforms.py` (K rescaling!) | M1 classical+MLP |
| Abhishek | `geometry/se3.py`, `geometry/projection.py` + their unit tests | M2 CNN+RNN |
| Suryansh | `metrics/{ate,rpe,depth}.py`, `io/trajectory.py`, `compare.py` | M3 self-supervised |

Each person's base assignment is *adjacent to their model*: Suryansh owns depth
metrics and also owns the depth model; Abhishek owns SE(3) and also owns the
sequence model. So the base work is genuine preparation, not a chore.

These four pieces are near-independent — 4 branches, 4 PRs, minimal conflicts.
Interfaces are agreed **before** anyone codes (a 45-minute meeting settling §2.1–2.5
saves a week).

## 5. Git workflow for clean attribution

1. One GitHub repo, 4 collaborators. `main` protected: no direct pushes, PR + 1 review.
2. **Each member sets `git config user.email` to the email on their GitHub account.**
   Otherwise commits show as an unlinked ghost author and the checker cannot verify them.
   Verify with `git shortlog -sne` monthly.
3. Branch naming: `m1/feature-tracker`, `base/se3-utils`, `docs/lit-review-pranav`.
4. **Merge with `--no-ff` merge commits, never squash.** Squash-merge collapses 12
   individual commits into 1 and destroys exactly the granular history being graded.
   Turn off "allow squash merge" in GitHub repo settings.
5. Never push someone else's code under your name. If you pair, use
   `Co-authored-by: Name <email>` in the commit trailer — GitHub attributes both.
6. Commit small and often; ~3+ commits/week each, real messages, from day 1. A single
   push at the end is called out by name in the guidelines.
7. `reports/CONTRIBUTIONS.md` updated in the *same PR* as the work, not retroactively.
8. Tag `interim-2026-09-25` and `final-2026-10-31` so the checker sees exact state.
9. Cheap CI (`pytest tests/` + ruff) on PRs — a green check per PR is more contribution
   evidence, and it stops a broken `common/` from blocking three people.

## 6. Per-model plan

**M1 — Classical + MLP (Pranav).** ORB (or Shi-Tomasi + Lucas-Kanade) correspondences
between t and t+1 → fixed-length descriptor of the flow field (e.g. bin the image into
an 8x8 grid, average dx/dy + inlier count per cell → 192-d) → MLP → 6-DoF. Feature
extraction is deterministic, so **precompute features once** to a `.npy` cache; then
training is seconds on CPU. Ablate against a 5-point essential-matrix + GT-scale
baseline to show what the MLP adds. Risk: this is the least "deep" model, so the MLP
must be substantive (BN, dropout, proper tuning), not a 2-layer toy.

**M2 — CNN + GRU/LSTM (Abhishek).** ResNet18 (ImageNet-pretrained) on stacked
[I_t, I_{t+1}] (6-channel — inflate conv1 weights by duplicating and halving) →
per-step feature → 2-layer LSTM/GRU (hidden 512) → 6-DoF per step. Sequence length 5,
stateful across a sequence at eval. DeepVO's loss weighting matters: `L = ||t̂-t||² +
k·||r̂-r||²` with k≈100. Deliverable ablation: GRU vs LSTM vs no-recurrence (the
last one is essentially M4-without-attention, which makes the comparison table
much more interesting).

**M3 — Self-supervised PoseNet + DepthNet (Suryansh).** The hardest and most
time-hungry — start first. DepthNet = ResNet18 encoder + monodepth2-style decoder
with 4 scales; PoseNet = ResNet18 on 2 frames → axis-angle + translation.
Loss = 0.85·SSIM + 0.15·L1 photometric with **min-over-reference-frames** reduction,
+ edge-aware smoothness (1e-3), + auto-masking. Needs correct K after resize.
Reports depth RMSE too. Because vKITTI 2 has 100% dense GT depth, this model can be
evaluated against GT depth even though it never trains on it — that is a genuinely
strong Results section. Stretch: add CoProU-style uncertainty masking.

**M4 — CNN + CBAM/SE attention (Vaibhav).** Same ResNet18 6-channel backbone as M2 but
with CBAM (channel + spatial) or SE blocks inserted after each residual stage,
2-frame input, direct pose regression. **The controlled comparison is the whole
point**: identical backbone/data/schedule with attention on vs off isolates the
attention contribution. Ship attention-map visualisations over the test sequence —
they are the best single slide in the presentation, and they show whether attention
lands on static structure vs moving vehicles.

## 7. Timeline (today: 2 Sep 2026)

| Window | Deliverable | Who |
|---|---|---|
| **2–5 Sep** | Interface freeze meeting (§2). Repo created, scaffold, CODEOWNERS, CI, configs. | all / Vaibhav leads |
| **5–10 Sep** | vKITTI2 downloaded + subset; `vkitti.py` + `se3.py` + metrics all merged with tests green. Sanity: feed GT poses through eval → ATE ≈ 0. **← Milestone 2 (10 Sep)** | all |
| **10–20 Sep** | M1 and M4 trained end-to-end (fastest two). M2 and M3 in progress. Lit review: 2–3 papers each into `reports/literature/`. | all |
| **20–25 Sep** | Interim report Part B: lit table (8–10, from the 9 PDFs in `REFERENCES/` + 1 more), preprocessing description, ≥2 models with results, contribution log, risk plan. Tag `interim-2026-09-25`. **← Deadline 25 Sep** | all |
| **25 Sep–10 Oct** | M2 and M3 complete. All four on identical protocol. | all |
| **10–20 Oct** | Hyperparameter tuning, ablations, robustness run on held-out weather variations, `compare.py` final table, error analysis. **← Milestone 4 (20 Oct)** | all |
| **20–26 Oct** | Report Part C — each member writes their own Methodology subsection + defends it. Slides. Turnitin (<15% plag, <20% AI). Viva rehearsal: each member grilled by the other three. | all |
| **26–31 Oct** | Presentation + final submission with repo link. Tag `final-2026-10-31`. | all |

Note the buffer: models are due 20 Oct but the deadline is 31 Oct. Guard it —
self-supervised training (M3) is the item most likely to overrun.

## 8. Data + compute

- No CUDA on the dev machine; 26 GB free. **Do not download all of vKITTI 2** (~40 GB+
  across all variations for RGB + depth).
- `prepare_data.py` fetches only the needed scene/variation/camera subset:
  RGB for train+val+test variations listed in §2.3, and depth **only for Scene20/clone
  + val** (M3's evaluation) plus train depth only if a supervised-depth ablation is
  wanted. Estimated ~6–8 GB.
- Cache the resized 128x416 tensors once to `.npy`/LMDB — every training run after
  that is I/O-cheap, and the four members can share the same cache archive.
- Training on Colab (T4/L4) or Kaggle (P100, 30h/week). Each member has their own
  notebook in `notebooks/` that does `git clone` + `pip install -e .` + `train.py
  --model mX`, so the notebook is thin and the logic stays version-controlled.
- Checkpoints go to Drive or GitHub Releases, **never** into git. `.gitignore` them.
- Small artefacts (metrics.json, trajectory .txt, PNG plots) ARE committed — they are
  the evidence trail for the Interim 1.5 marks.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Dataset is small (~5k frames) → overfitting | Use 4 weather variations for train as augmentation; heavy aug (colour jitter, horizontal flip with pose sign-flip); early stopping on val ATE |
| Convention bugs (sign, order, world-vs-cam) | `se3.py` unit tests + GT-through-eval sanity check on day one |
| M3 photometric training fails to converge | Start it first; fall back to smaller resolution / fewer scales; pretrained encoder is non-negotiable |
| Synthetic-only results overstate performance | Already acknowledged in the synopsis; optionally run test-time inference on a handful of real KITTI frames for a qualitative domain-gap figure |
| One member falls behind → uneven commits | Weekly 15-min standup; CONTRIBUTIONS.md reviewed at each; reassign scope early, not in October |
| Borrowed code triggers plagiarism flags | THIRD_PARTY.md + per-file headers + report appendix |
