# Third-Party Code

Required by the project guidelines, Sec. 9: any use of external code or tutorials
must be cited in the report.

**This file is an appendix of the final report. Keep it current in the same PR as the
work — not retroactively.**

## Rules

1. Adapted code goes in `mvo/common/`, never inside a model folder, and carries a header:
   `# Adapted from <repo> @ <commit>, <license>. Modified: <what changed>.`
2. Add a row below in the same commit.
3. **Never vendor GPL code** (`evo`, `rl_vo`). Invoke them as command-line tools instead.
4. Model architectures are written by their owner. Each member is examined individually
   on their model in the viva, so borrowed architecture code is a direct mark risk.

## Candidate sources

| Repo | Paper (REFERENCES/) | Intended use | License | Status |
|---|---|---|---|---|
| `ClementPinard/SfmLearner-Pytorch` | [4] Zhou 2017 | `inverse_warp.py`, pose eval utils | MIT | not yet used |
| `nianticlabs/monodepth2` | [5] Godard 2019 | `layers.py` — SSIM, BackprojectDepth, Project3D | **Non-commercial research only** | not yet used |
| `ChiWeiHsiao/DeepVO-pytorch` | [3] Wang 2017 | structural reference for M2 only | MIT | not yet used |
| `jchao-xie.github.io/CoProU` | [2] Xie 2025 | uncertainty masking, M3 stretch goal | check | not yet used |
| `MichaelGrupp/evo` | — | independent cross-check of our ATE/RPE | GPLv3 | CLI only, not vendored |

## Files actually adapted

| Our file | Source file | Commit | License | What we changed |
|---|---|---|---|---|
| _(none yet)_ | | | | |
