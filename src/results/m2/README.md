# M2 result provenance

The files directly in this directory are the **2 October 2026, 60-epoch FP32
LSTM run** used for the interim report. They were copied from the locally held
`m2_fp32_60_artifacts.zip` after its ZIP integrity check passed. The original
bundle is kept locally but ignored by Git because it includes a checkpoint.

- Dataset: `vaibhavkrishnabansal/vkitti2-subset`, 128 x 416, Camera_0;
  training, validation and test splits follow `mvo/common/data/splits.py`.
- Training/evaluation code revision recorded in `metrics.json`:
  `b67d37a070d383f0a52186b2e4e02f572c98b58a`.
- Config recorded in the checkpoint: seed 42, `configs/m2.yaml` model settings,
  60 epochs, FP32 (`amp: false`), validation ATE checkpoint selection.
- `best.pt` was selected at epoch 27 (zero-based); validation ATE was
  11.5247862594 m. The saved checkpoint and its history are under
  `src/checkpoints/m2/` (ignored by Git).
- Test split: `Scene20/clone`. Test Sim(3) ATE is 61.6459925003 m and
  scale-fixed SE(3) ATE is 82.8354759348 m. See `metrics.json` for the
  complete metric set; `history.csv` has all 60 epochs.
- SHA-256 of the selected `best.pt`:
  `159adf081f9fcec596a27131818d6c88b141c8e23dd76d640fd22430a6ff192f`.

The older 25-epoch result files from September 2026 are preserved under
`previous_25_epoch/`. Their loose `metrics.json` was re-evaluated using full
recurrent context at revision `fbda660`, and is **not** the 60-epoch result.
The earlier checkpoint and histories are likewise preserved under
`src/checkpoints/m2/previous_25_epoch/` locally. Do not mix those files with
the result above when generating the final comparison table.
