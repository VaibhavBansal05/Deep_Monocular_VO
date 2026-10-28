# M4 CBAM result provenance

The files directly in this directory are from the **60-epoch M4 CBAM run**.
They were copied from the locally held `m4_cbam_60_artifacts.zip` after its
ZIP integrity check passed. The archive includes the selected checkpoint and
is ignored by Git.

- Dataset: `vaibhavkrishnabansal/vkitti2-subset`; the training, validation,
  test, and robustness sequences follow `mvo/common/data/splits.py` at the
  recorded code revision.
- Code revision: `b67d37a070d383f0a52186b2e4e02f572c98b58a`.
- Configuration: `configs/m4.yaml`, attention `cbam`, seed 42, batch size 16,
  60 epochs, full precision (`amp: false`), and validation ATE checkpoint
  selection. See `manifest_cbam_60.json` and the checkpoint in the local ZIP.
- Selected checkpoint: epoch 18 (zero-based), validation Sim(3) ATE
  7.3576067815 m. Its SHA-256 is
  `61bd9a7c09f7a7a7e4994398a4d299326e060f593279e1efcf0ddf357144d270`.
- Test: `Scene20/clone`; Sim(3) ATE **25.4706332041 m** and SE(3) ATE
  **75.2728900597 m**. See `metrics.json` and `pred_Scene20_clone.txt`.
- Robustness: held-out `Scene20` rain, fog, 15-degree-left, and
  30-degree-right variations. See `robustness/metrics.json` and its trajectory
  files. Compare per-sequence values against a matched no-attention run;
  the aggregate includes several different conditions.
- `history.csv` contains all 60 epochs; `summary.json` records the selected
  validation result.

The earlier loose metrics and trajectory, recorded at code revision
`6e76660`, are preserved in `previous_run_6e76660/`. The separate
`history_unwhitened_run.csv` is also historical; it has not been identified
as belonging to that earlier metrics file. Neither is the CBAM 60-epoch
result.

The verified `m4_none_60` run is kept under `no_attention/`, with its own
metrics, trajectory, history, effective config, manifest, and robustness
files. Its full archive is held locally as `m4_none_60_artifacts.zip` and is
ignored by Git. The archive was initially downloaded with a CBAM filename;
its internal directory, checkpoint, and manifest identify the no-attention
run. See `ATTENTION_COMPARISON.md` for the matched comparison.

Both full archives also have verified local copies under
`src/checkpoints/m4/artifact_backups/` (ignored by Git). Their SHA-256 hashes
are in `ARCHIVE_CHECKSUMS.sha256`. These copies share the same disk, so the
original Kaggle notebook outputs should also be retained for recovery.
