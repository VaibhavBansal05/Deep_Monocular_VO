# Results visualiser

Interactive console for the held-out test sequence: 3D camera path, synchronised
footage, drift trace, comparison table and training curves.

## Regenerate after new results

```bash
cd ~/DL_PROJECT/src
.venv/bin/python3 scripts/export_viz.py \
    --data-root data/vkitti2 --results results --out reports/viz
```

That rewrites `data.json` (trajectories, per-frame error, drift curves, training
histories, metrics, baselines) and `frames.jpg` (a sprite sheet of every third
frame, drawn tile by tile in the browser to play the sequence back).

Then republish `index.html` together with both assets.

## What it shows

- **Camera path** — ground truth and each model in 3D, with a wireframe frustum
  at the current frame showing real camera orientation, not just position.
  Predictions are sim(3)-aligned for display, exactly as ATE aligns them; without
  that every prediction would start at the origin while ground truth starts at
  its world pose.
- **Sequence** — the actual Camera_0 footage, scrubbed in step with the 3D view.
- **Position error** — metres against distance travelled. Drift that grows
  linearly indicates a constant per-frame bias rather than random error.
- **Training** — train and validation loss. The dashed line at 1.0 is the
  reference: in whitened units that is the loss of simply predicting the mean.

## Notes

- `frames.jpg` is ~0.9 MB at stride 3. Lowering `STRIDE` in `export_viz.py`
  gives smoother playback and a larger file; the artifact limit is 16 MB.
- The table highlights the best value per column among models only. Baselines
  are shown for reference and are excluded from that comparison.
