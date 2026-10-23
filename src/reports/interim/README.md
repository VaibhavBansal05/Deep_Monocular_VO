# Interim submission (Part B) — updated 3 Oct 2026

| File | Purpose |
|---|---|
| `ICT4442_Interim_Report_PartB.docx` | the submission (editable) |
| `ICT4442_Interim_Report_PartB.pdf` | rendered submission copy |
| `build_report.py` | regenerates both; edit this rather than the .docx so changes are versioned |

Regenerate after editing:

```bash
cd ~/DL_PROJECT
src/.venv/bin/python3 src/reports/interim/build_report.py
cd src/reports/interim && soffice --headless --convert-to pdf ICT4442_Interim_Report_PartB.docx --outdir .
```

## Results used in this version

- `src/results/m1/metrics.json` from merged commit `22bb6dd`
- `src/results/m2/m2_fp32_60_artifacts.zip`
- `src/results/m4/m4_cbam_60_artifacts.zip`

All three models were evaluated on `Scene20/clone`. M2 and M4 are 60-epoch FP32 runs,
and the M4 archive also includes rain, fog and camera-rotation robustness results. Older loose files in
`src/results/m2/` and `src/results/m4/` are from superseded experiments and are not cited.

## Still to do by hand

- **Verify the literature figures.** Every number came from the PDFs in `REFERENCES/`, but
  PDF table extraction is unreliable. Each owner should re-check the papers listed against
  their name in `../literature/README.md`.
- **Correct the synopsis citation.** Reference [7] is *Sensors* 2025, 25, 5448 — not
  "PMC Open Access, 2024".

## What the submission needs alongside this

The guidelines ask for the report via MS Forms. A GitHub repo link is not required at
Interim, but the contribution log is cross-checked against commit history, so tag the
state at submission:

```bash
git tag interim-2026-09-25 && git push --tags
```
