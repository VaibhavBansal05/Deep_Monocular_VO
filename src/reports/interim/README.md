# Interim submission (Part B) — due Fri 25 Sep 2026, 11:59 PM

| File | Purpose |
|---|---|
| `ICT4442_Interim_Report_PartB.docx` | the submission (editable) |
| `ICT4442_Interim_Report_PartB.pdf` | rendered, 4 pages incl. cover — within the 3–5 page limit |
| `build_report.py` | regenerates both; edit this rather than the .docx so changes are versioned |

Regenerate after editing:

```bash
cd ~/DL_PROJECT
src/.venv/bin/python3 src/reports/interim/build_report.py
cd src/reports/interim && soffice --headless --convert-to pdf ICT4442_Interim_Report_PartB.docx --outdir .
```

## Still to fill — shown in red in the document

1. **Model 4 ATE / RPE.** `best.pt` (epoch 12) is in Drive but has not been evaluated.
   ```bash
   cd ~/DL_PROJECT/src
   .venv/bin/python3 scripts/eval.py --model m4 --ckpt checkpoints/m4/best.pt --data-root data/vkitti2
   ```
2. **Model 2 results.** Training on Kaggle. Same command with `--model m2`.

Then put the numbers into `build_report.py` (the `models` list) and regenerate.

## Still to do by hand

- **Signatures.** The contribution log has an empty Signature column, and the guidelines
  require the log to be signed by all members. Print, sign, and scan — or have each member
  insert a signature image.
- **Verify the literature figures.** Every number came from the PDFs in `REFERENCES/`, but
  PDF table extraction is unreliable. Each owner should re-check the papers listed against
  their name in `../literature/README.md`.
- **Correct the synopsis citation.** Reference [7] is *Sensors* 2025, 25, 5448 — not
  "PMC Open Access, 2024".
- **Confirm the team number**, which the template asks for and the synopsis does not record.

## What the submission needs alongside this

The guidelines ask for the report via MS Forms. A GitHub repo link is not required at
Interim, but the contribution log is cross-checked against commit history, so tag the
state at submission:

```bash
git tag interim-2026-09-25 && git push --tags
```
