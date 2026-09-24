"""Build the ICT-4442 Part B interim report as a .docx."""
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

TODO = RGBColor(0xC0, 0x00, 0x00)

doc = Document()
for s in doc.sections:
    s.page_width = Cm(21.0)      # A4, not python-docx's US Letter default
    s.page_height = Cm(29.7)
    s.top_margin = s.bottom_margin = Cm(1.9)
    s.left_margin = s.right_margin = Cm(1.8)

st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(10.5)
st.paragraph_format.space_after = Pt(4)


def para(text="", size=10.5, bold=False, italic=False, align=None, after=4, color=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    if align is not None:
        p.alignment = align
    if text:
        r = p.add_run(text)
        r.bold, r.italic = bold, italic
        r.font.size = Pt(size)
        if color is not None:
            r.font.color.rgb = color
    return p


def field(label, value, todo=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(label + "  ")
    r.bold = True
    r.font.size = Pt(10.5)
    r2 = p.add_run(value)
    r2.font.size = Pt(10.5)
    if todo:
        r2.font.color.rgb = TODO
        r2.bold = True
    return p


def heading(text, size=12):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(size)
    return p


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:color"), "auto")
    el.set(qn("w:fill"), hexcolor)
    tcPr.append(el)


def table(headers, rows, widths, font=8.5):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    # LibreOffice and Word honour the tblGrid, not per-cell widths alone.
    tblPr = t._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)
    for i, w in enumerate(widths):
        t.columns[i].width = Cm(w)
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.width = Cm(widths[i])
        c.text = ""
        p = c.paragraphs[0]
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(font)
        shade(c, "D9E2F3")
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].width = Cm(widths[i])
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            import re as _re
            for seg in _re.split(r"(<<[^>]*>>)", val):
                if not seg:
                    continue
                r = p.add_run(seg)
                r.font.size = Pt(font)
                if seg.startswith("<<"):
                    r.bold = True
                    r.font.color.rgb = TODO
    return t


# ───────────────────────────── cover page ─────────────────────────────
for _ in range(4):
    para(after=0)
para("Deep Learning-Based Monocular Visual Odometry for Autonomous Spatial Mapping",
     size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=16)
para("Interim report on", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
para("Deep Learning Project", size=12, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
para("[ICT-4442]", size=12, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=18)
para("Submitted By", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, after=10)
for n, r in [("Vaibhav Krishna Bansal", "230911546"), ("Pranav Joshi", "230953248"),
             ("Abhishek Laxman Jadhav", "230953360"), ("Suryansh Bajaj", "230953250")]:
    para(f"{n}\t\t{r}", size=11, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
para(after=30)
para("SCHOOL OF COMPUTER ENGINEERING", size=12, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
para("MANIPAL INSTITUTE OF TECHNOLOGY,", size=12, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
para("MANIPAL ACADEMY OF HIGHER EDUCATION", size=12, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, after=16)
para("September 2026", size=11, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_page_break()

# ───────────────────────────── header block ─────────────────────────────
heading("PART B : INTERIM REPORT", 13)
field("Team No. and Names (with Registration Numbers):",
      "Vaibhav Krishna Bansal, 230911546; Pranav Joshi, 230953248; "
      "Abhishek Laxman Jadhav, 230953360; Suryansh Bajaj, 230953250")
field("Title of the Project (confirmed from Synopsis):",
      "Deep Learning-Based Monocular Visual Odometry for Autonomous Spatial Mapping")
field("GitHub Repository Link:", "https://github.com/VaibhavBansal05/Deep_Monocular_VO")

# ───────────────────────────── 1. literature ─────────────────────────────
heading("1. Literature Review")
para("Nine papers, summarised and compared. Full per-paper summaries are maintained in "
     "the repository at src/reports/literature/.", size=9.5, italic=True)

lit = [
    ("Cabon et al., 2020 — Virtual KITTI 2",
     "Unity-rendered photo-realistic clones of 5 KITTI tracking sequences; 10 variants each "
     "(weather, camera rotation); RGB, depth, flow, segmentation and 6-DoF poses",
     "vKITTI 2 (synthetic) vs real KITTI",
     "Re-ran depth, pose and segmentation baselines on both; synthetic data is a usable "
     "proxy, but a measurable domain gap remains",
     "Our dataset. 100% dense depth (vs ~5% LiDAR coverage) and exact poses"),
    ("Wang et al., 2017 — DeepVO (ICRA)",
     "CNN (FlowNet-pretrained) to 2-layer LSTM to 6-DoF pose; fully end-to-end, no classical "
     "pipeline module retained",
     "KITTI odometry",
     "Competitive with classical methods; learns absolute scale implicitly, which monocular "
     "geometry cannot recover unaided",
     "Model 2's blueprint, including the rotation loss weighting"),
    ("Zhou et al., 2017 — SfMLearner (CVPR)",
     "DepthNet + PoseNet trained only by photometric view-synthesis loss; learned "
     "explainability mask for non-rigid regions",
     "KITTI odometry / Eigen split",
     "ATE 0.021+/-0.017 (Seq 09) and 0.020+/-0.015 (Seq 10) over 5-frame snippets; beats "
     "ORB-SLAM (short) 0.064, behind ORB-SLAM (full) 0.014/0.012",
     "Model 3's blueprint; pose is learnable with no labels at all"),
    ("Godard et al., 2019 — Monodepth2 (ICCV)",
     "Three targeted fixes: minimum-over-reference reprojection loss, full-resolution "
     "multi-scale sampling, and auto-masking of static pixels",
     "KITTI (Eigen split)",
     "Abs Rel 0.115 monocular at 640x192 (0.132 without ImageNet pretraining); ablations "
     "isolate each component",
     "Model 3's loss design, item for item; also a template for our ablations"),
    ("Zhao et al., 2019 — LKN-VO (Robot. Auton. Syst.)",
     "Learning Kalman Network: deep observation and transition models fused by the Kalman "
     "update instead of an RNN hidden state",
     "KITTI",
     "Learns absolute scale without geometric constraints; filter parameters learned rather "
     "than hand-designed",
     "The principled alternative to Model 2's recurrence"),
    ("Zhi et al., 2022 — Multi-Scale Unsupervised MVO (Sensors)",
     "Densely-linked atrous convolutions to widen the receptive field without downsampling, "
     "plus non-local self-attention for long-range dependencies",
     "KITTI",
     "Competitive with unsupervised SOTA; state-of-the-art rotation estimation, with gains "
     "concentrated in rotating scenes",
     "Closest published analogue to Model 4; source of its central hypothesis"),
    ("Albukhari et al., 2025 — MINI-DROID-SLAM (Sensors)",
     "Replaces DROID-SLAM's ConvGRU update operator with a Mini-GRU; retains dense depth and "
     "bundle adjustment",
     "TartanAir",
     "ATE comparable to DROID-SLAM at reduced training time and computational cost, "
     "trainable on a single GPU",
     "Evidence for Model 2's GRU-vs-LSTM ablation"),
    ("Xie et al., 2025 — CoProU-VO (GCPR)",
     "Projects reference-frame uncertainty into the target frame and combines it with the "
     "target's own under one probabilistic mask; ViT backbone",
     "KITTI, nuScenes",
     "Outperforms prior unsupervised two-frame end-to-end methods, with the largest margin "
     "on highway scenes where others fail",
     "Most recent work in our family; natural extension to Model 3"),
    ("Messikommer et al., 2024 — RL Meets VO (ECCV)",
     "An RL agent inside a classical VO pipeline selecting keyframes and grid size online; "
     "reward from pose error and runtime",
     "Public VO benchmarks",
     "Improves accuracy and robustness while removing weeks of expert heuristic tuning",
     "Shows learning can augment rather than replace classical VO; framing for Model 1"),
]
table(["Paper (Author, Year)", "Method", "Dataset", "Key Result", "Relevance to Project"],
      lit, [3.3, 4.4, 2.1, 4.1, 3.4], font=7.5)

para(after=6)
para("Synthesis. ", bold=True, after=2)
for txt in [
    "The field has moved from replacing the geometric pipeline to repairing it. DeepVO and "
    "SfMLearner argued the whole pipeline could be learned end-to-end; the strongest later "
    "work is narrower, with Monodepth2 winning on three specific loss fixes and RL-VO "
    "leaving a classical pipeline intact while learning only its heuristics. This is why our "
    "four models deliberately span architecture families.",
    "Supervision level and scale handling are entangled. Supervised models learn absolute "
    "scale from pose labels; unsupervised models recover structure only up to an unknown "
    "global scale. A comparison that ignores this flatters the supervised side, so our "
    "protocol reports ATE both with scale solved (sim3) and with scale fixed (SE3).",
    "Attention is the least-settled axis and the gap this project addresses. Only two of the "
    "nine papers use it, and neither isolates it: Zhi et al. introduce atrous convolutions "
    "and non-local attention together, and CoProU-VO changes the backbone to a ViT. Model 4's "
    "ablation holds backbone, data, schedule and seed fixed and varies only the attention "
    "module, measuring something the literature leaves confounded.",
    "None of the nine uses Virtual KITTI 2 as its primary benchmark. Our results are "
    "therefore not directly comparable to published KITTI numbers and are reported as an "
    "optimistic upper bound rather than as parity.",
]:
    p = para(txt, size=9.5, after=4)
    p.paragraph_format.first_line_indent = Cm(0.4)

doc.add_page_break()

# ───────────────────────────── 2. dataset ─────────────────────────────
heading("2. Dataset Acquisition & Preprocessing")
para("Acquisition. ", bold=True, after=2)
para("Virtual KITTI 2 is distributed as monolithic archives (~7.2 GB RGB, ~8 GB depth). "
     "scripts/prepare_data.py streams the archives and extracts only the scenes, variations "
     "and camera required by the split, yielding a 1.4 GB working subset of 8,642 frames "
     "across 18 sequences. Frame counts were verified against the declared scene lengths "
     "(Scene01 447, Scene02 233, Scene06 270, Scene18 339, Scene20 837).",
     size=9.5, after=6)

para("Preprocessing pipeline. ", bold=True, after=2)
for txt in [
    "Resize. RGB frames are bilinearly resized from 1242x375 to 128x416.",
    "Intrinsics rescaling. K is scaled by the same factors as the image (fx, cx by width; "
    "fy, cy by height). This is required for the photometric reprojection in Model 3; an "
    "unscaled K produces a systematic reprojection error the loss cannot correct.",
    "Normalization. ImageNet mean/std, matching the pretrained ResNet-18 backbones.",
    "Temporal stacking. Two frames for the pairwise models (Models 1 and 4) and five frames "
    "for the sequence models (Models 2 and 3).",
    "Pose conversion. extrinsic.txt supplies world-to-camera matrices E. Relative pose is "
    "computed as T(t to t+1) = E(t+1) * inverse(E(t)), and absolute camera-to-world as "
    "inverse(E). Camera 0 rows are filtered explicitly, since both cameras are interleaved "
    "in the file.",
    "Augmentation. Colour jitter applied identically across all frames of a sequence, plus "
    "horizontal flip with the corresponding pose mirroring.",
]:
    p = doc.add_paragraph(txt, style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    for r in p.runs:
        r.font.size = Pt(9.5)

para("Split. ", bold=True, after=2)
para("The split is by scene, so no geometric content is shared between train and test. "
     "Train: Scene01, Scene06, Scene18 under clone, morning, overcast and sunset (4,212 "
     "samples). Validation: Scene02/clone (232). Test: Scene20/clone (836). The rain, fog, "
     "15-degree and 30-degree variations are held out entirely for a robustness ablation at "
     "the Final stage. The split is defined once in mvo/common/data/splits.py and is "
     "identical for all four models; unit tests assert that no scene appears in two splits.",
     size=9.5, after=6)

para("Verification. ", bold=True, after=2)
para("Integrating the ground-truth relative poses through the full pipeline reproduces the "
     "ground-truth trajectory as an exact rigid transform (residual 7e-14 m when the true "
     "initial pose is applied), confirming the pose conventions are consistent end to end. "
     "The extrinsic files store roughly seven significant digits, so their rotation matrices "
     "are orthonormal only to ~5e-7; chaining 837 of them accumulates to ~2e-3 m. This is "
     "the precision floor of the dataset and no model can be meaningfully scored below it.",
     size=9.5, after=6)

# ───────────────────────────── 3. models ─────────────────────────────
heading("3. Models Implemented So Far")
models = [
    ("Model 4 — CNN + CBAM/SE attention", "Vaibhav Krishna Bansal",
     "Baseline trained (attention = none); CBAM/SE in progress",
     "Best val loss 0.02543 at epoch 12 of 21; 12.50 M parameters. "
     "ATE / RPE: <<pending evaluation>>",
     "ResNet-18 with conv1 inflated to 6 channels for the stacked pair. Validation loss "
     "minimised at epoch 12 and rose thereafter while training loss continued to fall."),
    ("Model 2 — CNN + GRU/LSTM", "Abhishek Laxman Jadhav",
     "Implemented; training in progress",
     "<<pending — training running>>", "15.39 M parameters, sequence length 5, 2-layer LSTM "
     "over a shared ResNet-18 encoder. Supports rnn = lstm | gru | none for ablation."),
    ("Model 1 — Classical ORB/LK + MLP", "Pranav Joshi",
     "Not started; dataset and preprocessing pipeline delivered",
     "n/a", "Feature extraction is deterministic and will be precomputed, so training is "
     "CPU-feasible."),
    ("Model 3 — Self-supervised PoseNet/DepthNet", "Suryansh Bajaj",
     "Not started; metrics and evaluation harness delivered",
     "n/a", "Longest pole; depends on the projection utilities, which are complete and "
     "tested."),
]
table(["Model", "Owner (Member)", "Status", "Preliminary Metric", "Notes"],
      models, [3.6, 2.9, 3.3, 3.9, 3.6], font=8)

para(after=4)
para("Shared infrastructure. ", bold=True, after=2)
para("All four models subclass a common BaseVOModel contract and are trained by a single "
     "engine and scored by a single evaluation script, so the comparison protocol cannot "
     "drift between members. The repository currently holds 60 automated tests covering the "
     "SE(3) utilities, projection, dataset contract, metrics and model interfaces, run in "
     "continuous integration on every pull request.",
     size=9.5, after=4)
para("Rotation loss weighting. ", bold=True, after=2)
para("The rotation weight k in the pose-regression loss was measured rather than taken from "
     "the literature. On the training split E[t^2] = 2.00e-1 against E[r^2] = 3.41e-5, a "
     "ratio of approximately 5869:1. DeepVO's published k = 100 leaves the rotation term "
     "about 59 times under-weighted at this scale, allowing the model to ignore rotation and "
     "still reduce the loss. Models 2 and 4 therefore use k = 5900, reproducible via "
     "scripts/measure_pose_stats.py.",
     size=9.5, after=6)

# ───────────────────────────── 4. contributions ─────────────────────────────
heading("4. Individual Contribution Log (to date)")
para("Cross-checked against GitHub commit history (git shortlog -sne) on the main branch.",
     size=9.5, italic=True, after=4)
contrib = [
    ("Vaibhav Krishna Bansal", "230911546",
     "Repository scaffold and shared contracts (pose convention, batch schema, split, "
     "metrics schema); training engine; model registry; CI; Model 4 backbone and pose "
     "decoder; measured rotation-weight correction for Models 2 and 4; dataset acquisition "
     "and distribution; Colab and Kaggle training notebooks; literature review "
     "(3 papers). 33 commits.", ""),
    ("Pranav Joshi", "230953248",
     "vKITTI 2 subset downloader with streaming extraction; image transforms and intrinsics "
     "rescaling; VKitti2 dataset implementing the shared batch contract; depth loading; "
     "dataset unit tests. 5 commits.", ""),
    ("Abhishek Laxman Jadhav", "230953360",
     "SE(3) and SO(3) utilities (Rodrigues exponential and logarithm with small-angle "
     "stability, analytic inverse, relative/absolute trajectory chaining); perspective "
     "projection and inverse warping; Model 2 CNN+RNN architecture; geometry and projection "
     "test suites. 7 commits.", ""),
    ("Suryansh Bajaj", "230953250",
     "Umeyama similarity alignment with reflection guard; ATE with separate sim(3) and SE(3) "
     "reporting; RPE and KITTI-style relative drift; monocular depth metrics with median "
     "scaling; TUM trajectory I/O for cross-validation against evo; comparison-table "
     "generator; metrics test suite. 9 commits.", ""),
]
table(["Member Name", "Reg. No.", "Task(s) Completed", "Signature"],
      contrib, [3.4, 1.9, 9.0, 3.0], font=8)

# ───────────────────────────── 5. risk ─────────────────────────────
heading("5. Risk / Plan for Remaining Work")
para("Remaining models. ", bold=True, after=2)
para("Model 1 (classical + MLP) and Model 3 (self-supervised PoseNet/DepthNet) are not yet "
     "implemented. The shared infrastructure both depend on is complete and tested: Model 1 "
     "requires only the feature extractor and regressor, and Model 3's projection and "
     "warping utilities are already merged and unit-tested.",
     size=9.5, after=6)

risks = [
    ("Model 3 fails to converge", "High impact, moderate likelihood. Photometric "
     "self-supervision is the most fragile objective of the four. Mitigation: start "
     "immediately, keep the ImageNet-pretrained encoder (non-negotiable at our dataset "
     "size), and fall back to fewer scales or lower resolution. Monodepth2's reported "
     "0.132-vs-0.115 gap without pretraining bounds the expected cost."),
    ("Overfitting on a small training set", "Confirmed, not hypothetical. Model 4's "
     "validation loss minimised at epoch 12 and rose thereafter. The cause is identified: "
     "the augmentation RNG is seeded per sample index, so each sample receives identical "
     "jitter every epoch and augmentation is effectively inactive. A fix is scheduled; "
     "both trained models will be re-run afterwards."),
    ("GPU availability", "Realised. The free Colab allowance was exhausted mid-run. "
     "Mitigation already in place: training migrated to Kaggle (approximately 30 h/week, "
     "headless execution up to 12 h), with local CPU training as a fallback at roughly "
     "9 minutes per epoch for Model 4."),
    ("Unequal training budgets", "Models trained for different numbers of epochs are not "
     "strictly comparable. For the Final submission all four will be re-run under one "
     "budget on the same hardware; the Interim figures are explicitly preliminary."),
    ("Uneven member availability", "Two of four members were unavailable during the week "
     "before this submission. Tracked weekly via git shortlog so that workload can be "
     "reassigned early rather than at the deadline."),
]
table(["Risk", "Assessment and mitigation"], risks, [4.6, 12.7], font=8.5)

para(after=6)
para("Timeline to Final submission.", bold=True, after=3)
plan = [
    ("25 Sep - 30 Sep", "Fix the augmentation defect; implement Model 1 (Pranav) and begin "
     "Model 3 (Suryansh); complete Model 4's CBAM/SE variants (Vaibhav)."),
    ("1 Oct - 12 Oct", "Model 3 trained and evaluated. All four models re-run under an "
     "identical epoch budget on identical hardware."),
    ("13 Oct - 20 Oct", "Hyperparameter tuning; ablations (attention on/off, LSTM/GRU/none); "
     "robustness evaluation on the held-out rain, fog and rotated variations; unified "
     "comparison table and error analysis."),
    ("21 Oct - 26 Oct", "Final report Part C, one methodology subsection per model owner; "
     "slides; plagiarism check; viva rehearsal."),
    ("26 Oct - 31 Oct", "Presentation and final submission with repository link."),
]
table(["Window", "Planned work"], plan, [3.4, 13.9], font=8.5)

out = "src/reports/interim/ICT4442_Interim_Report_PartB.docx"
doc.save(out)
print("wrote", out)
