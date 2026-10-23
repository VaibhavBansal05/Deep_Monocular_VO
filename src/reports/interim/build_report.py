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
    p.paragraph_format.keep_with_next = True
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
    cant_split = OxmlElement("w:cantSplit")
    t.rows[0]._tr.get_or_add_trPr().append(cant_split)
    repeat_header = OxmlElement("w:tblHeader")
    repeat_header.set(qn("w:val"), "true")
    t.rows[0]._tr.get_or_add_trPr().append(repeat_header)
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
        cant_split = OxmlElement("w:cantSplit")
        t.rows[-1]._tr.get_or_add_trPr().append(cant_split)
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
para("October 2026", size=11, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_page_break()
# ───────────────────────────── header block ─────────────────────────────
heading("PART B : INTERIM REPORT", 13)
field("Team Members (with Registration Numbers):",
      "Vaibhav Krishna Bansal, 230911546; Pranav Joshi, 230953248; "
      "Abhishek Laxman Jadhav, 230953360; Suryansh Bajaj, 230953250")
field("Title of the Project:",
      "Deep Learning-Based Monocular Visual Odometry for Autonomous Spatial Mapping")
field("GitHub Repository Link:", "https://github.com/VaibhavBansal05/Deep_Monocular_VO")

# ───────────────────────────── 1. literature ─────────────────────────────
heading("1. Literature Review")
para("The nine reviewed papers cover the main ideas used in our four models: classical visual "
     "odometry, supervised learning, self-supervised learning, recurrent networks and "
     "attention.", size=9.2, after=3)
para("Virtual KITTI 2 recreates scenes from the Karlsruhe Institute of Technology and Toyota "
     "Technological Institute benchmark (KITTI). It provides red-green-blue (RGB) images, "
     "depth and exact six-degree-of-freedom (6-DoF) camera poses.", size=8.5, after=3)
para("The table uses the following common terms: convolutional neural network (CNN), long "
     "short-term memory (LSTM), gated recurrent unit (GRU), convolutional GRU (ConvGRU), "
     "vision transformer (ViT), reinforcement learning (RL), graphics processing unit (GPU) "
     "and light detection and ranging (LiDAR).", size=8.5, after=4)

lit = [
    ("Cabon et al., 2020 — Virtual KITTI 2",
     "Synthetic copies of five KITTI sequences with weather and camera changes; includes "
     "RGB, depth, motion, segmentation and camera poses",
     "Virtual KITTI 2 (synthetic) versus real KITTI",
     "Synthetic data is useful for training and testing, but it still differs from real data",
     "Provides the synthetic images, camera data and exact poses used in this project"),
    ("Wang et al., 2017 — DeepVO",
     "A CNN extracts image features and a two-layer LSTM predicts camera motion",
     "KITTI odometry",
     "Reached performance close to classical methods and learned metric scale from data",
     "Provides the main architectural basis for Model 2"),
    ("Zhou et al., 2017 — Structure-from-Motion Learner",
     "DepthNet and PoseNet learn from the difference between a real image and a reconstructed image",
     "KITTI odometry / Eigen split",
     "Outperformed a short-sequence classical baseline but not its full-sequence version",
     "Provides the starting point for Model 3's self-supervised training"),
    ("Godard et al., 2019 — Monodepth2",
     "Improved the image-reconstruction loss, multi-scale training and handling of static pixels",
     "KITTI (Eigen split)",
     "Reported 0.115 absolute-relative depth error at 640x192 resolution",
     "Guides Model 3's photometric loss and planned ablation studies"),
    ("Zhao et al., 2019 — Learning Kalman Network for Visual Odometry",
     "Learns observation and motion models, then combines them with a Kalman update",
     "KITTI",
     "Learns metric scale and filter parameters from data",
     "Provides an alternative to the recurrent design used in Model 2"),
    ("Zhi et al., 2022 — Multi-Scale Unsupervised Monocular Visual Odometry",
     "Uses multi-scale convolutions and self-attention to capture local and long-range motion",
     "KITTI",
     "Improved rotation estimates, especially in scenes with large turns",
     "Supports the attention-based design tested in Model 4"),
    ("Albukhari et al., 2025 — MINI-DROID-SLAM (Sensors)",
     "Replaces DROID-SLAM's ConvGRU with a smaller GRU while keeping depth and optimisation",
     "TartanAir",
     "Similar accuracy with lower training time and cost",
     "Evidence for Model 2's recurrent-unit ablation"),
    ("Xie et al., 2025 — CoProU-VO",
     "Uses a ViT and an uncertainty mask to reduce the effect of unreliable image regions",
     "KITTI, nuScenes",
     "Outperformed earlier unsupervised two-frame methods, especially on highways",
     "Suggests uncertainty modelling as a possible extension to Model 3"),
    ("Messikommer et al., 2024 — Reinforcement Learning Meets Visual Odometry",
     "An RL agent chooses keyframes and grid size inside a classical visual-odometry system",
     "Public VO benchmarks",
     "Improved accuracy and reduced manual tuning",
     "Supports Model 1's combination of classical geometry and learned regression"),
]
table(["Paper (Author, Year)", "Method", "Dataset", "Key Result", "Relevance to Project"],
      lit, [3.3, 4.4, 2.1, 4.1, 3.4], font=7.1)

synthesis_heading = para("Synthesis. ", bold=True, after=2)
synthesis_heading.paragraph_format.space_before = Pt(6)
for txt in [
    "Early work tried to learn the complete visual-odometry pipeline. Later studies often kept "
    "the useful geometric parts and learned only selected components. Our four models cover "
    "both approaches.",
    "Supervised models learn metric scale from pose labels. Self-supervised monocular models "
    "usually recover motion only up to an unknown scale. For this reason, we report trajectory error "
    "with and without correcting one global scale factor.",
    "The reviewed attention methods also change other parts of the network, so they do not "
    "show the effect of attention alone. Our Model 4 tests will keep everything else fixed and "
    "change only the attention module. Published KITTI scores are not directly comparable with "
    "our Virtual KITTI 2 results because the datasets and test protocols differ.",
]:
    p = para(txt, size=8.9, after=2)
    p.paragraph_format.first_line_indent = Cm(0.4)

# ───────────────────────────── 2. dataset ─────────────────────────────
dataset_heading = heading("2. Dataset and Preprocessing")
dataset_heading.paragraph_format.page_break_before = True
para("Acquisition. ", bold=True, after=2)
para("Virtual KITTI 2 provides separate archives for RGB images, depth and "
     "text ground truth. The current working subset contains RGB, intrinsics and poses; "
     "dense depth has not yet been added. Our subset contains 8,642 RGB frames from 18 "
     "sequences. We checked the extracted frame counts against the dataset documentation.",
     size=9.5, after=6)

para("Preprocessing pipeline. ", bold=True, after=2)
for txt in [
    "Resize. Images are resized from 1242x375 to 128x416 pixels.",
    "Camera parameters. Focal lengths and principal-point coordinates are scaled to match "
    "the resized image.",
    "Normalization. Images use the same ImageNet normalization as the pretrained 18-layer "
    "residual network (ResNet-18).",
    "Frame groups. Models 1 and 4 use two frames. Models 2 and 3 use five frames.",
    "Pose conversion. World-to-camera poses are converted into motion between consecutive "
    "frames. Camera 0 is selected because both cameras are stored in the same file.",
    "Augmentation. All frames in a group receive the same colour change and horizontal flip. "
    "The seed, epoch and sample number control the random choices, making reruns reproducible.",
]:
    p = doc.add_paragraph(txt, style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    for r in p.runs:
        r.font.size = Pt(9.5)

para("Split. ", bold=True, after=2)
para("The split is by scene, so no geometric content is shared between train and test. "
     "Train: Scene01, Scene06, Scene18 under clone, morning, overcast and sunset (4,212 "
     "samples). Validation: Scene02/clone (232). Test: Scene20/clone (836). The rain, fog, "
     "15-degree and 30-degree variations are held out entirely for robustness evaluation. "
     "All four models use this split. Automated checks confirm that no scene appears in more "
     "than one partition.",
     size=9.5, after=6)

para("Verification. ", bold=True, after=2)
para("We rebuilt the ground-truth trajectory from the relative poses. It matched the original "
     "trajectory to numerical precision, which confirms that the pose direction and camera "
     "coordinates are handled correctly.",
     size=9.5, after=6)

# ───────────────────────────── 3. models ─────────────────────────────
heading("3. Models and Interim Results")
para("Protocol. ", bold=True, after=2)
para("Models 1, 2 and 4 use the same data split and random seed (42). Model 1 (M1) combines "
     "classical feature matching with a multilayer perceptron (MLP). Model 2 (M2) uses "
     "ResNet-18 followed by a two-layer long short-term memory network. Model 4 (M4) uses "
     "ResNet-18 with a Convolutional Block Attention Module (CBAM). M2 and M4 were trained "
     "for 60 epochs using 32-bit floating-point (FP32) arithmetic. The best validation "
     "checkpoint from each model was tested once on Scene20/clone.",
     size=9.5, after=3)
para("Absolute trajectory error (ATE) measures the difference between the full predicted and "
     "true paths. Similarity transformation alignment, Sim(3), corrects one overall scale "
     "factor. Rigid-body transformation alignment, SE(3), keeps the predicted scale. Relative "
     "pose error (RPE) measures frame-to-frame motion. "
     "The t_rel and r_rel columns show translation and rotation drift. Lower values are better.",
     size=9.5, after=5)

results = [
    ("M1 - classical grid descriptor + MLP", "Pranav Joshi", "24.209", "68.320",
     "0.388", "0.218", "36.61", "8.40", "0.27", "9.73"),
    ("M2 — CNN + 2-layer LSTM", "Abhishek Laxman Jadhav", "61.646", "82.835",
     "0.347", "0.251", "54.52", "19.76", "15.39", "35.80"),
    ("M4 — pairwise CNN + CBAM", "Vaibhav Krishna Bansal", "25.471", "75.273",
     "0.364", "0.196", "35.67", "7.89", "11.23", "17.08"),
]
table(["Model", "Owner", "ATE Sim(3)\n(m)", "ATE SE(3)\n(m)",
       "RPE trans\n(m)", "RPE rot\n(deg)", "t_rel\n(%)", "r_rel\n(deg/100m)",
       "Model size\n(M)", "Latency\n(ms/frame)"],
      results, [3.1, 2.7, 1.25, 1.25, 1.25, 1.2, 1.05, 1.35, 1.4, 1.45], font=7.1)

result_heading = para("Result interpretation. ", bold=True, after=2)
result_heading.paragraph_format.space_before = Pt(6)
result_heading.paragraph_format.keep_with_next = True
para("M1 gives the lowest full-trajectory error. It is also the smallest and fastest model. "
     "M2 gives the lowest frame-to-frame translation error, while M4 gives the lowest "
     "rotation error and drift. M1 overestimates forward movement by an average of 0.266 "
     "metres per frame. This small repeated error builds up over the complete route.",
     size=9.5, after=5)

para("M4 held-out robustness. ", bold=True, after=2)
robust = [
    ("Scene20/rain", "44.275", "47.441", "0.291", "0.301", "33.53", "12.21"),
    ("Scene20/fog", "29.919", "31.878", "0.319", "0.286", "24.97", "8.99"),
    ("Scene20/15-deg-left", "55.792", "105.589", "0.447", "0.244", "43.10", "13.08"),
    ("Scene20/30-deg-right", "189.036", "190.757", "0.523", "0.446", "62.61", "41.60"),
]
table(["Sequence", "ATE Sim(3) (m)", "ATE SE(3) (m)", "RPE trans (m)",
       "RPE rot (deg)", "t_rel (%)", "r_rel (deg/100m)"],
      robust, [3.7, 2.2, 2.2, 2.25, 2.2, 2.0, 2.75], font=7.5)
robust_text = para("Rain and fog increase the error, but camera rotation causes a much larger drop in "
     "performance. The 30-degree-right sequence reaches 189.0 m Sim(3) ATE. We still need "
     "to compare CBAM with no attention and Squeeze-and-Excitation (SE) attention before "
     "deciding how much the attention module helps.",
     size=9.5, after=5)
robust_text.paragraph_format.space_before = Pt(4)

para("Implementation status. ", bold=True, after=2)
para("M1 is complete. It detects up to 1,000 Oriented FAST and Rotated BRIEF (ORB) features. "
     "FAST means Features from Accelerated Segment Test, and BRIEF means Binary Robust "
     "Independent Elementary Features. An 8 by 8 grid summarises horizontal motion, vertical "
     "motion and reliable matches, and a Multilayer Perceptron (MLP) predicts the camera movement. "
     "M2 is complete and evaluated. It uses ResNet-18 with a two-layer Long Short-Term Memory "
     "(LSTM) network to learn motion from five-frame sequences. M4 is also complete and evaluated. "
     "It uses a pairwise ResNet-18 model with Convolutional Block Attention Module (CBAM) attention. "
     "M3, the self-supervised PoseNet and DepthNet model, is still under development and needs the "
     "depth-image archive for evaluation.", size=9.5, after=5)

shared_heading = para("Shared infrastructure. ", bold=True, after=2)
shared_heading.paragraph_format.space_before = Pt(5)
para("All four models use the same data loader, training structure and evaluation code. This "
     "keeps the data split, pose format and metrics consistent. Fixed seeds make the runs "
     "repeatable, and automated tests check the data, geometry, models and metrics.",
     size=9.5, after=4)
para("Geometric pose objective. ", bold=True, after=2)
para("M1, M2 and M4 calculate translation and rotation errors separately. Translation uses "
     "straight-line distance in metres. Rotation uses the smallest angle between the predicted "
     "and true orientations, called the geodesic angle on the three-dimensional special "
     "orthogonal group (SO(3)). Each part is scaled so that one does not dominate the loss. A "
     "small extra term penalises a repeated positive or negative error in any direction.",
     size=9.5, after=6)

# ───────────────────────────── 4. contributions ─────────────────────────────
heading("4. Group Contribution and Individual Responsibilities")
para("This is a jointly developed project. All members contributed to planning, technical "
     "discussion, integration review, experiment interpretation and report preparation. "
     "The table shows the main area handled by each member.",
     size=9.5, italic=True, after=4)
contrib = [
    ("Vaibhav Krishna Bansal", "230911546",
     "Led Model 4: ResNet-18 pose model, CBAM integration, training and robustness tests. "
     "Also coordinated model integration, reproducibility work and the interim report."),
    ("Pranav Joshi", "230953248",
     "Led the data pipeline and Model 1: dataset preparation, image and camera-parameter "
     "processing, ORB matching, grid descriptor, feature cache, MLP regressor, tests and "
     "evaluation."),
    ("Abhishek Laxman Jadhav", "230953360",
     "Led Model 2: ResNet-18 with LSTM and GRU sequence models. Also developed pose geometry, "
     "trajectory reconstruction, image projection and related tests."),
    ("Suryansh Bajaj", "230953250",
     "Leads Model 3 and evaluation: self-supervised PoseNet and DepthNet, trajectory "
     "alignment, ATE, RPE, drift and depth metrics, result comparison and tests."),
]
table(["Member Name", "Reg. No.", "Primary and Shared Responsibilities"],
      contrib, [3.7, 2.1, 11.5], font=8)

# ───────────────────────────── 5. risk ─────────────────────────────
risk_heading = heading("5. Risks and Remaining Work")
para("Remaining model. ", bold=True, after=2)
para("Model 3 is the only proposed model that has not yet been completed. Its projection and "
     "warping code is ready, but the depth-image archive is still needed for evaluation. M1 "
     "is complete. Its next steps are robustness testing and reducing forward-motion bias.",
     size=9.5, after=6)

risks = [
    ("Model 3 training", "High impact, moderate likelihood. Photometric "
     "self-supervised training can be unstable. We will start with an ImageNet-pretrained "
     "encoder. If needed, we will use fewer image scales or a lower resolution."),
    ("M1 bias and robustness", "M1 is small and accurate, but its +0.266 m average forward "
     "error accumulates. We will inspect weak matches, tune the bias penalty on validation "
     "data, and test rain, fog and rotated cameras."),
    ("Overfitting", "Validation error stops improving while training loss continues to fall. "
     "The best checkpoints were epoch 27 for M2 and epoch 18 for M4. We will use validation "
     "ATE and early stopping."),
    ("Camera-angle changes", "M4 degrades when the camera is rotated by 30 degrees. We will "
     "add camera-angle variation during training and repeat the test for all models."),
    ("Effect of attention", "M4 differs from M1 and M2 in more than its attention module. "
     "We will compare three otherwise identical M4 models: CBAM, SE attention and no attention."),
    ("Numerical stability", "Mixed precision produced non-finite geometric-loss gradients "
     "on the Kaggle NVIDIA T4 GPU. The current runs use FP32, and training checks the gradients "
     "before each update."),
]
table(["Risk", "Assessment and mitigation"], risks, [4.6, 12.7], font=8.5)

timeline_heading = para("Timeline to Final submission.", bold=True, after=3)
timeline_heading.paragraph_format.space_before = Pt(8)
timeline_heading.paragraph_format.keep_with_next = True
plan = [
    ("2 Oct - 8 Oct", "Finish M1 reporting, add the depth dataset, complete the M3 training "
     "pipeline, and run the remaining M4 attention tests."),
    ("9 Oct - 16 Oct", "Train and evaluate M3. Test M1 under weather and camera changes. Run "
     "the remaining M2 comparisons."),
    ("17 Oct - 23 Oct", "Run the final selected models on the same hardware and prepare the "
     "combined comparison and error analysis."),
    ("24 Oct - 31 Oct", "Prepare the final report, presentation, repository release and viva."),
]
table(["Window", "Planned work"], plan, [3.4, 13.9], font=8.5)

out = "src/reports/interim/ICT4442_Interim_Report_PartB.docx"
doc.save(out)
print("wrote", out)
