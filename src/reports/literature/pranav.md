# Paper summaries — Pranav Joshi

Both papers keep classical geometry and add learning on top — M1's design question.

---

## [6] Learning Kalman Network: A Deep Monocular VO for On-Road Driving — Zhao, Sun, Yan, Neumann, Duckett, Stolkin (Birmingham / Oxford / Lincoln), *Robotics and Autonomous Systems* 121:103234, 2019

**Method.** A hybrid that keeps the Kalman filter's probabilistic structure but learns its
components. Most learning-based VO compares only the two most recent frames; LKN-VO adds a
**discriminative state estimator** over a sequence of previous measurements. Deep networks
supply the non-linear observation and transition models, while the state update follows the
standard Kalman mechanism. A sparse representation captures correlations in the state.

**Dataset.** KITTI.

**Key result.** Learns a more accurate **absolute scale without geometric constraints**,
which model-based monocular VO cannot do. Against a classical Kalman filter, the optimal
parameters are learned from data rather than hand-designed — the paper's central argument.

**Relevance.** The clearest statement of the thesis M1 embodies: classical structure plus
learned components can beat either alone. M1 is the simpler instance — classical
correspondences, learned regression.

**The comparison M1 should make.** LKN-VO's argument is that learning replaces *manual
parameter design*, not the geometry. That is exactly the ablation worth running: classical
ORB/Lucas-Kanade correspondences fed to a 5-point essential-matrix solver with
ground-truth scale, versus the same correspondences fed to our MLP. The difference is
what learning adds on top of geometry, and it is the likeliest viva question.

---

## [9] Reinforcement Learning Meets Visual Odometry — Messikommer et al. (UZH Robotics & Perception Group), ECCV 2024

**Method.** Reframes VO as sequential decision-making. Rather than replacing the pipeline,
an RL agent sits **inside a classical VO system** and makes online choices — keyframe
selection, grid size — from real-time observations of keypoints, map statistics and prior
poses. The reward combines pose error and runtime. Code at `github.com/uzhrpg/rl_vo`.

**Dataset.** Public VO benchmarks with classical VO front-ends.

**Key result.** Improved accuracy and robustness, and the motivating claim: existing VO
relies on heuristics needing **weeks of expert hyperparameter tuning**, which harms
generalisation. The agent removes that tuning.

**Relevance.** The strongest published argument that M1's family is not merely a baseline.
A 2024 ECCV paper deliberately *retaining* the classical pipeline is a good counter to the
assumption that end-to-end supersedes classical VO — useful framing for M1's section, which
otherwise reads as the weakest model by construction.

**Positioning M1 honestly.** M1 is the classical/MLP baseline our brief requires, and it
will likely place last on accuracy. Its value is diagnostic: it establishes what
hand-engineered correspondences achieve, so the three deep models are measured against
something real rather than against each other alone. Say that plainly in the report.

**Licence note.** `rl_vo` is GPLv3. Do not vendor it into this repository — invoke as a
command-line tool if used at all. See `THIRD_PARTY.md`.
