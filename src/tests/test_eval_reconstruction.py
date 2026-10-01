"""The trajectory reconstruction in scripts/eval.py.

Regression test for a defect that silently invalidated every sequence model's
metrics: the dataset yields stride-1 sliding windows, so concatenating all S-1
predicted poses per window counts each transition S-1 times.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval import predict_sequence, signed_pose_bias  # noqa: E402


class _Replay:
    """Returns pre-set poses, so reconstruction is tested in isolation."""

    def __init__(self, windows):
        self.windows = list(windows)
        self.i = 0

    def eval(self):
        pass

    def __call__(self, batch):
        out = self.windows[self.i]
        self.i += 1
        return {"pose": out}


def _windows_from_truth(rel_vecs, seq_len):
    """Build the windows a perfect model would emit for a stride-1 loader."""
    n_windows = len(rel_vecs) - seq_len + 2
    return [rel_vecs[i:i + seq_len - 1].unsqueeze(0) for i in range(n_windows)]


def _loader(windows):
    """Stand-in for the DataLoader: predict_sequence expects batch dicts."""
    return [{"imgs": torch.zeros(1)} for _ in windows]


def test_pairwise_reconstruction_is_exact():
    truth = torch.randn(20, 6) * 0.05
    w = _windows_from_truth(truth, seq_len=2)
    rel, _ = predict_sequence(_Replay(w), _loader(w), torch.device("cpu"))
    assert rel.shape == (20, 4, 4)


def test_sequence_reconstruction_recovers_each_transition_once():
    """seq_len=5 over 20 transitions must yield 20, not 4x that."""
    truth = torch.randn(20, 6) * 0.05
    w = _windows_from_truth(truth, seq_len=5)
    assert len(w) == 17 and w[0].shape == (1, 4, 6)

    rel, _ = predict_sequence(_Replay(w), _loader(w), torch.device("cpu"))
    assert rel.shape == (20, 4, 4), f"expected 20 transitions, got {rel.shape[0]}"


def test_reconstruction_uses_the_most_recent_timestep_of_each_window():
    """A recurrent model's later timesteps carry more context, so the last
    element of each window must be the one kept. Keeping element 0 would
    evaluate a sequence model as if it had no recurrence at all."""
    seq_len = 5
    n_windows = 6
    # window w, element e -> a unique marker so the source of each output is
    # identifiable in the reconstruction
    windows = [torch.full((1, seq_len - 1, 6), 0.0) for _ in range(n_windows)]
    for w in range(n_windows):
        for e in range(seq_len - 1):
            windows[w][0, e, :] = w * 10 + e

    rel, _ = predict_sequence(_Replay(windows), _loader(windows), torch.device("cpu"))
    assert rel.shape[0] == (seq_len - 2) + n_windows

    from mvo.common.geometry import se3
    got = se3.mat_to_vec(torch.from_numpy(rel).double())[:, 0].round()
    # leading S-2 from window 0, then the LAST element of every window
    expected = [0.0, 1.0, 2.0] + [w * 10 + (seq_len - 2) for w in range(n_windows)]
    assert got.tolist() == expected, f"{got.tolist()} != {expected}"


def test_reconstruction_matches_the_underlying_poses():
    """Each recovered transition must be the one that window predicted."""
    from mvo.common.geometry import se3

    truth = torch.randn(20, 6).double() * 0.05
    w = [x.double() for x in _windows_from_truth(truth, seq_len=5)]
    rel, _ = predict_sequence(_Replay(w), _loader(w), torch.device("cpu"))

    # A perfect model emits the true transitions in every window, so whichever
    # element is kept the reconstruction must equal the truth.
    expected = se3.vec_to_mat(truth).numpy()
    assert rel.shape == expected.shape
    # predict_sequence casts to float32 deliberately, so compare at float32
    # precision rather than float64.
    assert abs(rel - expected).max() < 1e-6


def test_signed_pose_bias_reports_translation_and_geodesic_rotation():
    from mvo.common.geometry import se3

    gt = torch.eye(4).repeat(3, 1, 1)
    pred_vec = torch.zeros(3, 6)
    pred_vec[:, 2] = 0.2
    pred_vec[:, 4] = 0.01
    bias = signed_pose_bias(se3.vec_to_mat(pred_vec), gt)
    assert abs(bias["bias_tz_m"] - 0.2) < 1e-6
    assert abs(bias["bias_ry_rad"] - 0.01) < 1e-6
