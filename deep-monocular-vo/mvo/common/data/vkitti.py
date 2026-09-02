"""Virtual KITTI 2 dataset.

OWNER: Pranav Joshi  (branch: base/dataset)
STATUS: STUB -- implement on your branch.

The batch dict below is THE contract. All four models consume exactly this.
Nobody writes their own loader.

    {
      "imgs":      (S, 3, 128, 416) float, ImageNet-normalised
      "K":         (3, 3) float   -- RESCALED to 128x416, not the raw 1242x375 K
      "K_inv":     (3, 3) float
      "rel_pose":  (S-1, 4, 4) float -- GT T_{t->t+1}
      "abs_pose":  (S, 4, 4) float   -- GT camera-to-world
      "depth":     (S, 1, 128, 416) float, metres -- only if need_depth=True
      "seq_id":    str, e.g. "Scene20/clone"
      "frame_idx": (S,) long
    }
"""

from torch.utils.data import Dataset


class VKitti2(Dataset):
    def __init__(self, root: str, split: str, seq_len: int = 2,
                 need_depth: bool = False, augment: bool = False,
                 img_size: tuple[int, int] = (128, 416)):
        raise NotImplementedError("owner: Pranav — branch base/dataset")

    def __len__(self) -> int:
        raise NotImplementedError("owner: Pranav — branch base/dataset")

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError("owner: Pranav — branch base/dataset")
