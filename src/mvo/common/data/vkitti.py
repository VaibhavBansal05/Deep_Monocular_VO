"""Virtual KITTI 2 dataset. See the batch-dict contract below.

OWNER: Pranav Joshi

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

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from mvo.common.data import splits
from mvo.common.data.transforms import color_jitter, hflip_pose, normalize, rescale_intrinsics


def _read_extrinsics(path) -> dict[int, np.ndarray]:
    """extrinsic.txt -> {frame: 4x4 world-to-camera}. Camera_0 rows only."""
    poses = {}
    with open(path) as f:
        next(f)  # header
        for line in f:
            parts = line.split()
            frame, cam = int(parts[0]), int(parts[1])
            if cam != 0:
                continue
            poses[frame] = np.array(parts[2:18], dtype=np.float32).reshape(4, 4)
    return poses


def _read_intrinsics(path) -> np.ndarray:
    """intrinsic.txt -> 3x3 K for Camera_0 (constant across frames)."""
    with open(path) as f:
        next(f)
        parts = f.readline().split()
    fx, fy, cx, cy = (float(x) for x in parts[2:6])
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


class VKitti2(Dataset):
    def __init__(self, root, split, seq_len=2, need_depth=False,
                 augment=False, img_size=(128, 416), seed=42):
        self.root = Path(root)
        self.split = split
        self.seq_len = seq_len
        self.need_depth = need_depth
        self.augment = augment
        self.img_size = img_size
        self.seed = int(seed)
        self.epoch = 0

        self.samples = []          # (seq_id, start_frame)
        self.meta = {}             # seq_id -> {"K", "extrinsics", "dir"}

        for scene, variation in splits.sequences(split):
            sid = splits.seq_id(scene, variation)
            d = self.root / scene / variation
            rgb_dir = d / "frames" / "rgb" / splits.CAMERA
            if not rgb_dir.is_dir():
                raise FileNotFoundError(f"missing {rgb_dir} — run scripts/prepare_data.py")

            ext = _read_extrinsics(d / "extrinsic.txt")
            K = rescale_intrinsics(_read_intrinsics(d / "intrinsic.txt"), img_size)
            n = len(sorted(rgb_dir.glob("rgb_*.jpg")))

            self.meta[sid] = {"K": K, "ext": ext, "dir": d, "n": n}
            self.samples += [(sid, i) for i in range(n - seq_len + 1)]

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        """Choose the deterministic augmentation stream for an epoch.

        The tuple (base seed, epoch, sample index) is stable across reruns and
        independent of DataLoader worker scheduling, but changes every epoch.
        """
        self.epoch = int(epoch)

    def _load_rgb(self, sid, frame) -> torch.Tensor:
        p = self.meta[sid]["dir"] / "frames" / "rgb" / splits.CAMERA / f"rgb_{frame:05d}.jpg"
        img = Image.open(p).convert("RGB").resize(
            (self.img_size[1], self.img_size[0]), Image.BILINEAR)
        return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

    def _load_depth(self, sid, frame) -> torch.Tensor:
        """16-bit PNG in CENTIMETRES -> metres. 65535 is sky/invalid."""
        p = self.meta[sid]["dir"] / "frames" / "depth" / splits.CAMERA / f"depth_{frame:05d}.png"
        d = np.asarray(Image.open(p).resize(
            (self.img_size[1], self.img_size[0]), Image.NEAREST)).astype(np.float32)
        d = d / 100.0
        d[d >= 655.0] = 0.0        # 0 = invalid, masked out in the loss
        return torch.from_numpy(d).unsqueeze(0)

    def absolute_poses(self, sid) -> np.ndarray:
        """(N,4,4) GT camera-to-world for a whole sequence. Used by scripts/eval.py."""
        ext = self.meta[sid]["ext"]
        return np.stack([np.linalg.inv(ext[i]) for i in sorted(ext)], axis=0)

    def __getitem__(self, idx) -> dict:
        sid, start = self.samples[idx]
        m = self.meta[sid]
        frames = list(range(start, start + self.seq_len))

        imgs = torch.stack([self._load_rgb(sid, f) for f in frames])

        # extrinsic.txt is world->camera. Point in cam frame t maps to t+1 by
        #     T_{t->t+1} = E_{t+1} @ inv(E_t)
        # and camera-to-world is inv(E_t).
        E = [m["ext"][f] for f in frames]
        abs_pose = np.stack([np.linalg.inv(e) for e in E])
        rel_pose = np.stack([E[i + 1] @ np.linalg.inv(E[i])
                             for i in range(len(E) - 1)])

        K = m["K"].copy()

        flipped = False
        if self.augment:
            # Reproducible but not frozen: the same sample has the same
            # transform within an epoch and a different stream next epoch.
            # Deriving it from idx also makes results independent of which
            # DataLoader worker happened to load the sample.
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch, int(idx)]))
            imgs = color_jitter(imgs, rng)
            if rng.random() < 0.5:
                flipped = True
                imgs = torch.flip(imgs, dims=[-1])
                # torch.flip maps pixel u -> W-1-u, so the mirrored
                # principal point is (W-1) - cx, not W - cx. One pixel, but it
                # is a constant bias in K that a photometric loss cannot correct.
                K[0, 2] = (self.img_size[1] - 1) - K[0, 2]
                from mvo.common.geometry import se3  # Abhishek's module
                vec = se3.mat_to_vec(torch.from_numpy(rel_pose)).numpy()
                vec = np.stack([hflip_pose(v) for v in vec])
                rel_pose = se3.vec_to_mat(torch.from_numpy(vec)).numpy()

        if flipped:
            # A horizontal flip mirrors the camera frame, so the stored
            # world-frame absolute poses stop agreeing with the mirrored
            # rel_pose. Re-integrate them from the mirrored chain, anchored on
            # the unchanged first pose, so the two stay consistent.
            #
            # Dropping the key instead would break collation: flipped and
            # unflipped samples would carry different keys and default_collate
            # raises KeyError on the first mixed batch.
            abs_pose = abs_pose.copy()
            for i in range(len(rel_pose)):
                abs_pose[i + 1] = abs_pose[i] @ np.linalg.inv(rel_pose[i])

        out = {
            "imgs": normalize(imgs),
            "K": torch.from_numpy(K),
            "K_inv": torch.from_numpy(np.linalg.inv(K)),
            "rel_pose": torch.from_numpy(rel_pose).float(),
            "abs_pose": torch.from_numpy(abs_pose).float(),
            "seq_id": sid,
            "frame_idx": torch.tensor(frames, dtype=torch.long),
        }

        if self.need_depth:
            out["depth"] = torch.stack([self._load_depth(sid, f) for f in frames])
        return out
