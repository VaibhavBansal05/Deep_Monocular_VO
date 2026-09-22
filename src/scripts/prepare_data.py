#!/usr/bin/env python3
"""Download the Virtual KITTI 2 subset defined in ``splits.py``.

The official archives are large and monolithic. This script streams each archive
directly into ``tar`` and extracts only the sequences and camera used by the
project, so the complete archives never occupy disk space.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from mvo.common.data import splits

BASE_URL = "https://download.europe.naverlabs.com/virtual_kitti_2.0.3"
ARCHIVES = {
    "rgb": f"{BASE_URL}/vkitti_2.0.3_rgb.tar",
    "depth": f"{BASE_URL}/vkitti_2.0.3_depth.tar",
    "textgt": f"{BASE_URL}/vkitti_2.0.3_textgt.tar.gz",
}


def needed_sequences() -> list[str]:
    """Return every unique scene/variation pair used by any project split."""
    sequences = {
        splits.seq_id(scene, variation)
        for split in splits.SPLITS
        for scene, variation in splits.sequences(split)
    }
    return sorted(sequences)


def stream_extract(
    url: str,
    root: Path,
    patterns: list[str],
    *,
    gzipped: bool = False,
) -> None:
    """Stream *url* into tar and extract only members matching *patterns*.

    ``curl`` supplies the stream and Python's streaming ``tarfile`` reader avoids
    requiring GNU tar (Windows ships bsdtar, which has no ``--wildcards`` option).
    The complete archive is never stored on disk. Both subprocesses and archive
    extraction are checked so a failure cannot look like success.
    """
    root.mkdir(parents=True, exist_ok=True)
    curl_command = [
        "curl",
        "--location",
        "--fail",
        "--retry",
        "3",
        "--retry-all-errors",
        url,
    ]
    print(f"[prepare] downloading {url}")
    print(f"[prepare] extracting {len(patterns)} member patterns into {root}")

    with subprocess.Popen(curl_command, stdout=subprocess.PIPE) as download:
        if download.stdout is None:  # pragma: no cover - defensive subprocess guard
            raise RuntimeError("curl stdout pipe was not created")
        try:
            mode = "r|gz" if gzipped else "r|"
            with tarfile.open(fileobj=download.stdout, mode=mode) as archive:
                matched = 0
                for member in archive:
                    if not any(fnmatch.fnmatchcase(member.name, pattern) for pattern in patterns):
                        continue
                    _extract_member(archive, member, root)
                    matched += 1
        finally:
            download.stdout.close()

        curl_status = download.wait()

    if curl_status != 0:
        raise subprocess.CalledProcessError(curl_status, curl_command)
    if matched == 0:
        raise RuntimeError(f"no archive members matched patterns for {url}")


def _extract_member(archive: tarfile.TarFile, member: tarfile.TarInfo, root: Path) -> None:
    """Extract one regular file or directory without allowing path traversal."""
    member_path = Path(member.name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise RuntimeError(f"unsafe archive member path: {member.name!r}")
    if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
        raise RuntimeError(f"unsupported archive member: {member.name!r}")

    destination = (root / member_path).resolve()
    root_resolved = root.resolve()
    if os.path.commonpath((root_resolved, destination)) != str(root_resolved):
        raise RuntimeError(f"archive member escapes destination: {member.name!r}")

    if member.isdir():
        destination.mkdir(parents=True, exist_ok=True)
        return

    source = archive.extractfile(member)
    if source is None:  # pragma: no cover - defensive tarfile guard
        raise RuntimeError(f"could not read archive member: {member.name!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream and extract the vKITTI 2 subset used by this project."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/vkitti2"),
        help="destination directory (default: data/vkitti2)",
    )
    parser.add_argument(
        "--skip-depth",
        action="store_true",
        help="skip depth maps, which are only needed for M3 evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sequences = needed_sequences()
    print(f"[prepare] selected {len(sequences)} sequences: {sequences}")

    stream_extract(
        ARCHIVES["textgt"],
        args.root,
        [f"{sequence}/*" for sequence in sequences],
        gzipped=True,
    )
    stream_extract(
        ARCHIVES["rgb"],
        args.root,
        [f"{sequence}/frames/rgb/{splits.CAMERA}/*" for sequence in sequences],
    )

    if not args.skip_depth:
        evaluation_sequences = sorted(
            {
                splits.seq_id(scene, variation)
                for split in ("val", "test")
                for scene, variation in splits.sequences(split)
            }
        )
        stream_extract(
            ARCHIVES["depth"],
            args.root,
            [
                f"{sequence}/frames/depth/{splits.CAMERA}/*"
                for sequence in evaluation_sequences
            ],
        )

    print(f"[prepare] done -> {args.root}")


if __name__ == "__main__":
    main()
