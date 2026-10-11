"""Console + CSV logging."""

import csv
import logging
from pathlib import Path

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S")
        _CONFIGURED = True
    return logging.getLogger(name)


class CSVLogger:
    """Append-only CSV of per-epoch metrics. Commit these -- they are evidence."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields: list[str] | None = None
        self._last_epoch: int | None = None

    def resume(self, epoch: int, expected_fields: list[str] | None = None) -> None:
        """Preserve the existing header/history when resuming a checkpoint.

        A checkpoint older than the history requires a new output directory;
        appending it here would duplicate epochs and corrupt the evidence trail.
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            self.fields = None
            self._last_epoch = None
            return
        try:
            with self.path.open(newline="") as f:
                reader = csv.DictReader(f, strict=True)
                fields = reader.fieldnames
                if not fields or "epoch" not in fields or len(set(fields)) != len(fields):
                    raise ValueError("malformed history header; use a new output directory")
                if expected_fields is not None and set(fields) != set(expected_fields):
                    raise ValueError("history schema changed; use a new output directory")
                last_epoch = None
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError("malformed history row; use a new output directory")
                    try:
                        row_epoch = int(row["epoch"])
                    except ValueError as exc:
                        raise ValueError("malformed history epoch; use a new output directory") from exc
                    if row_epoch < 0 or (last_epoch is not None and row_epoch <= last_epoch):
                        raise ValueError("history epochs must increase without duplicates; use a new output directory")
                    if row_epoch > epoch:
                        raise ValueError("history is newer than checkpoint; resume into a new output directory")
                    last_epoch = row_epoch
        except csv.Error as exc:
            raise ValueError("malformed history CSV; use a new output directory") from exc
        self.fields = fields
        self._last_epoch = last_epoch

    def append(self, row: dict) -> None:
        if self.fields is not None and set(row) != set(self.fields):
            raise ValueError("history schema changed; use a new output directory")
        row_epoch = int(row["epoch"]) if "epoch" in row else None
        if (row_epoch is not None and self._last_epoch is not None
                and row_epoch <= self._last_epoch):
            raise ValueError("history epochs must increase without duplicates")
        if self.fields is None:
            self.fields = list(row)
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(
                {k: row.get(k, "") for k in self.fields})
        self._last_epoch = row_epoch
