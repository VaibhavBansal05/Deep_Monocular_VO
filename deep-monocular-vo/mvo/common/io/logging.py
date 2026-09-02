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

    def append(self, row: dict) -> None:
        if self.fields is None:
            self.fields = list(row)
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(
                {k: row.get(k, "") for k in self.fields})
