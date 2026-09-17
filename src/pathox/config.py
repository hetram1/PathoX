from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    root: Path
    panda: Path
    processed: Path
    experiments: Path

    @classmethod
    def from_environment(cls) -> "DataPaths":
        root = Path(
            os.environ.get(
                "PATHOX_DATA_ROOT",
                "/mnt/d/PathoXData",
            )
        ).expanduser()

        return cls(
            root=root,
            panda=root / "panda",
            processed=root / "processed",
            experiments=root / "experiments",
        )

    def create(self) -> None:
        self.panda.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.processed.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.experiments.mkdir(
            parents=True,
            exist_ok=True,
        )
