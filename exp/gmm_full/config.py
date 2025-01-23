from pathlib import Path
from typing import Literal

import pydantic
from pydantic.dataclasses import dataclass


@dataclass(config=pydantic.ConfigDict(extra="forbid"))
class PSConfig:
    problem: Literal["sc/ls", "ls-/ls+", "sc-/sc+"]
    data_dir_s: str
    region: Literal["13", "14", "15", "all"]
    components: int
    mz_min: float
    mz_max: float
    max_epochs: int
    convergence_threshold: float
    batch_size: int
    lr: float

    wandb_name: str | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "online"

    @pydantic.model_validator(mode="after")
    def validate(self):
        if not self.wandb_project:
            self.wandb_project = "gmm-simple-v1"

        return self

    @property
    def data_dir(self) -> Path:
        path = Path(self.data_dir_s)
        assert path.is_absolute()
        assert path.is_dir()
        assert list(path.iterdir())
        return path
