from pathlib import Path
from typing import Literal

import pydantic
from pydantic.dataclasses import dataclass


@dataclass(config=pydantic.ConfigDict(extra="forbid"))
class PSConfig:
    problem: Literal["sc/ls", "ls-/ls+", "sc-/sc+"]
    data_dir_s: str
    alpha: float
    n_peaks: int
    mz_min: float
    mz_max: float
    norm_type: Literal["batch-norm", "thresh-counter"]
    head_type: Literal["linear", "mlp"]
    epochs: int
    batch_size: int
    lr: float
    c_overlap: float
    c_width: float
    c_weights: float

    wandb_name: str | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "online"

    @pydantic.model_validator(mode="after")
    def validate(self):
        if not self.wandb_project:
            self.wandb_project = "peak-sense-v1"

        return self

    @property
    def data_dir(self) -> Path:
        path = Path(self.data_dir_s)
        assert path.is_absolute()
        assert path.is_dir()
        assert list(path.iterdir())
        return path
