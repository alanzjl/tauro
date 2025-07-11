import abc
from dataclasses import dataclass
from pathlib import Path


@dataclass(kw_only=True)
class RobotConfig(abc.ABC):
    # Allows to distinguish between different robots of the same type
    id: str | None = None
    # Directory to store calibration file
    calibration_dir: Path | None = None

    def __post_init__(self):
        pass
