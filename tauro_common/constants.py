"""Common constants used across all Tauro modules."""

from pathlib import Path

# Observation and action keys
OBS_ENV_STATE = "observation.environment_state"
OBS_STATE = "observation.state"
OBS_IMAGE = "observation.image"
OBS_IMAGES = "observation.images"
ACTION = "action"
REWARD = "next.reward"

# Robot and teleoperator keys
ROBOTS = "robots"
ROBOT_TYPE = "robot_type"
TELEOPERATORS = "teleoperators"

# Files & directories
CHECKPOINTS_DIR = "checkpoints"
LAST_CHECKPOINT_LINK = "last"
PRETRAINED_MODEL_DIR = "pretrained_model"
TRAINING_STATE_DIR = "training_state"
RNG_STATE = "rng_state.safetensors"
TRAINING_STEP = "training_step.json"
OPTIMIZER_STATE = "optimizer_state.safetensors"
OPTIMIZER_PARAM_GROUPS = "optimizer_param_groups.json"
SCHEDULER_STATE = "scheduler_state.json"

# Cache directory
default_cache_path = Path.home() / ".cache" / "tauro"

# Calibration directory
default_calibration_path = default_cache_path / "calibration"

# gRPC settings
DEFAULT_GRPC_PORT = 50051
DEFAULT_GRPC_HOST = "localhost"
MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10MB
KEEPALIVE_TIME_MS = 10000  # 10 seconds
