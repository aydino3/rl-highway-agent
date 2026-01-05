from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class BaseConfig:
    env_id: str = "highway-v0"
    seed: int = 42
    test_steps: int = 10


@dataclass(frozen=True)
class TrainConfig:
    env_id: str = "highway-v0"
    seed: int = 42

    total_timesteps: int = 30_000
    mid_checkpoint_timesteps: int = 150_000
    n_envs: int = 8

    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5

    log_dir: str = "outputs/logs"
    model_dir: str = "outputs/models"
    video_dir: str = "outputs/videos"
