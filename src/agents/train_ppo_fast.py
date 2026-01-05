from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from src.config import TrainConfig


class CustomRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        w_speed: float,
        w_right_lane: float,
        w_collision: float,
        w_lane_change: float,
        v_min: float,
        v_max: float,
    ) -> None:
        super().__init__(env)
        self.w_speed = w_speed
        self.w_right_lane = w_right_lane
        self.w_collision = w_collision
        self.w_lane_change = w_lane_change
        self.v_min = v_min
        self.v_max = v_max
        self._prev_lane_index: int | None = None

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._prev_lane_index = self._get_lane_index()
        return obs, info

    def step(self, action: Any):
        obs, _r, terminated, truncated, info = self.env.step(action)

        crashed = bool(info.get("crashed", False))
        collision = 1.0 if crashed else 0.0

        speed = float(getattr(self.env.unwrapped.vehicle, "speed", 0.0))
        speed_norm = (speed - self.v_min) / max(1e-6, (self.v_max - self.v_min))
        speed_norm = float(np.clip(speed_norm, 0.0, 1.0))

        lane_index = self._get_lane_index()
        right_lane_ratio = self._right_lane_ratio(lane_index)

        lane_change = 0.0
        if self._prev_lane_index is not None and lane_index is not None:
            lane_change = 1.0 if lane_index != self._prev_lane_index else 0.0
        self._prev_lane_index = lane_index

        shaped = (
            self.w_speed * speed_norm
            + self.w_right_lane * right_lane_ratio
            - self.w_lane_change * lane_change
            - self.w_collision * collision
        )
        shaped = float(np.clip(shaped, 0.0, 1.0))
        info["custom_reward"] = shaped
        return obs, shaped, terminated, truncated, info

    def _get_lane_index(self) -> int | None:
        try:
            li = getattr(self.env.unwrapped.vehicle, "lane_index", None)
            if li is None:
                return None
            return int(li[2]) if len(li) >= 3 else int(li[1])
        except Exception:
            return None

    def _right_lane_ratio(self, lane_index: int | None) -> float:
        try:
            lanes_count = int(getattr(self.env.unwrapped, "config", {}).get("lanes_count", 4))
        except Exception:
            lanes_count = 4
        if lane_index is None:
            return 0.0
        return float(np.clip(lane_index / max(1, lanes_count - 1), 0.0, 1.0))


def make_wrapped_env(cfg: TrainConfig) -> gym.Env:
    import highway_env  # noqa: F401
    env = gym.make(getattr(cfg, "env_id", "highway-fast-v0"), render_mode=None)
    env = CustomRewardWrapper(
        env,
        w_speed=getattr(cfg, "w_speed", 0.6),
        w_right_lane=getattr(cfg, "w_right_lane", 0.3),
        w_collision=getattr(cfg, "w_collision", 1.0),
        w_lane_change=getattr(cfg, "w_lane_change", 0.1),
        v_min=getattr(cfg, "v_min", 20.0),
        v_max=getattr(cfg, "v_max", 30.0),
    )
    return env


def main() -> None:
    cfg = TrainConfig()

    # Fallback paths (do NOT depend on cfg.*_dir fields)
    outputs_dir = Path("outputs")
    models_dir = outputs_dir / "models"
    logs_dir = outputs_dir / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    n_envs = int(getattr(cfg, "n_envs", 8))
    n_steps = int(getattr(cfg, "n_steps", 512))
    batch_size = int(getattr(cfg, "batch_size", 256))
    total_timesteps = int(getattr(cfg, "total_timesteps", 30_000))
    seed = int(getattr(cfg, "seed", 42))

    vec_env = make_vec_env(
        lambda: make_wrapped_env(cfg),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )

    vec_env = VecMonitor(vec_env, filename=str(logs_dir / "monitor.csv"))

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=float(getattr(cfg, "learning_rate", 3e-4)),
        gamma=float(getattr(cfg, "gamma", 0.99)),
        gae_lambda=float(getattr(cfg, "gae_lambda", 0.95)),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=int(getattr(cfg, "n_epochs", 10)),
        clip_range=float(getattr(cfg, "clip_range", 0.2)),
        ent_coef=float(getattr(cfg, "ent_coef", 0.0)),
        vf_coef=float(getattr(cfg, "vf_coef", 0.5)),
        max_grad_norm=float(getattr(cfg, "max_grad_norm", 0.5)),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device="cpu",
    )

    half = total_timesteps // 2
    model.learn(total_timesteps=half, progress_bar=True)
    model.save(str(models_dir / "ppo_half"))

    model.learn(total_timesteps=total_timesteps - half, reset_num_timesteps=False, progress_bar=True)
    model.save(str(models_dir / "ppo_final"))

    vec_env.close()
    print("Saved models:", models_dir / "ppo_half.zip", models_dir / "ppo_final.zip")
    print("Monitor log:", logs_dir / "monitor.csv")


if __name__ == "__main__":
    main()
