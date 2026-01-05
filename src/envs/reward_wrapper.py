from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np


class CustomRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        w_speed: float = 1.0,
        w_crash: float = 5.0,
        w_alive: float = 0.05,
    ) -> None:
        super().__init__(env)
        self.w_speed = w_speed
        self.w_crash = w_crash
        self.w_alive = w_alive

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        crashed = bool(info.get("crashed", False))
        speed = float(info.get("speed", 0.0))

        r_speed = speed / 30.0
        r_crash = -1.0 if crashed else 0.0
        r_alive = 1.0

        custom_reward = (
            (self.w_speed * r_speed)
            + (self.w_crash * r_crash)
            + (self.w_alive * r_alive)
        )
        return obs, float(custom_reward), terminated, truncated, info
