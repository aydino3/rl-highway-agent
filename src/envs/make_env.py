from __future__ import annotations

import gymnasium as gym
import highway_env  # noqa: F401  (register envs)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

from src.envs.reward_wrapper import CustomRewardWrapper


def make_training_env(env_id: str, n_envs: int, seed: int) -> VecEnv:
    def wrap(env: gym.Env) -> gym.Env:
        return CustomRewardWrapper(env)

    return make_vec_env(env_id, n_envs=n_envs, seed=seed, wrapper_class=wrap)
