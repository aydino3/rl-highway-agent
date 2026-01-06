from __future__ import annotations

import gymnasium as gym

from src.config import BaseConfig
from src.envs.reward_wrapper import CustomRewardWrapper


def main() -> None:
    cfg = BaseConfig()
    env = gym.make(cfg.env_id)
    env = CustomRewardWrapper(env)
    obs, info = env.reset(seed=cfg.seed)
    total_reward = 0.0

    for t in range(cfg.test_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
        print(f"step={t} custom_reward={reward:.3f} crashed={info.get('crashed', False)}")
        if done:
            break

    print(f"Total custom reward: {total_reward:.3f}")
    env.close()


if __name__ == "__main__":
    main()
