from __future__ import annotations

import gymnasium as gym

from src.config import BaseConfig


def main() -> None:
    cfg = BaseConfig()
    env = gym.make(cfg.env_id)
    obs, info = env.reset(seed=cfg.seed)
    print(f"Env: {cfg.env_id}")
    print(f"Obs type: {type(obs)}")
    total_reward = 0.0

    for t in range(cfg.test_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
        print(f"step={t} reward={reward:.3f} done={done}")
        if done:
            break

    print(f"Total reward: {total_reward:.3f}")
    env.close()


if __name__ == "__main__":
    main()
