from __future__ import annotations

import argparse
import os
from dataclasses import replace

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from src.config import TrainConfig
from src.envs.make_env import make_training_env


def _auto_steps(total_timesteps: int, n_envs: int, default_n_steps: int) -> int:
    # Rollout size = n_envs * n_steps. If total_timesteps is small, large rollout looks stuck.
    target = max(64, total_timesteps // max(1, n_envs))
    cap = 256
    return min(default_n_steps, target, cap)


def _auto_batch(batch_size: int, n_envs: int, n_steps: int) -> int:
    rollout = max(1, n_envs * n_steps)
    bs = min(batch_size, rollout)
    while rollout % bs != 0 and bs > 1:
        bs //= 2
    return bs


def train(cfg: TrainConfig) -> None:
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.video_dir, exist_ok=True)

    env = make_training_env(cfg.env_id, cfg.n_envs, cfg.seed)

    n_steps = _auto_steps(cfg.total_timesteps, cfg.n_envs, cfg.n_steps)
    batch_size = _auto_batch(cfg.batch_size, cfg.n_envs, n_steps)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg.learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        verbose=1,
        seed=cfg.seed,
    )

    logger = configure(cfg.log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    mid_path = os.path.join(cfg.model_dir, "ppo_mid.zip")
    final_path = os.path.join(cfg.model_dir, "ppo_final.zip")

    model.learn(total_timesteps=cfg.mid_checkpoint_timesteps)
    model.save(mid_path)

    remaining = max(0, cfg.total_timesteps - cfg.mid_checkpoint_timesteps)
    if remaining > 0:
        model.learn(total_timesteps=remaining)
        model.save(final_path)

    env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--mid-timesteps", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig()

    env_id = args.env_id or cfg.env_id
    seed = args.seed if args.seed is not None else cfg.seed
    total_timesteps = (
        args.total_timesteps if args.total_timesteps is not None else cfg.total_timesteps
    )
    mid_checkpoint_timesteps = (
        args.mid_timesteps
        if args.mid_timesteps is not None
        else cfg.mid_checkpoint_timesteps
    )
    n_envs = args.n_envs if args.n_envs is not None else cfg.n_envs

    cfg = replace(
        cfg,
        env_id=env_id,
        seed=seed,
        total_timesteps=total_timesteps,
        mid_checkpoint_timesteps=mid_checkpoint_timesteps,
        n_envs=n_envs,
    )
    train(cfg)


if __name__ == "__main__":
    main()
